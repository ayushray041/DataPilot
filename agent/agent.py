"""Controlled Gemini-powered question answering for DataPilot datasets."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from google.genai import types
from google.genai.errors import APIError

from agent.gemini_tools import build_gemini_tools
from agent.llm import get_client, get_model_name
from agent.tool_registry import TOOL_REGISTRY, build_tool_context


_MAX_TOOL_ROUNDS = 5
_MAX_RESULT_CHARS = 10_000
_MAX_DATAFRAME_ROWS = 20
_ARGUMENT_TYPES = {
    "column": str,
    "column1": str,
    "column2": str,
    "category_column": str,
    "value_column": str,
    "x_column": str,
    "y_column": str,
    "operator": str,
    "aggregation": str,
    "value": str,
    "descending": bool,
    "limit": int,
    "contamination": (int, float),
    "columns": list,
}


def _to_json_safe(value: Any) -> Any:
    """Convert tool outputs to simple JSON-compatible values."""
    if isinstance(value, pd.DataFrame):
        row_count = len(value)
        rows = [
            {
                str(column): _to_json_safe(value.iloc[row_index, column_index])
                for column_index, column in enumerate(value.columns)
            }
            for row_index in range(min(row_count, _MAX_DATAFRAME_ROWS))
        ]
        return {
            "type": "dataframe",
            "columns": [str(column) for column in value.columns],
            "row_count": row_count,
            "rows": rows,
            "truncated": row_count > _MAX_DATAFRAME_ROWS,
        }
    if isinstance(value, pd.Series):
        return [_to_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _to_json_safe(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _bounded_tool_result(result: Any) -> dict[str, Any]:
    """Return a serializable tool response small enough for a model turn."""
    safe_result = _to_json_safe(result)
    encoded_result = json.dumps(safe_result, ensure_ascii=False, allow_nan=False)
    if len(encoded_result) <= _MAX_RESULT_CHARS:
        return {"result": safe_result}

    # Keep shrinking the preview until the complete JSON envelope fits the cap.
    preview = encoded_result[:_MAX_RESULT_CHARS]
    payload = {}
    while preview:
        payload = {
            "result_truncated": True,
            "preview": preview,
            "message": "The tool result was shortened because it was too large.",
        }
        if len(json.dumps(payload, ensure_ascii=False)) <= _MAX_RESULT_CHARS:
            break
        preview = preview[: len(preview) // 2]
    return payload


class DataAgent:
    """Answer questions about one in-memory DataFrame using registered tools."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Validate and store a dataset, then configure the Gemini client.

        Args:
            df: A non-empty DataFrame that tools can use to answer questions.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
            ValueError: If ``df`` has no rows or no columns.
            RuntimeError: If Gemini client configuration is missing.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty or len(df.columns) == 0:
            raise ValueError("df must be a non-empty DataFrame with rows and columns")

        self.df = df
        self.client = get_client()
        self.model = get_model_name()
        self.last_tool_results: list[dict[str, Any]] = []

    def ask(self, question: str) -> str:
        """Answer a natural-language question with up to five tool rounds.

        Gemini may request one or more registered functions per round. Only
        those named functions are dispatched; generated code is never run.

        Args:
            question: A non-empty natural-language question.

        Returns:
            Gemini's final text answer, or a helpful quota/round-limit message.

        Raises:
            TypeError: If ``question`` is not a string.
            ValueError: If ``question`` is empty or only whitespace.
            APIError: For non-quota Gemini API errors.
        """
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if not question.strip():
            raise ValueError("question must not be empty")

        self.last_tool_results = []
        dataset_context = build_tool_context(self.df)
        user_message = (
            "Use the dataset information and available tools below to answer "
            "the user's question. Call tools when needed. Do not invent facts "
            "that are not supported by the dataset or tool results. Do not "
            "assume currency symbols or units that the data does not specify.\n\n"
            f"Dataset and tool information:\n"
            f"{json.dumps(dataset_context, ensure_ascii=False)}\n\n"
            f"User question: {question.strip()}"
        )
        conversation: list[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=user_message)])
        ]
        config = types.GenerateContentConfig(
            tools=build_gemini_tools(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        for round_number in range(_MAX_TOOL_ROUNDS + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=conversation,
                    config=config,
                )
            except APIError as error:
                if self._is_quota_error(error):
                    return (
                        "Gemini's request quota or rate limit has been reached. "
                        "Please wait and try again, or check the Gemini API "
                        "project's quota and billing settings."
                    )
                raise

            model_content = self._get_model_content(response)
            if model_content is None:
                return response.text or "Gemini returned no response."

            # Keep the exact model message so its function calls stay in context.
            conversation.append(model_content)
            function_calls = [
                part.function_call
                for part in (model_content.parts or [])
                if part.function_call is not None
            ]
            if not function_calls:
                return response.text or self._text_from_content(model_content)

            if round_number == _MAX_TOOL_ROUNDS:
                return (
                    "I stopped because this question needed more than five "
                    "tool-execution rounds. Please ask a more focused question."
                )

            function_response_parts = []
            for function_call in function_calls:
                tool_name = function_call.name or ""
                arguments = function_call.args or {}
                payload = self._execute_registered_tool(tool_name, arguments)
                function_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=function_call.id,
                            name=tool_name or "unknown_tool",
                            response=payload,
                        )
                    )
                )

            # Gemini receives structured function results as a user message.
            conversation.append(
                types.Content(role="user", parts=function_response_parts)
            )

        return "I could not complete the request within the allowed tool rounds."

    def _execute_registered_tool(
        self, tool_name: str, arguments: Any
    ) -> dict[str, Any]:
        """Validate and run one registry function without evaluating code."""
        entry = TOOL_REGISTRY.get(tool_name)
        if entry is None:
            error_result = {"error": f"Unknown tool {tool_name!r}."}
            self.last_tool_results.append(
                {"tool": tool_name or "unknown_tool", "result": error_result}
            )
            return error_result

        if not isinstance(arguments, Mapping):
            error_result = {"error": "Tool arguments must be an object."}
            self.last_tool_results.append({"tool": tool_name, "result": error_result})
            return error_result

        allowed_arguments = entry["arguments"]
        unexpected_arguments = set(arguments) - set(allowed_arguments)
        if unexpected_arguments:
            names = ", ".join(sorted(str(name) for name in unexpected_arguments))
            error_result = {
                "error": f"Unexpected argument(s) for {tool_name!r}: {names}."
            }
            self.last_tool_results.append({"tool": tool_name, "result": error_result})
            return error_result

        missing_arguments = set(entry["required"]) - set(arguments)
        if missing_arguments:
            names = ", ".join(sorted(missing_arguments))
            error_result = {
                "error": f"Missing required argument(s) for {tool_name!r}: {names}."
            }
            self.last_tool_results.append({"tool": tool_name, "result": error_result})
            return error_result

        for argument_name, argument_value in arguments.items():
            expected_type = _ARGUMENT_TYPES[argument_name]
            if argument_name == "limit":
                valid_type = isinstance(argument_value, int) and not isinstance(
                    argument_value, bool
                )
            elif argument_name == "contamination":
                valid_type = isinstance(argument_value, (int, float)) and not isinstance(
                    argument_value, bool
                )
            elif argument_name == "columns":
                valid_type = isinstance(argument_value, list) and all(
                    isinstance(column, str) for column in argument_value
                )
            else:
                valid_type = isinstance(argument_value, expected_type)

            if not valid_type:
                error_result = {
                    "error": (
                        f"Argument {argument_name!r} for tool {tool_name!r} "
                        "has an invalid type."
                    )
                }
                self.last_tool_results.append(
                    {"tool": tool_name, "result": error_result}
                )
                return error_result

        try:
            # Only functions imported into TOOL_REGISTRY can be executed.
            result = entry["function"](self.df, **dict(arguments))
        except Exception as error:
            error_result = {
                "error": f"Tool {tool_name!r} failed: {type(error).__name__}: {error}"
            }
            self.last_tool_results.append({"tool": tool_name, "result": error_result})
            return _bounded_tool_result(error_result)

        self.last_tool_results.append({"tool": tool_name, "result": result})
        return _bounded_tool_result(result)

    @staticmethod
    def _get_model_content(response: Any) -> types.Content | None:
        """Read the first candidate's content from an SDK response."""
        candidates = response.candidates or []
        if not candidates:
            return None
        return candidates[0].content

    @staticmethod
    def _text_from_content(content: types.Content) -> str:
        """Join text parts when the response does not expose response.text."""
        return "\n".join(
            part.text for part in (content.parts or []) if part.text is not None
        )

    @staticmethod
    def _is_quota_error(error: APIError) -> bool:
        """Recognize quota and rate-limit errors from the SDK."""
        message = f"{error.status or ''} {error.message or ''}".lower()
        return error.code == 429 or any(
            phrase in message
            for phrase in ("quota", "rate limit", "resource exhausted")
        )