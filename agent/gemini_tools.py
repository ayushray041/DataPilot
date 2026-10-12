"""Convert DataPilot's tool registry into Gemini function declarations."""

import re
from collections.abc import Mapping

from google.genai import types

from agent.tool_registry import TOOL_REGISTRY


_ARGUMENT_TYPES = {
    "column": "STRING",
    "column1": "STRING",
    "column2": "STRING",
    "category_column": "STRING",
    "value_column": "STRING",
    "x_column": "STRING",
    "y_column": "STRING",
    "operator": "STRING",
    "aggregation": "STRING",
    # Numeric values may be passed as strings and converted by filter_data.
    "value": "STRING",
    "descending": "BOOLEAN",
    "limit": "INTEGER",
    "contamination": "NUMBER",
    "columns": "ARRAY",
}

_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_registry_entry(name: object, entry: object) -> None:
    """Raise ValueError when registry metadata cannot form a tool declaration."""
    if not isinstance(name, str) or not _VALID_NAME.fullmatch(name):
        raise ValueError(f"Invalid tool name: {name!r}")
    if not isinstance(entry, Mapping):
        raise ValueError(f"Registry entry for {name!r} must be a dictionary")
    if not callable(entry.get("function")):
        raise ValueError(f"Registry entry for {name!r} must include a callable function")

    description = entry.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Tool {name!r} must have a non-empty description")

    arguments = entry.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError(f"Tool {name!r} arguments must be a dictionary")
    for argument_name, argument_description in arguments.items():
        if not isinstance(argument_name, str) or not _VALID_NAME.fullmatch(argument_name):
            raise ValueError(f"Tool {name!r} has an invalid argument name: {argument_name!r}")
        if argument_name in ("df", "output_path"):
            raise ValueError(
                f"Tool {name!r} must not expose {argument_name!r} to Gemini"
            )
        if argument_name not in _ARGUMENT_TYPES:
            raise ValueError(
                f"Tool {name!r} has unsupported argument {argument_name!r}"
            )
        if not isinstance(argument_description, str) or not argument_description.strip():
            raise ValueError(
                f"Argument {argument_name!r} for tool {name!r} "
                "must have a non-empty description"
            )

    required = entry.get("required")
    if not isinstance(required, list):
        raise ValueError(f"Tool {name!r} required arguments must be a list")
    for argument_name in required:
        if not isinstance(argument_name, str) or argument_name not in arguments:
            raise ValueError(
                f"Tool {name!r} has an invalid required argument: {argument_name!r}"
            )
    if len(required) != len(set(required)):
        raise ValueError(f"Tool {name!r} required arguments must not contain duplicates")


def build_gemini_tools() -> list[types.Tool]:
    """Build Gemini tools from DataPilot's registered Python functions.

    Each registry entry becomes one Gemini function declaration. Argument
    descriptions and required status are copied directly from the registry,
    while argument types are mapped to JSON Schema types Gemini understands.

    Returns:
        A one-item list containing a Gemini ``Tool`` with every registered
        function declaration.

    Raises:
        ValueError: If registry metadata is missing, invalid, or contains an
            argument whose schema type is not supported here.
    """
    declarations = []

    for name, entry in TOOL_REGISTRY.items():
        _validate_registry_entry(name, entry)
        properties = {}
        for argument_name, description in entry["arguments"].items():
            argument_schema = {
                "type": _ARGUMENT_TYPES[argument_name],
                "description": description,
            }
            if argument_name == "columns":
                # Gemini's array schema needs the type of each list item.
                argument_schema["items"] = {"type": "STRING"}
            properties[argument_name] = argument_schema

        # Keep optional arguments out of this list so Gemini knows they can
        # be omitted and Python defaults can take effect.
        parameters_schema = {
            "type": "OBJECT",
            "properties": properties,
            "required": list(entry["required"]),
        }
        declarations.append(
            types.FunctionDeclaration(
                name=name,
                description=entry["description"],
                parameters_json_schema=parameters_schema,
            )
        )

    return [types.Tool(function_declarations=declarations)]