"""Streamlit dashboard for exploring datasets with DataPilot."""

import hashlib
from pathlib import Path

import pandas as pd

import sys
from pathlib import Path

import streamlit as st
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the project's own modules after setting the path.
from agent.agent import DataAgent

from tools.visualization import (
    create_bar_chart,
    create_histogram,
    create_line_chart,
    create_scatter_plot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHART_TOOLS = {"bar_chart", "line_chart", "scatter_plot", "histogram"}


def _initialize_session_state() -> None:
    """Create the session keys used to keep the dataset and chat between reruns."""
    defaults = {
        "upload_hash": None,
        "upload_error": None,
        "dataframe": None,
        "data_agent": None,
        "agent_error": None,
        "messages": [],
        "last_question": None,
        "last_answer": None,
        "last_tool_results": [],
        "pending_question": None,
        "manual_charts": [],
        "manual_chart_counter": 0,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _load_uploaded_dataset(uploaded_file) -> None:
    """Load a newly uploaded CSV or Excel file, identified by its content."""
    file_contents = uploaded_file.getvalue()
    content_hash = hashlib.sha256(file_contents).hexdigest()
    if content_hash == st.session_state.upload_hash:
        return

    # Reset file-specific state only when the file bytes actually change.
    st.session_state.upload_hash = content_hash
    st.session_state.upload_error = None
    st.session_state.dataframe = None
    st.session_state.data_agent = None
    st.session_state.agent_error = None
    st.session_state.messages = []
    st.session_state.last_question = None
    st.session_state.last_answer = None
    st.session_state.last_tool_results = []
    st.session_state.pending_question = None
    st.session_state.manual_charts = []
    st.session_state.manual_chart_counter = 0

    file_suffix = Path(uploaded_file.name).suffix.lower()
    try:
        uploaded_file.seek(0)
        if file_suffix == ".csv":
            dataframe = pd.read_csv(uploaded_file)
        elif file_suffix in {".xlsx", ".xls"}:
            dataframe = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Choose a CSV or Excel file (.csv, .xlsx, or .xls).")
    except Exception:
        st.session_state.upload_error = (
            "The file could not be read. Check that it is a valid CSV or Excel "
            "file and try uploading it again."
        )
        return

    st.session_state.dataframe = dataframe
    try:
        st.session_state.data_agent = DataAgent(dataframe)
    except (RuntimeError, TypeError, ValueError) as error:
        # Data can still be previewed if Gemini configuration is unavailable.
        st.session_state.agent_error = str(error)


def _resolve_chart_path(path_value) -> Path | None:
    """Resolve a chart path relative to the project root when necessary."""
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _render_tool_results(tool_results: list[dict]) -> None:
    """Render displayable DataFrames and chart images from agent results."""
    for tool_result in tool_results:
        tool_name = tool_result.get("tool", "tool")
        result = tool_result.get("result")

        if isinstance(result, pd.DataFrame):
            st.dataframe(
                result,
                width="stretch",
                hide_index=True,
                alt=f"Data returned by the {tool_name} tool",
            )
        elif isinstance(result, list) and all(
            isinstance(row, dict) for row in result
        ):
            st.dataframe(
                pd.DataFrame(result),
                width="stretch",
                hide_index=True,
                alt=f"Rows returned by the {tool_name} tool",
            )
        elif tool_name in CHART_TOOLS:
            chart_path = _resolve_chart_path(result)
            if chart_path is None or not chart_path.is_file():
                missing_path = str(chart_path) if chart_path is not None else "unknown path"
                st.warning(
                    f"The {tool_name.replace('_', ' ')} chart could not be found "
                    f"at {missing_path}. Try creating the chart again."
                )
            else:
                st.image(
                    str(chart_path),
                    caption=tool_name.replace("_", " ").title(),
                    width="stretch",
                    alt=f"{tool_name.replace('_', ' ').title()} created from the dataset",
                )
        elif isinstance(result, dict) and "error" in result:
            st.warning(str(result["error"]))


def _render_visualizations(messages: list[dict]) -> None:
    """Show all chart images saved in the conversation's tool results."""
    found_chart = False
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_result in message.get("tool_results", []):
            if tool_result.get("tool") not in CHART_TOOLS:
                continue
            found_chart = True
            _render_tool_results([tool_result])
    for chart in st.session_state.manual_charts:
        found_chart = True
        _render_tool_results([chart])
    if not found_chart:
        st.info(
            "No charts yet. Ask the AI Analyst to create a bar, line, scatter, "
            "or histogram chart to see it here, or create one below without AI."
        )


def _numeric_columns(dataframe: pd.DataFrame) -> list:
    """Return real numeric columns supported by the visualization helpers."""
    return [
        column
        for column in dataframe.columns
        if is_numeric_dtype(dataframe[column].dtype)
        and not is_bool_dtype(dataframe[column].dtype)
        and not is_complex_dtype(dataframe[column].dtype)
    ]


def _render_manual_chart_creator(dataframe: pd.DataFrame) -> None:
    """Create and save a chart locally without sending a request to Gemini."""
    st.subheader("Create chart without AI")
    st.caption(
        "Choose chart settings below. This uses your uploaded data and local "
        "Python tools, so it does not use Gemini."
    )

    chart_type = st.selectbox(
        "Chart type",
        ["Bar chart", "Line chart", "Scatter plot", "Histogram"],
        key="manual_chart_type",
    )
    numeric_columns = _numeric_columns(dataframe)
    all_columns = list(dataframe.columns)

    category_column = None
    value_column = None
    x_column = None
    y_column = None
    histogram_column = None
    aggregation = "mean"

    if chart_type == "Bar chart":
        if not all_columns:
            st.info("This dataset has no columns to chart.")
            return
        if not numeric_columns:
            st.info("A bar chart needs at least one numeric value column.")
            return
        category_column = st.selectbox(
            "Category column",
            all_columns,
            key="manual_bar_category",
        )
        value_column = st.selectbox(
            "Numeric value column",
            numeric_columns,
            key="manual_bar_value",
        )
        aggregation = st.selectbox(
            "Aggregation",
            ["mean", "sum", "count", "min", "max", "none"],
            key="manual_bar_aggregation",
        )
    elif chart_type in {"Line chart", "Scatter plot"}:
        if len(numeric_columns) < 2:
            st.info(
                f"{chart_type} needs at least two numeric columns. "
                "Check the data types in your dataset."
            )
            return
        x_column = st.selectbox(
            "Numeric x-axis column",
            numeric_columns,
            key=f"manual_{chart_type}_x",
        )
        y_options = [column for column in numeric_columns if column != x_column]
        y_column = st.selectbox(
            "Numeric y-axis column",
            y_options,
            key=f"manual_{chart_type}_y",
        )
    else:
        if not numeric_columns:
            st.info("A histogram needs at least one numeric column.")
            return
        histogram_column = st.selectbox(
            "Numeric column",
            numeric_columns,
            key="manual_histogram_column",
        )

    if st.button("Generate chart", key="generate_manual_chart"):
        output_directory = PROJECT_ROOT / "outputs"
        chart_kind = chart_type.lower().replace(" ", "_")
        chart_number = st.session_state.manual_chart_counter
        output_path = output_directory / f"manual_{chart_kind}_{chart_number + 1:03d}.png"
        while output_path.exists():
            chart_number += 1
            output_path = (
                output_directory / f"manual_{chart_kind}_{chart_number + 1:03d}.png"
            )
        st.session_state.manual_chart_counter = chart_number + 1

        try:
            if chart_type == "Bar chart":
                chart_path = create_bar_chart(
                    dataframe,
                    category_column,
                    value_column,
                    output_path=output_path,
                    aggregation=aggregation,
                )
                tool_name = "bar_chart"
            elif chart_type == "Line chart":
                chart_path = create_line_chart(
                    dataframe,
                    x_column,
                    y_column,
                    output_path=output_path,
                )
                tool_name = "line_chart"
            elif chart_type == "Scatter plot":
                chart_path = create_scatter_plot(
                    dataframe,
                    x_column,
                    y_column,
                    output_path=output_path,
                )
                tool_name = "scatter_plot"
            else:
                chart_path = create_histogram(
                    dataframe,
                    histogram_column,
                    output_path=output_path,
                )
                tool_name = "histogram"
        except (TypeError, ValueError) as error:
            st.error(f"Could not create the chart: {error}")
        else:
            st.session_state.manual_charts.append(
                {"tool": tool_name, "result": chart_path}
            )
            st.success(f"Chart saved as {Path(chart_path).name}.")

    # Re-display saved manual charts on every rerun without regenerating them.
    for chart in st.session_state.manual_charts:
        _render_tool_results([chart])


def _render_overview(dataframe: pd.DataFrame) -> None:
    """Show key dataset metrics, a preview, and column-level details."""
    missing_count = int(dataframe.isna().sum().sum())
    duplicate_count = int(dataframe.duplicated().sum())

    metrics = st.columns(4)
    metric_values = (
        ("Rows", f"{len(dataframe):,}"),
        ("Columns", f"{len(dataframe.columns):,}"),
        ("Missing cells", f"{missing_count:,}"),
        ("Duplicate rows", f"{duplicate_count:,}"),
    )
    for column, (label, value) in zip(metrics, metric_values):
        column.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Data preview")
    st.dataframe(
        dataframe.head(100),
        width="stretch",
        hide_index=True,
        alt="Preview of the first 100 dataset rows",
    )

    with st.expander("Column information", expanded=False):
        column_information = pd.DataFrame(
            {
                "Column": [str(column) for column in dataframe.columns],
                "Data type": [str(dtype) for dtype in dataframe.dtypes],
                "Missing values": [
                    int(count) for count in dataframe.isna().sum().tolist()
                ],
                "Unique values": [
                    int(dataframe.iloc[:, position].nunique(dropna=True))
                    for position in range(len(dataframe.columns))
                ],
            }
        )
        st.dataframe(
            column_information,
            width="stretch",
            hide_index=True,
            alt="Data types and value counts for each dataset column",
        )


def _queue_chat_question() -> None:
    """Save a newly submitted chat question for one-time processing."""
    submitted_value = st.session_state.get("dataset_question")
    if isinstance(submitted_value, str) and submitted_value.strip():
        st.session_state.pending_question = submitted_value.strip()


def _render_chat(messages: list[dict]) -> None:
    """Display saved messages and process only explicitly submitted questions."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_tool_results(message.get("tool_results", []))

    st.chat_input(
        "Ask a question about your dataset",
        key="dataset_question",
        disabled=st.session_state.data_agent is None,
        on_submit=_queue_chat_question,
    )
    # The callback only queues a prompt on submit; tab-change reruns have none.
    question = st.session_state.pending_question
    if question is None:
        return

    question = question.strip()
    # Clear before calling external code so reruns cannot replay this request.
    st.session_state.pending_question = None
    if not question:
        return

    # Chat input returns a value only on its submit run; tab reruns return None.
    st.session_state.last_question = question
    st.session_state.last_answer = None
    st.session_state.last_tool_results = []
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    agent = st.session_state.data_agent
    if agent is None:
        answer = (
            "The dataset is loaded, but the AI Analyst is not ready. "
            "Check the Gemini configuration and try uploading the file again."
        )
        tool_results = []
    else:
        try:
            answer = agent.ask(question)
            tool_results = agent.last_tool_results
        except Exception as error:
            # DataAgent handles quota errors itself; this prevents unexpected API
            # or runtime failures from taking down the Streamlit page.
            st.session_state.agent_error = (
                f"{type(error).__name__}: request failed; check the server logs "
                "and Gemini configuration."
            )
            answer = (
                "I couldn't complete that request. Please check your connection "
                "and Gemini configuration, then try again."
            )
            tool_results = agent.last_tool_results

    saved_message = {
        "role": "assistant",
        "content": answer,
        "tool_results": list(tool_results),
    }
    st.session_state.last_answer = answer
    st.session_state.last_tool_results = list(tool_results)
    st.session_state.messages.append(saved_message)
    with st.chat_message("assistant"):
        st.markdown(answer)
        _render_tool_results(tool_results)


def main() -> None:
    """Run the DataPilot Streamlit dashboard."""
    st.set_page_config(
        page_title="DataPilot",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _initialize_session_state()
    st.markdown(
        """
        <style>
        .stApp {
            background: var(--background-color);
            color: var(--text-color);
        }
        .metric-card {
            background: var(--secondary-background-color);
            border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
            border-radius: 14px;
            padding: 1rem 1.15rem;
            min-height: 94px;
        }
        .metric-label {
            color: var(--text-color);
            opacity: 0.72;
            font-size: 0.9rem;
        }
        .metric-value {
            color: var(--primary-color);
            font-size: 1.65rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }
        @media (max-width: 700px) {
            .metric-card {
                padding: 0.8rem;
                min-height: 82px;
            }
            .metric-value {
                font-size: 1.35rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("DataPilot")
    st.caption("A clearer view of your data, with an AI analyst by your side.")

    with st.sidebar:
        st.header("Your dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Supported formats: .csv, .xlsx, and .xls",
            key="dataset_upload",
        )
        if uploaded_file is not None:
            _load_uploaded_dataset(uploaded_file)

        dataframe = st.session_state.dataframe
        if dataframe is not None:
            st.success(f"Loaded {len(dataframe):,} rows and {len(dataframe.columns):,} columns.")
        else:
            st.caption("Upload a file to explore your data.")

    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)

    dataframe = st.session_state.dataframe
    if dataframe is None:
        st.info(
            "Start by uploading a CSV or Excel dataset in the sidebar. "
            "Your overview, AI analysis, and visualizations will appear here."
        )
    elif dataframe.empty or len(dataframe.columns) == 0:
        st.warning(
            "This file loaded, but it has no data rows or columns to analyze."
        )

    if dataframe is not None and st.session_state.agent_error:
        st.warning(
            "The dataset is ready to preview, but the AI Analyst could not be "
            f"initialized: {st.session_state.agent_error}"
        )

    # Older sessions are upgraded without discarding saved messages.
    if st.session_state.messages and st.session_state.last_question is None:
        for message in reversed(st.session_state.messages):
            if message.get("role") == "user":
                st.session_state.last_question = message.get("content")
                break
    if st.session_state.messages and st.session_state.last_answer is None:
        for message in reversed(st.session_state.messages):
            if message.get("role") == "assistant":
                st.session_state.last_answer = message.get("content")
                st.session_state.last_tool_results = list(
                    message.get("tool_results", [])
                )
                break

    overview_tab, analyst_tab, visualizations_tab = st.tabs(
        ["Overview", "AI Analyst", "Visualizations"]
    )
    with overview_tab:
        if dataframe is None:
            st.info("Upload a dataset to see its overview and preview.")
        else:
            _render_overview(dataframe)

    with analyst_tab:
        st.subheader("AI Analyst")
        if dataframe is None:
            st.info("Upload a dataset to start a conversation with the AI Analyst.")
        elif st.session_state.data_agent is None:
            st.info(
                "The AI Analyst is unavailable until a non-empty dataset is "
                "loaded and Gemini is configured."
            )
        elif not st.session_state.messages:
            st.info(
                "Ask a question about your columns, request a summary, or ask "
                "for a chart to get started."
            )
        if dataframe is not None:
            _render_chat(st.session_state.messages)

    with visualizations_tab:
        st.subheader("Visualizations")
        if dataframe is None:
            st.info("Upload a dataset to create and view charts.")
        else:
            _render_visualizations(st.session_state.messages)
            _render_manual_chart_creator(dataframe)


main()