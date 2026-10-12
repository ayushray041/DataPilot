# DataPilot

DataPilot is a beginner-friendly data exploration app built with Python,
pandas, and Streamlit. Upload a CSV or Excel dataset to inspect it, run common
data-analysis tasks, create charts locally, or ask a Gemini-powered analyst
questions about the data.

## Key features

- **Dataset overview:** See row and column counts, missing cells, duplicate
  rows, a data preview, and per-column types and value counts.
- **Profiling:** Summarize dataset shape, column names and types, missing and
  distinct values, and duplicate rows.
- **Filtering and sorting:** Filter on text or numeric values using comparison
  operators, sort by a column, and optionally limit the returned rows.
- **Statistics and correlation:** Calculate descriptive statistics for a
  numeric column and Pearson correlation for two numeric columns.
- **Charts:** Create bar, line, scatter, and histogram charts. Bar charts can
  aggregate by mean, sum, count, minimum, or maximum, or show individual
  records.
- **Anomaly detection:** Use Isolation Forest to flag unusual observations in
  selected numeric columns. An anomaly is unusual according to the model; it
  does **not** prove that the observation is an error.
- **AI Analyst:** Ask natural-language questions about a loaded dataset.
  Gemini can request registered DataPilot tools, whose results are returned
  for the model to explain. The app also saves conversation and chart results
  across Streamlit reruns.
- **Local chart creation:** Create charts in the Visualizations tab without
  calling Gemini. Generated chart files are saved under `outputs/`.

## Architecture

1. `app/streamlit_app.py` loads an uploaded file into a pandas DataFrame and
   renders the Overview, AI Analyst, and Visualizations tabs.
2. The local analysis functions live in `tools/`. They validate their inputs
   and return summaries, filtered/sorted DataFrames, statistics, anomaly
   results, or saved chart paths.
3. `agent/tool_registry.py` describes the available functions and their
   arguments. `agent/gemini_tools.py` turns those descriptions into Google
   GenAI function declarations.
4. `agent/agent.py` sends the user's question and dataset/tool context to
   Gemini. It manually handles function calls and executes only functions
   registered in `TOOL_REGISTRY`; it does not run generated Python code.
5. `agent/llm.py` loads Gemini settings from the project-root `.env` file and
   creates the SDK client when requested.

Charts created by the AI Analyst and charts created locally are displayed in
the app and saved as image files. The local chart creator calls the existing
Python chart functions directly, so it can be used independently of Gemini.

## Tools and technologies

- Python 3.13
- pandas
- Streamlit
- Matplotlib (with a non-interactive backend for chart generation)
- scikit-learn (`IsolationForest` for anomaly detection)
- Google Gen AI SDK (`google-genai`)
- python-dotenv
- pytest

Dependencies are listed in `requirements.txt`.

## Project structure

```text
.
├── agent/
│   ├── agent.py
│   ├── gemini_tools.py
│   ├── llm.py
│   └── tool_registry.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── check_gemini_connection.py
│   ├── test_anomaly.py
│   ├── test_gemini_tools.py
│   ├── test_profiler.py
│   ├── test_query.py
│   ├── test_registry.py
│   ├── test_statistics.py
│   └── test_visualization.py
├── tools/
│   ├── anomaly.py
│   ├── profiler.py
│   ├── query.py
│   ├── statistics.py
│   └── visualization.py
├── .env                 # Create locally; do not commit
└── requirements.txt
```

## Setup on macOS

### 1. Check Python

Install Python 3.13 if needed, then open Terminal, go to the project folder,
and verify that the Python 3.13 launcher is available:

```bash
cd /path/to/AIML
python3.13 --version
```

The command should report Python 3.13.x.

### 2. Create and activate a virtual environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

When the environment is active, your shell prompt usually shows `(.venv)`.
Activate it again with `source .venv/bin/activate` whenever you open a new
Terminal window for this project.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The app accepts `.csv`, `.xlsx`, and `.xls` uploads. The listed
`openpyxl` dependency supports `.xlsx`; reading legacy `.xls` files may also
require installing the `xlrd` Excel engine in the active environment.

### 4. Configure Gemini

Create a file named `.env` in the project root (the same folder as
`requirements.txt`) and add your own Gemini API key:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

Replace `your_gemini_api_key_here` with your key. Do not paste a real key into
this README, commit it, or share it publicly. `.env` is listed in `.gitignore`.
`GEMINI_MODEL` is optional; if it is missing or blank, the current default is
`gemini-3.8-flash`. The AI Analyst needs a valid API key and network access.
Local chart creation does not require a Gemini API request.

## Run the app

With the virtual environment activated, from the project root run:

```bash
python -m streamlit run app/streamlit_app.py
```

Streamlit will print a local address to open in your browser. Upload a dataset
from the sidebar to explore it. The Overview tab shows a summary and preview;
the AI Analyst tab accepts questions; the Visualizations tab shows saved
charts and includes a local chart creator.

## Run the tests

Activate the virtual environment, move to the project root, and run:

```bash
python -m pytest
```

The unit tests cover the data tools, registry, and Gemini declaration
construction. Constructing the Gemini declarations does not call the Gemini
API, so the unit tests do not require a configured API key. The separate
`tests/check_gemini_connection.py` script makes a live Gemini request when run
directly and therefore requires a valid key, network access, and available
API quota.

## Limitations

- Gemini-backed questions require a working Gemini API key, network access,
  and available API quota. Rate limits or exhausted quota can prevent an
  answer; wait or check the API project's quota and billing settings.
- AI analysis uses registered tools and provided dataset context, but its
  natural-language explanations should still be checked against the data.
- Anomaly detection identifies observations that look unusual to the fitted
  model. It cannot determine whether they are mistakes or explain why they
  occurred.
- Chart functions need valid selected columns and at least one usable
  (non-missing) data point. Numeric charts require supported real numeric
  columns.
- The app handles upload parsing errors, but Excel file support depends on
  the required pandas Excel engine being installed. In particular, legacy
  `.xls` files may need the optional `xlrd` package.