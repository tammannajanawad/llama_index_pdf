# llama_index_pdf_qa

A small [Streamlit](https://streamlit.io/) app that lets you upload PDFs, build a vector index with [LlamaIndex](https://www.llamaindex.ai/), and ask questions. Answers are grounded in your documents only: the prompts instruct the model not to use outside knowledge or guess when the text is insufficient.

## How it works

- PDFs are loaded with `SimpleDirectoryReader`, embedded with OpenAI `text-embedding-3-small`, and indexed with `VectorStoreIndex`.
- Queries go through a QA + refine flow with custom templates so the assistant stays within the retrieved excerpts.

## Requirements

- Python **3.12+**
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to the chat and embedding models used in `main.py` (`gpt-4o-mini`, `text-embedding-3-small`).

## Setup

1. Clone the repo and enter the project directory.

2. Install dependencies (this project uses [uv](https://github.com/astral-sh/uv); `pip` works if you prefer):

   ```bash
   uv sync
   ```

3. Create a `.env` file in the project root with your API key:

   ```bash
   OPENAI_API_KEY=sk-...
   ```

   The app loads it via `python-dotenv`.

## Run

```bash
uv run streamlit run main.py
```

Or, with an activated virtual environment:

```bash
streamlit run main.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Usage

1. Upload one or more `.pdf` files.
2. Enter a question in the text field.
3. The app processes the files, builds the index, and shows an answer for that question.

> **Note:** The index is rebuilt on each run when you submit a question. There is no persistent store between sessions in the current `main.py`.
