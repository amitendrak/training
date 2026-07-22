"""InsightPulse core logic: a CSV + HR-policy-aware ReAct agent.

The agent has two tools:
  - pandas_query_tool: writes pandas code on the fly against the sales CSV,
    rather than relying on a fixed set of pre-defined aggregations.
  - hr_policy_tool: semantic search over an indexed HR policy PDF, for
    questions about leave, WFH, conduct, benefits, etc.

Import `build_agent()` from a UI layer (see app.py), or run this file
directly for the command-line interface.
"""

import asyncio
import logging
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Resolve paths relative to this file so the app runs from any directory
BASE_DIR = Path(__file__).parent
SALES_CSV = BASE_DIR / "sales_data.csv"
HR_POLICY_PDF = BASE_DIR / "hr_policy.pdf"
HR_INDEX_STORAGE_DIR = BASE_DIR / "hr_index_storage"

SYSTEM_PROMPT = (
    "You are InsightPulse, a sales analysis and HR policy assistant. "
    "Use pandas_query_tool for every question about the sales data — write a "
    "short pandas snippet against the `df` DataFrame and assign your final "
    "answer to a variable named `result`. Use hr_policy_tool for any question "
    "about HR policies such as leave, work-from-home, conduct, expenses, "
    "performance reviews, or benefits. Always state the numbers or policy "
    "clauses you used in your final response."
)

# Columns present in sales_data.csv, shown to the LLM so it writes valid code
# without needing to inspect the file first.
SALES_COLUMNS = [
    "OrderID", "Date", "Region", "Product", "Category",
    "Quantity", "UnitPrice", "TotalSale",
]

# Minimal builtins made available inside the sandboxed exec() call. Anything
# not listed here (import, open, eval, exec, __import__, etc.) is unavailable.
SAFE_BUILTINS = {
    "len": len, "range": range, "sum": sum, "min": min, "max": max,
    "abs": abs, "round": round, "sorted": sorted, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "str": str, "int": int,
    "float": float, "bool": bool, "enumerate": enumerate, "zip": zip,
}


def get_api_key() -> str:
    """Return the OpenAI API key, or raise a message the UI can display."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return api_key


def run_pandas_query(code: str) -> str:
    """Execute a short pandas snippet against the sales data and return the result.

    The sales data is preloaded as a pandas DataFrame named `df`, with columns:
    OrderID, Date, Region, Product, Category, Quantity, UnitPrice, TotalSale.
    Write one or more lines of pandas/python code that operate on `df` and
    assign the final answer to a variable named `result`. Examples:
        result = df[df["Region"] == "South"]["TotalSale"].sum()
        result = df.groupby("Category")["TotalSale"].mean().sort_values(ascending=False)

    Only `pd` and `df` are available — no imports, file access, or network
    calls are permitted inside the snippet.

    Args:
        code: Python code that operates on `df` and assigns the answer to `result`.
    """
    df = pd.read_csv(SALES_CSV)

    sandbox_globals = {"__builtins__": SAFE_BUILTINS, "pd": pd, "df": df}
    sandbox_locals: dict = {}

    try:
        exec(code, sandbox_globals, sandbox_locals)  # noqa: S102 - sandboxed, see SAFE_BUILTINS
    except Exception as e:
        raise ValueError(f"Error executing pandas code: {e}") from e

    if "result" not in sandbox_locals:
        raise ValueError(
            "Code did not assign a value to `result`. Assign the final answer "
            "to a variable named `result`."
        )

    result = sandbox_locals["result"]

    # DataFrames/Series can get long; cap what goes back to the LLM.
    if isinstance(result, (pd.DataFrame, pd.Series)):
        text = result.to_string(max_rows=50)
    else:
        text = str(result)

    if len(text) > 4000:
        text = text[:4000] + "\n... (truncated)"
    return text


def load_or_build_hr_index(embed_model) -> VectorStoreIndex:
    """Load the persisted HR-policy index, or build it from the PDF on first run."""
    if HR_INDEX_STORAGE_DIR.exists():
        logger.info("Loading existing HR policy index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=str(HR_INDEX_STORAGE_DIR))
        return load_index_from_storage(storage_context, embed_model=embed_model)

    if not HR_POLICY_PDF.exists():
        raise FileNotFoundError(
            f"HR policy PDF not found at {HR_POLICY_PDF}. Add a file there "
            "(e.g. hr_policy.pdf) before starting the app."
        )

    logger.info("Loading and splitting HR policy PDF...")
    # SimpleDirectoryReader auto-selects a PDF reader (pypdf under the hood)
    # and returns one Document per page, with page_label metadata attached.
    documents = SimpleDirectoryReader(input_files=[str(HR_POLICY_PDF)]).load_data()

    # Chunk each page into overlapping windows so retrieval isn't limited to
    # whole-page granularity — useful once the handbook grows past a few pages.
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[splitter],
    )
    # Persist index to disk so later runs skip re-embedding the PDF.
    index.storage_context.persist(persist_dir=str(HR_INDEX_STORAGE_DIR))
    logger.info("HR policy indexing complete and saved to disk.")
    return index


@lru_cache(maxsize=1)
def build_agent() -> ReActAgent:
    """Build the ReAct agent with the pandas and HR-policy tools."""
    api_key = get_api_key()
    llm = OpenAI(model=LLM_MODEL, api_key=api_key)
    embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=api_key)

    pandas_tool = FunctionTool.from_defaults(
        fn=run_pandas_query,
        name="pandas_query_tool",
        description=(
            "Executes pandas code against the sales DataFrame `df` "
            f"(columns: {', '.join(SALES_COLUMNS)}). Write code that assigns "
            "the final answer to a variable named `result`, e.g. "
            '`result = df[df["Region"] == "South"]["TotalSale"].sum()`. '
            "Use this for every question about the sales data — lookups, "
            "filters, aggregations, group-bys, and sorts."
        ),
    )

    hr_index = load_or_build_hr_index(embed_model)
    hr_query_engine = hr_index.as_query_engine(llm=llm, similarity_top_k=4)
    hr_policy_tool = QueryEngineTool.from_defaults(
        query_engine=hr_query_engine,
        name="hr_policy_tool",
        description=(
            "Semantic search over the company HR policy handbook (leave, "
            "work-from-home, code of conduct, expense reimbursement, "
            "performance reviews, termination notice, health benefits, etc.). "
            "Use this for any question about HR policy or employee rules."
        ),
    )

    return ReActAgent(
        tools=[pandas_tool, hr_policy_tool],
        llm=llm,
        verbose=True,
        system_prompt=SYSTEM_PROMPT,
    )


async def analyze_sales(query: str, chat_history: list[ChatMessage] | None = None) -> str:
    """Run one agent turn. `chat_history` seeds the agent's memory for follow-ups."""
    agent = build_agent()
    response = await agent.run(user_msg=query, chat_history=chat_history or [])
    return str(response)


def analyze_sales_sync(query: str, chat_history: list[ChatMessage] | None = None) -> str:
    """Blocking wrapper for UI frameworks that are not async-aware."""
    return asyncio.run(analyze_sales(query, chat_history))


# Interactive command-line loop with query history
async def main():
    chat_history: list[ChatMessage] = []
    query_history: list[tuple[str, str]] = []
    print("Welcome to InsightPulse: Your AI-Powered Sales Report Analysis Tool!")
    print("Enter your query (e.g., 'What is the total sales for Laptops in South in 2024?')")
    print("Type 'history' to view recent queries, 'exit' to quit.")

    while True:
        user_query = input("\nYour query: ").strip()
        if user_query.lower() == "exit":
            print("Exiting InsightPulse. Goodbye!")
            break
        if user_query.lower() == "history":
            if query_history:
                print("\nRecent Queries:")
                for i, (q, r) in enumerate(query_history[-5:], 1):  # Show last 5 queries
                    print(f"{i}. Query: {q}\n   Response: {r[:100]}...")  # Truncate long responses
            else:
                print("No query history yet.")
            continue
        if not user_query:
            print("Please enter a valid query.")
            continue

        print(f"\nProcessing query: {user_query}")
        try:
            answer = await analyze_sales(user_query, chat_history)
            chat_history.append(ChatMessage(role="user", content=user_query))
            chat_history.append(ChatMessage(role="assistant", content=answer))
            query_history.append((user_query, answer))
            print(f"Response: {answer}")
        except Exception as e:
            print(f"Error processing query: {e}")


if __name__ == "__main__":
    asyncio.run(main())