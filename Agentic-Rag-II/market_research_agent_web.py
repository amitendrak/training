"""Market research agent: live web search + OpenAI reasoning over the results.

Import `research()` from a UI layer (see app.py), or run this file directly for
the Gradio interface.
"""

import os
from functools import lru_cache

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
PAGE_CHAR_LIMIT = 6000  # keep each source small enough to fit the context window

SYSTEM_PROMPT = (
    "You are a market research analyst. Answer only from the numbered sources "
    "provided. Cite them inline as [1], [2]. If the sources do not cover "
    "something, say so instead of guessing."
)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Build the OpenAI client lazily so importing this module never fails hard."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key)


# Fetch the readable text of a single page; fall back to the search snippet
def fetch_page_text(url, fallback):
    try:
        page = requests.get(url, headers=HEADERS, timeout=10)
        page.raise_for_status()
        soup = BeautifulSoup(page.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:PAGE_CHAR_LIMIT] if text else fallback
    except requests.RequestException:
        # A dead or blocked link should not kill the whole research run
        return fallback


# Web search returning grounded documents (title, url, text)
def web_search(query, max_results=5):
    results = DDGS().text(query, max_results=max_results)
    web_docs = []
    for r in results:
        web_docs.append(
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "text": fetch_page_text(r.get("href", ""), r.get("body", "")),
            }
        )
    return web_docs


# Query OpenAI for reasoning over the retrieved documents
def agentic_rag_with_openai(query, web_docs):
    if not web_docs:
        return "No search results were found for that query. Try rephrasing it."

    context = "\n\n".join(
        f"[{i}] {doc['title']} ({doc['url']})\n{doc['text']}"
        for i, doc in enumerate(web_docs, 1)
    )
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nSources:\n{context}"},
        ],
    )
    return response.choices[0].message.content


def research(query, max_results=5):
    """Run the full search-then-reason pipeline.

    Returns (answer, web_docs) so a UI can render the sources separately.
    """
    if not query.strip():
        return "Please enter a query.", []
    web_docs = web_search(query, max_results=max_results)
    return agentic_rag_with_openai(query, web_docs), web_docs


# Gradio Interface function: flattens the answer and sources into one string
def market_research_agent(query):
    answer, web_docs = research(query)
    if not web_docs:
        return answer
    sources = "\n".join(f"[{i}] {doc['url']}" for i, doc in enumerate(web_docs, 1))
    return f"{answer}\n\n---\nSources:\n{sources}"


def launch_gradio():
    import gradio as gr

    iface = gr.Interface(
        fn=market_research_agent,
        inputs=gr.Textbox(label="Enter Your Market Research Query"),
        outputs=gr.Textbox(label="Agent Response"),
        title="Market Research Agent",
        description="This agent uses real-time web search and OpenAI's LLM for adaptive market research.",
    )
    iface.launch()


if __name__ == "__main__":
    launch_gradio()
