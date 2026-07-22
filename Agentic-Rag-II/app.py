"""Streamlit chat UI for the Market Research Agent.

Run with:  streamlit run app.py
"""

import streamlit as st

from market_research_agent_web import MODEL, get_client, research

st.set_page_config(page_title="Market Research Agent", page_icon="", layout="wide")

EXAMPLE_QUERIES = [
    "What is the electric vehicle market size in 2025?",
    "Who are the leading vendors in the agentic AI space?",
    "What are the main growth drivers for the cloud security market?",
]


def render_sidebar():
    with st.sidebar:
        st.header("About")
        st.write(
            "This agent searches the live web, pulls the text of each result, and "
            "answers only from what it retrieved, citing sources inline."
        )
        st.caption(f"Model: `{MODEL}`")

        max_results = st.slider("Sources per query", 1, 10, 5)

        st.subheader("Example queries")
        for query in EXAMPLE_QUERIES:
            if st.button(query, width="stretch"):
                st.session_state.pending_query = query

        if st.button("Clear conversation", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    return max_results


def render_sources(web_docs):
    if not web_docs:
        return
    with st.expander(f"Sources ({len(web_docs)})"):
        for i, doc in enumerate(web_docs, 1):
            st.markdown(f"**[{i}] [{doc['title'] or doc['url']}]({doc['url']})**")
            st.caption(f"{len(doc['text']):,} characters retrieved")


st.title("Market Research Agent")
st.caption("Agentic RAG over live web search")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Fail early and visibly if the API key is missing
try:
    get_client()
except Exception as e:
    st.error(f"Could not start the agent: {e}")
    st.stop()

max_results = render_sidebar()

# Replay the conversation so far, including each turn's sources
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources"))

# A sidebar example button and the chat box are two ways into the same path
query = st.chat_input("Ask a market research question...")
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the web and reading sources..."):
            try:
                answer, web_docs = research(query, max_results=max_results)
            except Exception as e:
                answer, web_docs = f"Error processing query: {e}", []
        st.markdown(answer)
        render_sources(web_docs)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": web_docs}
    )
