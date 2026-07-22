"""Streamlit chat UI for InsightPulse.

Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st
from llama_index.core.llms import ChatMessage

from sales_analysis_agent import (
    LLM_MODEL,
    SALES_CSV,
    analyze_sales_sync,
    build_agent,
)

st.set_page_config(page_title="InsightPulse", page_icon="", layout="wide")

EXAMPLE_QUERIES = [
    "What is the total sales for Laptops in South in 2024?",
    "Which region has the highest average unit price?",
    "How many Monitors were sold in total?",
]


@st.cache_resource(show_spinner="Loading sales index and building agent...")
def get_agent():
    """Build the agent once per server process, not once per rerun."""
    return build_agent()


@st.cache_data(show_spinner=False)
def get_sales_data():
    return pd.read_csv(SALES_CSV)


def render_sidebar():
    with st.sidebar:
        st.header("About")
        st.write(
            "InsightPulse is a ReAct agent over your sales data. It picks between "
            "semantic search on the records and exact pandas aggregations."
        )
        st.caption(f"Model: `{LLM_MODEL}`")

        st.subheader("Example queries")
        for query in EXAMPLE_QUERIES:
            if st.button(query, width="stretch"):
                st.session_state.pending_query = query

        st.subheader("Sales data")
        try:
            df = get_sales_data()
            st.caption(f"{len(df):,} rows")
            st.dataframe(df.head(20), width="stretch", height=240)
        except Exception as e:
            st.warning(f"Could not read sales data: {e}")

        if st.button("Clear conversation", width="stretch"):
            st.session_state.messages = []
            st.rerun()


def to_chat_history(messages) -> list[ChatMessage]:
    """Convert the Streamlit message list into llama-index chat history."""
    return [ChatMessage(role=m["role"], content=m["content"]) for m in messages]


st.title("InsightPulse")
st.caption("AI-powered sales report analysis")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Fail early and visibly if the key or index is not usable
try:
    get_agent()
except Exception as e:
    st.error(f"Could not start the agent: {e}")
    st.stop()

render_sidebar()

# Replay the conversation so far
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# A sidebar example button and the chat box are two ways into the same path
query = st.chat_input("Ask about your sales data...")
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

if query:
    # History must be captured before the new turn is appended
    history = to_chat_history(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = analyze_sales_sync(query, history)
            except Exception as e:
                answer = f"Error processing query: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
