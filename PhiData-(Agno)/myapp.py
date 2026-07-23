"""
Multi-Agent Demo — LangGraph + OpenAI + Streamlit
--------------------------------------------------
A simple 3-agent pipeline on a shared state graph:

  Researcher  ->  Analyst  ->  Writer

Each agent"""  """ is """  """just an OpenAI chat call with its own system prompt.
LangGraph wires them together and passes a shared "state" dict between nodes.
"""

import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv

# ---- 1. Load Environment Variables from .env ----
load_dotenv()
# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Agent Demo (LangGraph)", layout="centered")
st.title("Multi-Agent Demo - LangGraph + OpenAI")
st.caption("Researcher -> Analyst -> Writer, orchestrated as a LangGraph pipeline")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get("api_key", ""))
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    st.session_state["api_key"] = api_key

topic = st.text_input("Enter a topic", placeholder="e.g. Impact of AI on the job market in India")
run_btn = st.button("Run Multi-Agent Pipeline", type="primary")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    topic: str
    research: str
    analysis: str
    final_report: str

# ---------------------------------------------------------------------------
# Helper: one shared function to call an OpenAI agent with a role prompt
# ---------------------------------------------------------------------------
def call_agent(client: OpenAI, model: str, system_prompt: str, user_content: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content

# ---------------------------------------------------------------------------
# Agent node definitions
# ---------------------------------------------------------------------------
def build_graph(client: OpenAI, model: str):

    def researcher_node(state: AgentState) -> AgentState:
        content = call_agent(
            client, model,
            system_prompt="You are a Research Agent. Given a topic, list 4-6 concise, "
                          "well-organized key facts, trends, or data points relevant to it. "
                          "No fluff, just substantive bullet points.",
            user_content=state["topic"],
        )
        return {"research": content}

    def analyst_node(state: AgentState) -> AgentState:
        content = call_agent(
            client, model,
            system_prompt="You are an Analyst Agent. Given research notes, identify the "
                          "3 most important insights and their implications. Be sharp and specific.",
            user_content=f"Topic: {state['topic']}\n\nResearch notes:\n{state['research']}",
        )
        return {"analysis": content}

    def writer_node(state: AgentState) -> AgentState:
        content = call_agent(
            client, model,
            system_prompt="You are a Writer Agent. Using the research and analysis provided, "
                          "write a polished, engaging summary (3-4 short paragraphs) for a "
                          "general business audience.",
            user_content=(
                f"Topic: {state['topic']}\n\n"
                f"Research:\n{state['research']}\n\n"
                f"Analysis:\n{state['analysis']}"
            ),
        )
        return {"final_report": content}

    graph = StateGraph(AgentState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)

    return graph.compile()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run_btn:
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not topic:
        st.warning("Please enter a topic.")
    else:
        client = OpenAI(api_key=api_key)
        app = build_graph(client, model)

        with st.status("Running agent pipeline...", expanded=True) as status:
            st.write("Researcher agent gathering facts...")
            result = {}
            for step_output in app.stream({"topic": topic, "research": "", "analysis": "", "final_report": ""}):
                node_name, node_state = next(iter(step_output.items()))
                result.update(node_state)

                if node_name == "researcher":
                    st.write("Research complete")
                    with st.expander("Research Notes", expanded=False):
                        st.markdown(node_state["research"])
                    st.write("Analyst agent evaluating insights...")

                elif node_name == "analyst":
                    st.write("Analysis complete")
                    with st.expander("Analysis", expanded=False):
                        st.markdown(node_state["analysis"])
                    st.write("Writer agent drafting summary...")

                elif node_name == "writer":
                    st.write("Report complete")

            status.update(label="Pipeline finished", state="complete", expanded=False)

        st.subheader("Final Report")
        st.markdown(result.get("final_report", "No output generated."))
