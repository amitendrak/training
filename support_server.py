import os
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="AI Support Ticket Assistant API",
    version="1.0",
    description="A modular RAG & LCEL microservice that classifies support tickets and generates responses."
)

# Initialize OpenAI Model (gpt-4o-mini)
# It automatically picks up the OPENAI_API_KEY from environment variables.
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 1: Classification & Sentiment Analysis Prompts
sentiment_prompt = ChatPromptTemplate.from_template(
    "Analyze the sentiment of the following customer support ticket.\n"
    "Respond with exactly one word: 'Angry', 'Neutral', or 'Happy'.\n\n"
    "Ticket: {ticket_text}"
)

category_prompt = ChatPromptTemplate.from_template(
    "Classify the following customer support ticket into one of these categories:\n"
    "- 'Billing' (for payments, invoices, refunds)\n"
    "- 'Technical Support' (for system errors, bugs, account access)\n"
    "- 'General' (for general inquiries, feedback)\n\n"
    "Respond with exactly one of the category names.\n\n"
    "Ticket: {ticket_text}"
)

# StrOutputParser to parse the output cleanly as a string
output_parser = StrOutputParser()

# Subchains for parallel execution
sentiment_chain = sentiment_prompt | model | output_parser
category_chain = category_prompt | model | output_parser

# Step 2: Parallel Triage Chain
triage_chain = RunnableParallel(
    sentiment=sentiment_chain,
    category=category_chain,
    ticket_text=RunnablePassthrough() # Passes the original ticket_text forward
)

# Step 3: Response Templates
billing_template = ChatPromptTemplate.from_template(
    "You are an empathetic Billing Support Specialist at Acme Corp.\n"
    "The customer has submitted the following ticket with a sentiment of '{sentiment}'.\n"
    "Ticket: {ticket_text}\n\n"
    "Write a professional, polite, and reassuring response. Let them know we take billing queries very seriously.\n"
    "Response:"
)

tech_template = ChatPromptTemplate.from_template(
    "You are an expert Technical Support Engineer at Acme Corp.\n"
    "The customer has submitted the following ticket with a sentiment of '{sentiment}'.\n"
    "Ticket: {ticket_text}\n\n"
    "Write a helpful, technical-yet-friendly response. Acknowledge their issue, explain that our engineering team is looking into it, and provide hope for a quick resolution.\n"
    "Response:"
)

general_template = ChatPromptTemplate.from_template(
    "You are a Customer Service Representative at Acme Corp.\n"
    "The customer has submitted the following ticket with a sentiment of '{sentiment}'.\n"
    "Ticket: {ticket_text}\n\n"
    "Write a friendly and helpful response answering their ticket.\n"
    "Response:"
)

# Step 4: Routing Function
def route_ticket(inputs):
    category = inputs["category"].strip().lower()
    if "billing" in category:
        return billing_template
    elif "technical" in category or "tech" in category:
        return tech_template
    else:
        return general_template

# Complete LCEL Pipeline
# Ingest raw text -> Triage -> Route to correct prompt template -> Run through model -> Parse output
support_chain = triage_chain | RunnableLambda(route_ticket) | model | output_parser

# Expose chain via LangServe add_routes
add_routes(
    app,
    support_chain,
    path="/support"
)

if __name__ == "__main__":
    print("[SERVER] Starting FastAPI server on http://localhost:8000")
    print("[SERVER] Interactive API Playground is available at http://localhost:8000/support/playground/")
    uvicorn.run(app, host="localhost", port=8000)
