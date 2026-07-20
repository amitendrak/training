import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent
import logging
from pathlib import Path
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core import load_index_from_storage


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM and embeddings
llm = Gemini(model_name="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)
embed_model = GoogleGenAIEmbedding(
    model_name="models/gemini-embedding-001", 
    api_key=GOOGLE_API_KEY
)
# Define index storage path
INDEX_STORAGE_DIR = "index_storage"

# Custom analytics tool for statistical calculations
def compute_analytics(metric: str, column: str, filter_condition: str = None) -> float:
    """Compute statistical metrics (e.g., sum, average) on sales data with optional filtering."""
    df = pd.read_csv("sales_data.csv")
    if filter_condition:
        df = df.query(filter_condition)
    if metric == "sum":
        return df[column].sum()
    elif metric == "average":
        return df[column].mean()
    return 0.0

analytics_tool = FunctionTool.from_defaults(
    fn=compute_analytics,
    name="analytics_tool",
    description="Computes statistical metrics (sum, average) on sales data with optional filters."
)

# Check if index exists on disk
index_storage_path = Path(INDEX_STORAGE_DIR)
if index_storage_path.exists():
    logger.info("Loading existing index from disk...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
    #index = VectorStoreIndex.load_from_disk(storage_context,embed_model=embed_model)
    index = load_index_from_storage(storage_context,embed_model=embed_model)
    
else:
    logger.info("Indexing sales data...")
    # Read sales data from CSV
    sales_df = pd.read_csv("sales_data.csv")

    # Convert DataFrame to list of Documents
    documents = []
    for _, row in sales_df.iterrows():
        text = f"OrderID: {row['OrderID']}, Date: {row['Date']}, Region: {row['Region']}, " \
               f"Product: {row['Product']}, Category: {row['Category']}, Quantity: {row['Quantity']}, " \
               f"UnitPrice: {row['UnitPrice']}, TotalSale: {row['TotalSale']}"
        documents.append(Document(text=text))

    # Create index with sentence splitter
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[splitter]
    )
    # Persist index to disk
    index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
    logger.info("Indexing complete and saved to disk.")

# Create query engine
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

# Define query engine tool
sales_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="sales_data_tool",
    description="Provides insights from sales data including total sales, regional performance, and product analysis."
)

# Create ReAct agent with multiple tools
#agent = ReActAgent.from_tools([sales_tool, analytics_tool], llm=llm, verbose=True)
agent = ReActAgent(
    name="sales_analysis_agent",
    description="Analyses sales data and performs sales calculations.",
    tools=[sales_tool, analytics_tool],
    llm=llm,
    verbose=True
)
# Function to query sales data using the agent
def analyze_sales(query, query_history):
    response = agent.chat(query)
    query_history.append((query, str(response.response)))
    return response.response

# Interactive query loop with query history
if __name__ == "__main__":
    query_history = []
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
            response = analyze_sales(user_query, query_history)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error processing query: {e}")
