# 🧠 Market Research Agent (Agentic RAG + Cohere + Web)

This project is a **console-based intelligent agent** that performs **real-time market research** using:

* 🔍 **Live web search** (via DuckDuckGo)
* 🧠 **Cohere's Command-R+ LLM** for reasoning and summarization
* 📚 **Agentic RAG architecture**: dynamic reasoning, adaptive retrieval, and document grounding
* ✅ 100% open source and free-tier tools

---

## 📁 Folder Structure

```
DEMO 2/
├── market_research_agent_web.py       # Main agent logic
├── .env                               # Contains API keys (not committed)
├── requirements.txt                   # Required dependencies
├── env4/                              # Python virtual environment
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repo or Navigate to Folder

Ensure you're in:

```
D:\Agentic AI Course\Module 5\DEMO\demo 2
```

### 2. Create `.env` file

Create a file named `.env` and add your Cohere API key:

```
COHERE_API_KEY=your_cohere_api_key_here
```

You can get a free API key from: [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)

### 3. Activate Virtual Environment

If not already created:

```bash
python -m venv env4
```

Activate:
e.g path
```bash
D:\Agentic AI Course\Module 5\DEMO\demo 2\env4\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Agent

```bash
python market_research_agent_web.py
```

---

## 🧠 How It Works

1. **User enters a market research question**
2. Agent performs a **live web search** using DuckDuckGo
3. Top 3 URLs are fetched, cleaned, and summarized
4. These are sent to **Cohere's RAG LLM (`command-r-plus`)**
5. Agent returns a **concise, reasoned answer**

### Example Interaction

```
🧑‍💼 Enter your market research question (or 'exit'): What are the latest AI trends in retail?

📘 Retrieved:
- https://example1.com
- https://example2.com
- https://example3.com

🤖 Agentic RAG Answer:
"Recent AI trends in retail include the use of LLM-powered customer service chatbots, predictive inventory optimization, and hyper-personalized marketing through AI insights."
```

---

## 💡 Agentic RAG Concepts Used

* **Live retrieval** from external sources
* **Dynamic reasoning** based on real-time content
* **Multi-step agent architecture** (retrieval + generation)
* **Document-grounded generation** (not hallucinated)

---

## 🛠 Tech Stack

| Component      | Tool / API          |
| -------------- | ------------------- |
| Language Model | Cohere (Command-R+) |
| Web Search     | DuckDuckGo (HTML)   |
| Env Handling   | `python-dotenv`     |
| Parsing        | `beautifulsoup4`    |
| HTTP           | `requests`          |

---

## ✅ Requirements

```
cohere
requests
beautifulsoup4
python-dotenv
```

Installed via:

```bash
pip install -r requirements.txt
```

---

## 📌 To Do (Next Steps)

* [ ] Add conversation memory
* [ ] Add fallback search (Bing or Google via SerpAPI)
* [ ] Streamlit or Gradio UI
* [ ] PDF/CSV output generation
* [ ] Pluggable tools (charts, visual summaries)

---

## 📃 License

This project is open source and free to use for educational and non-commercial use.