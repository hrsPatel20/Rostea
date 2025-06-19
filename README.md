# Dynamic Coffee Machine Complaint Chatbot

This project is a **Streamlit-based chatbot** for analytics and Q&A on coffee machine complaints, repairs, products, engineers, and inventory, powered by Llama 3 (via Ollama), LangChain, and ChromaDB. It supports natural language queries, follow-up questions, and advanced analytics directly from your CSV database.

---

## Features

- **Natural language Q&A** about complaints, repairs, products, engineers, and inventory.
- **Conversational memory** for follow-up questions.
- **Hybrid engine:** Uses pandas for structured queries and LLM+RAG for open-ended analytics.
- **GPU acceleration** for embeddings and Llama 3 (if available).
- **Fully containerized**: All dependencies, Ollama, and Llama 3 run inside Docker.

---

## Quick Start

### 1. **Clone the repository**

```sh
git clone <your-repo-url>
cd Rostea
```

### 2. **Build the Docker image**

```sh
docker build -t rostea .
```

### 3. **Run the container (with GPU support)**

```sh
docker run --gpus all -p 8501:8501 -p 11434:11434 rostea
```

- The app will be available at [http://localhost:8501](http://localhost:8501)
- Ollama's API (for Llama 3) will be available at port 11434 inside the container.

---

## Project Structure

```
.
├── app.py                # Main Streamlit application
├── final_data.csv        # Your coffee machine complaints/products CSV database
├── requirements.txt      # Python dependencies
├── requirements.yml      # (Optional) Conda environment file
├── Dockerfile            # Docker build instructions (includes Ollama + Llama 3)
├── chroma_db/            # ChromaDB vector store (auto-generated)
└── ...
```

---

## How it Works

- **User asks a question** (e.g., "How many engineers are there?").
- If the question is structured (count, list, filter), **pandas** answers directly for accuracy.
- For open-ended or analytics questions, the app uses **LangChain RAG** with Llama 3 via Ollama.
- **All questions and answers are logged to the terminal.**
- **Follow-up questions** use conversation memory and context.

---

## Requirements (if running locally, not in Docker)

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running (`ollama serve`)
- Llama 3 model pulled: `ollama pull llama3`
- NVIDIA GPU + CUDA drivers (for GPU acceleration, optional)

Install Python dependencies:
```sh
pip install -r requirements.txt
```

---

## Customization

- To add more analytics logic, edit `answer_with_pandas()` in `app.py`.
- To change the prompt or LLM behavior, edit `setup_rag_chain()` in `app.py`.
- To use a different LLM, change the model in `initialize_llm()`.

---

## Troubleshooting

- **Ollama or Llama 3 not found?**  
  Make sure Ollama is installed and the model is pulled inside the container (the Dockerfile does this automatically).
- **CUDA not available?**  
  Ensure you run Docker with `--gpus all` and have NVIDIA drivers installed.
- **CSV not found?**  
  Place your `final_data.csv` in the project root.

---

## License

MIT License

---

## Credits

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.com/)
- [Llama 3](https://ollama.com/library/llama3)

