import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
import pandas as pd
import os
import uuid
import math
import re

import torch
print(torch.cuda.is_available())

# Page configuration
st.set_page_config(page_title="Dynamic Coffee Machine Complaint Chatbot", layout="wide")
st.title("Dynamic Coffee Machine Complaint Chatbot")
st.markdown(
    "Ask questions about coffee machine complaints, repairs, or product details. Follow-up questions are supported for a conversational experience."
)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

if "last_context" not in st.session_state:
    st.session_state.last_context = None

# Load CSV data
@st.cache_resource
def load_data():
    csv_file = "final_data.csv"
    if not os.path.exists(csv_file):
        st.error("CSV file not found. Please ensure 'final_data.csv' is in the same directory.")
        return None
    loader = CSVLoader(file_path=csv_file, encoding="utf-8")
    documents = loader.load()
    return documents

# Create or load vector store
@st.cache_resource
def create_vector_store(_documents):
    persist_directory = "./chroma_db"
    # Use a CUDA-enabled model (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}  # This tells transformers to use your GPU
    )

    if os.path.exists(persist_directory):
        st.write("Loading existing vector store from disk...")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    st.write("Creating new vector store. This may take a few minutes...")
    batch_size = 100
    num_batches = math.ceil(len(_documents) / batch_size)
    vector_store = None
    progress_bar = st.progress(0)

    for i in range(0, len(_documents), batch_size):
        batch = _documents[i:i + batch_size]
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            vector_store.add_documents(documents=batch)

        progress_value = min((i + batch_size) / len(_documents), 1.0)
        progress_bar.progress(progress_value)

    vector_store.persist()
    st.success("Vector store created and saved to disk.")
    return vector_store

# Initialize LLM
@st.cache_resource
def initialize_llm():
    return Ollama(model="llama3")

# Setup RAG chain
def setup_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt_template = """
You are a data analytics assistant for a coffee machine company. You have access to a CSV database with columns such as:
{context_columns}

**Rules for answering:**
- Only answer using the data provided in the context (CSV rows).
- For "how many", "which", "what is the average", "trend", "predict", or "compare" questions, use the relevant columns and perform the correct aggregation, filtering, or calculation.
- For "most", "highest", "top", or "frequently" questions, identify the maximum or most common value in the relevant column(s).
- For "per engineer", "per product", "per city/state", or "per month/year" questions, group and summarize the data accordingly.
- For "forecast" or "predict" questions, use recent trends in the data to provide a reasoned estimate (but do not hallucinate).
- For "out of stock" or "low stock", filter parts where Stock or balance is zero or below a threshold.
- For "failure rate", "resolution time", or "workload", calculate using the appropriate columns (e.g., complaint dates, duration, used_qty).
- If the answer is not present in the context, reply: "The information is not available in the data provided."
- Do not use outside knowledge or make up information.
- Use bullet points, tables, or lists for clarity when listing items.
- Be concise and accurate.

**Example question types you can answer:**
- Which spare parts are used the most?
- Which engineer has handled the most complaints?
- What is the average complaint resolution time by engineer?
- Which products fail the most within the warranty period?
- How many complaints were filed in a given month/year?
- Which customers complain the most?
- What is the average usage per engineer or region?
- Which part IDs are out of stock or low on stock?
- What is the failure trend of a specific product over months?
- What is the total quantity vs used quantity across products?
- Which engineer resolves complaints the fastest in a given city/state?
- Which high-priced parts are running low on stock?
- Predict complaint volume next month.
- ...and other analytics-style questions based on the columns.

**Data columns available:** {context_columns}

Conversation History:
{chat_history}

Context (CSV rows):
{context}

User Question:
{question}

Your Answer:
"""

    # Dynamically get columns for prompt context
    def get_context_columns():
        try:
            df = pd.read_csv("final_data.csv")
            return ", ".join(df.columns)
        except Exception:
            return "Unknown (could not load columns)"

    prompt = ChatPromptTemplate.from_template(prompt_template).partial(context_columns=get_context_columns())

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_context(inputs):
        return format_docs(retriever.invoke(inputs["question"]))

    chain = (
        {
            "context": get_context,
            "question": RunnablePassthrough(),
            "chat_history": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def answer_with_pandas(prompt, df, last_context=None):
    prompt_lower = prompt.lower()

    # Example: Handle follow-up context
    if last_context is not None and not last_context.empty:
        # If user says "What about last year?", filter last_context for previous year
        if "last year" in prompt_lower:
            if "cmp_date_year" in last_context.columns:
                last_year = last_context["cmp_date_year"].max() - 1
                filtered = last_context[last_context["cmp_date_year"] == last_year]
                count = filtered.shape[0]
                return f"There were {count} complaints last year."
        # Add more follow-up logic as needed

    # Example: Most used spare part
    if "spare part" in prompt_lower and "used the most" in prompt_lower:
        if "PartName_decoded" in df.columns and "used_qty" in df.columns:
            usage = df.groupby("PartName_decoded")["used_qty"].sum().sort_values(ascending=False)
            return f"Most used spare parts:\n{usage.head(5).to_string()}"

    # Example: Complaint trends per month/year
    if "complaint" in prompt_lower and "trend" in prompt_lower:
        if "cmp_date_year" in df.columns and "cmp_date_month" in df.columns:
            trend = df.groupby(["cmp_date_year", "cmp_date_month"]).size().unstack(fill_value=0)
            return f"Complaint trends per month/year:\n{trend.to_string()}"

    # Engineer-related queries
    if "engineer" in prompt_lower:
        if "how many" in prompt_lower or "total" in prompt_lower or "count" in prompt_lower:
            if "name_decoded" in df.columns:
                unique_engineers = df['name_decoded'].nunique()
                return f"There are {unique_engineers} unique engineers in the data."
            elif "eng_id" in df.columns:
                unique_engineers = df['eng_id'].nunique()
                return f"There are {unique_engineers} unique engineers in the data."
        if "list" in prompt_lower or "show" in prompt_lower or "all" in prompt_lower or "name" in prompt_lower:
            if "name_decoded" in df.columns:
                names = df['name_decoded'].dropna().unique()
                return "Engineers:\n" + "\n".join(str(n) for n in names)
            elif "eng_id" in df.columns:
                ids = df['eng_id'].dropna().unique()
                return "Engineer IDs:\n" + "\n".join(str(i) for i in ids)

    # Complaint-related queries
    if "complaint" in prompt_lower:
        if "how many" in prompt_lower or "total" in prompt_lower or "count" in prompt_lower:
            if "2024" in prompt_lower:
                if "cmp_date_year" in df.columns:
                    count = df[df["cmp_date_year"] == 2024].shape[0]
                    return f"There are {count} complaints in 2024."
            count = df.shape[0]
            return f"There are {count} complaints in the data."
        if "status" in prompt_lower or "completed" in prompt_lower or "pending" in prompt_lower:
            if "status_decoded_x" in df.columns:
                status_counts = df["status_decoded_x"].value_counts()
                return "Complaint status counts:\n" + status_counts.to_string()
        if "list" in prompt_lower or "show" in prompt_lower:
            if "complaint_number" in df.columns:
                numbers = df["complaint_number"].unique()
                return "Complaint Numbers:\n" + "\n".join(str(n) for n in numbers)

    # Product-related queries
    if "product" in prompt_lower:
        if "how many" in prompt_lower or "total" in prompt_lower or "count" in prompt_lower:
            if "Productname_decoded" in df.columns:
                count = df["Productname_decoded"].nunique()
                return f"There are {count} unique products."
        if "list" in prompt_lower or "show" in prompt_lower:
            if "Productname_decoded" in df.columns:
                names = df["Productname_decoded"].unique()
                return "Products:\n" + "\n".join(str(n) for n in names)

    # City/state queries
    if "city" in prompt_lower:
        if "how many" in prompt_lower or "total" in prompt_lower or "count" in prompt_lower:
            if "city_decoded" in df.columns:
                count = df["city_decoded"].nunique()
                return f"There are {count} unique cities."
        if "list" in prompt_lower or "show" in prompt_lower:
            if "city_decoded" in df.columns:
                names = df["city_decoded"].unique()
                return "Cities:\n" + "\n".join(str(n) for n in names)
    if "state" in prompt_lower:
        if "how many" in prompt_lower or "total" in prompt_lower or "count" in prompt_lower:
            if "state_decoded" in df.columns:
                count = df["state_decoded"].nunique()
                return f"There are {count} unique states."
        if "list" in prompt_lower or "show" in prompt_lower:
            if "state_decoded" in df.columns:
                names = df["state_decoded"].unique()
                return "States:\n" + "\n".join(str(n) for n in names)

    # Complaint title/issues
    if "issue" in prompt_lower or "title" in prompt_lower:
        if "how many" in prompt_lower or "total" in prompt_lower or "count" in prompt_lower:
            if "title_decoded" in df.columns:
                count = df["title_decoded"].nunique()
                return f"There are {count} unique complaint issues."
        if "list" in prompt_lower or "show" in prompt_lower:
            if "title_decoded" in df.columns:
                names = df["title_decoded"].unique()
                return "Complaint Issues:\n" + "\n".join(str(n) for n in names)

    # Status queries
    if "status" in prompt_lower:
        if "how many" in prompt_lower or "total" in prompt_lower or "count" in prompt_lower:
            if "status_decoded_x" in df.columns:
                count = df["status_decoded_x"].nunique()
                return f"There are {count} unique complaint statuses."
        if "list" in prompt_lower or "show" in prompt_lower:
            if "status_decoded_x" in df.columns:
                names = df["status_decoded_x"].unique()
                return "Complaint Statuses:\n" + "\n".join(str(n) for n in names)

    # Fallback: show available columns for debugging
    if "column" in prompt_lower or "field" in prompt_lower:
        return "Available columns:\n" + "\n".join(df.columns)

    return None

# Main application logic
def main():
    documents = load_data()
    if documents is None:
        return

    # Load the CSV as a DataFrame for structured queries
    df = pd.read_csv("final_data.csv")

    vector_store = create_vector_store(documents)
    llm = initialize_llm()
    rag_chain = setup_rag_chain(vector_store, llm)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask about coffee machine complaints, repairs, or products..."):
        if not isinstance(prompt, str):
            st.error("Prompt must be a string.")
            print(f"User asked: {prompt}\nError: Prompt must be a string.")
            return
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Try to answer with pandas for structured queries
        pandas_answer = answer_with_pandas(prompt, df, st.session_state.last_context)
        if pandas_answer:
            # If you filtered for a specific context, set last_context to that filtered DataFrame
            # Example: st.session_state.last_context = filtered_df
            # Otherwise, for general questions, you can reset to the full DataFrame or None
            st.session_state.last_context = df  # or filtered_df if you have one
            with st.chat_message("assistant"):
                st.markdown(pandas_answer)
            print(f"User asked: {prompt}\nGenerated answer: {pandas_answer}")
            st.session_state.memory.save_context({"input": prompt}, {"output": pandas_answer})
            st.session_state.messages.append({"role": "assistant", "content": pandas_answer})
            return

        # Otherwise, use RAG/LLM
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
                try:
                    response = rag_chain.invoke({
                        "question": prompt,
                        "chat_history": chat_history
                    })
                    st.markdown(response)
                    print(f"User asked: {prompt}\nGenerated answer: {response}")
                except Exception as e:
                    st.error("An error occurred while generating the answer.")
                    print(f"User asked: {prompt}\nError: {e}")

        # Save to memory for contextual chat
        st.session_state.memory.save_context({"input": prompt}, {"output": response if 'response' in locals() else str(e)})
        st.session_state.messages.append({"role": "assistant", "content": response if 'response' in locals() else str(e)})

if __name__ == "__main__":
    main()
