# Use Nvidia CUDA as base for GPU support
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl git python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama 3 model (this will take time and disk space)
RUN /bin/bash -c "ollama serve & sleep 10 && ollama pull llama3"

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 11434

CMD ollama serve & streamlit run app.py --server.port=8501 --server.address=0.0.0.0