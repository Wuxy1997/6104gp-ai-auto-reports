#!/bin/bash

# Start ollama service
ollama serve &

# Wait for ollama service to start
sleep 5

# Download mistral model
ollama pull mistral

# Start Streamlit application
streamlit run app.py --server.address 0.0.0.0 