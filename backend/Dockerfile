FROM python:3.11-slim
 
WORKDIR /app
 
# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Pre-download NLTK data at build time so it isn't re-downloaded on every container start
RUN python -c "import nltk; [nltk.download(r, quiet=True) for r in ['punkt','punkt_tab','wordnet','stopwords','omw-1.4']]"
 
# Copy app code
COPY diet_chatbot.py .
 
EXPOSE 8000
 
CMD ["uvicorn", "diet_chatbot:app", "--host", "0.0.0.0", "--port", "8000"]