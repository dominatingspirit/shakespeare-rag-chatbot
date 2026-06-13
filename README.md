# 🎭 Shakespeare AI Chatbot

A RAG-based (Retrieval-Augmented Generation) chatbot that converses in the
style of William Shakespeare, grounded in his complete works.

## How it works

1. Shakespeare's complete works are split into 14831 text chunks
2. Each chunk is converted into embeddings using `sentence-transformers`
   (runs locally, free)
3. Embeddings are stored in a ChromaDB vector database
4. User questions are embedded and matched against the database to find
   relevant passages
5. Retrieved passages + a Shakespeare persona prompt are sent to Groq's
   Llama 3.1 model
6. The model responds in Early Modern English, grounded in real excerpts

## Tech Stack

- Python
- Groq API (LLM)
- sentence-transformers (embeddings)
- ChromaDB (vector database)
- Streamlit (UI)

## Setup

1. Clone the repo
   \`\`\`
   git clone <your-repo-url>
   cd shakespeare-ai
   \`\`\`

2. Create virtual environment and install dependencies
   \`\`\`
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   \`\`\`

3. Create a `.env` file with your Groq API key:
   \`\`\`
   GROQ_API_KEY=your_key_here
   \`\`\`

4. Download Shakespeare's complete works into `data/shakespeare_works.txt`
   \`\`\`
   curl -L -o data/shakespeare_works.txt https://www.gutenberg.org/cache/epub/100/pg100.txt
   \`\`\`

5. Build the vector database (run once)
   \`\`\`
   python ingest.py
   \`\`\`

6. Run the app
   \`\`\`
   streamlit run app.py
   \`\`\`

## Screenshots

![Shakespeare AI UI]('Screenshotui/Screenshot 2026-06-13 at 1.03.26 PM.png')
