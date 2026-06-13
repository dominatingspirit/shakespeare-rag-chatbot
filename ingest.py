from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Load text
with open("data/shakespeare_works.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Chunk it
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

# Embed locally (free)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks).tolist()

# Store in ChromaDB
client = chromadb.PersistentClient(path="vectorstore")
collection = client.get_or_create_collection("shakespeare")

batch_size = 1000

for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i:i+batch_size]
    batch_embeddings = embeddings[i:i+batch_size]
    batch_ids = [f"chunk_{j}" for j in range(i, i+len(batch_chunks))]

    collection.add(
        documents=batch_chunks,
        embeddings=batch_embeddings,
        ids=batch_ids
    )
    print(f"Added batch {i} to {i+len(batch_chunks)}")

print(f"Ingested {len(chunks)} chunks.")