from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

client = Groq()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="vectorstore")
collection = chroma_client.get_collection("shakespeare")

SYSTEM_PROMPT = """You are William Shakespeare himself. Speak in Early Modern English,
using thee, thou, thy, hath, doth, and poetic flourishes. Draw on the provided
excerpts from your own works when relevant to inform your response, but always
stay in character."""

def chat(user_input):
    query_embedding = embedder.encode([user_input]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    context = "\n\n".join(results["documents"][0])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Relevant excerpts from thy works:\n{context}\n\nUser says: {user_input}"}
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    return response.choices[0].message.content

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    print("Shakespeare:", chat(user_input))