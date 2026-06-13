import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from sentence_transformers import SentenceTransformer
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import chromadb

import uuid

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Shakespeare AI",
    page_icon="🪶",
    layout="wide"
)

# ---------- CUSTOM STYLING ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=EB+Garamond:wght@400;500;600&display=swap');

    .stApp {
        background-color: #1a1410;
        background-image: radial-gradient(circle at top right, #2a2018 0%, #1a1410 60%);
    }

    * {
        font-family: 'EB Garamond', serif;
    }

    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
    }

    h1 {
        color: #d4af6a;
        text-align: center;
        font-weight: 600 !important;
        letter-spacing: 2px;
        margin-bottom: 0px;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        text-align: center;
        color: #9c8b6e !important;
        font-style: italic;
        letter-spacing: 1px;
    }

    hr {
        border-top: 1px solid #4a3c2c;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #241c15;
        border: 1px solid #3d3122;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #e8dcc4;
        font-size: 16px;
        line-height: 1.6;
    }

    /* User message accent */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-left: 3px solid #6b8e8e;
    }

    /* Assistant message accent */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 3px solid #d4af6a;
    }

    /* Chat input */
    .stChatInput textarea {
        background-color: #241c15 !important;
        color: #e8dcc4 !important;
        border: 1px solid #4a3c2c !important;
        font-size: 16px !important;
    }

    .stChatInput textarea::placeholder {
        color: #6b5d4a !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #15110c;
        border-right: 1px solid #3d3122;
    }

    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #d4af6a !important;
    }

    /* Buttons */
    .stButton button {
        background-color: #2a2018;
        color: #d4af6a;
        border: 1px solid #4a3c2c;
        border-radius: 6px;
        font-family: 'EB Garamond', serif;
        width: 100%;
        text-align: left;
    }

    .stButton button:hover {
        background-color: #3d3122;
        border-color: #d4af6a;
        color: #f0e6d2;
    }

    /* Spinner */
    .stSpinner > div {
        color: #d4af6a !important;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ---------- LOAD RESOURCES ----------
@st.cache_resource
def load_resources():
    client = Groq()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="vectorstore")
    collection = chroma_client.get_collection("shakespeare")
    return client, embedder, collection

client, embedder, collection = load_resources()

SYSTEM_PROMPT = """You are William Shakespeare himself. Speak in Early Modern English,
using thee, thou, thy, hath, doth, and poetic flourishes. Draw on the provided
excerpts from your own works when relevant to inform your response, but always
stay in character."""

def chat(user_input, history):
    query_embedding = embedder.encode([user_input]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    context = "\n\n".join(results["documents"][0])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({
        "role": "user",
        "content": f"Relevant excerpts from thy works:\n{context}\n\nUser says: {user_input}"
    })

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    return response.choices[0].message.content

# ---------- SESSION STATE SETUP ----------
# 'conversations' = dict of {conversation_id: {"title": ..., "messages": [...]}}
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_conversation" not in st.session_state:
    # create a first conversation
    new_id = str(uuid.uuid4())
    st.session_state.conversations[new_id] = {"title": "New Conversation", "messages": []}
    st.session_state.active_conversation = new_id

active_id = st.session_state.active_conversation
active_conv = st.session_state.conversations[active_id]

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🪶 Shakespeare AI")
    st.caption("Conversations")

    if st.button("➕ New Conversation"):
        new_id = str(uuid.uuid4())
        st.session_state.conversations[new_id] = {"title": "New Conversation", "messages": []}
        st.session_state.active_conversation = new_id
        st.rerun()

    st.markdown("---")

    # List existing conversations (most recent first)
    for conv_id, conv in reversed(list(st.session_state.conversations.items())):
        label = conv["title"]
        if conv_id == active_id:
            label = f"📖 {label}"
        else:
            label = f"📜 {label}"

        if st.button(label, key=f"conv_{conv_id}"):
            st.session_state.active_conversation = conv_id
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Delete Current Conversation"):
        del st.session_state.conversations[active_id]
        if not st.session_state.conversations:
            new_id = str(uuid.uuid4())
            st.session_state.conversations[new_id] = {"title": "New Conversation", "messages": []}
            st.session_state.active_conversation = new_id
        else:
            st.session_state.active_conversation = list(st.session_state.conversations.keys())[-1]
        st.rerun()

    st.markdown("---")
    st.caption("Suggested topics")
    st.markdown("""
    <small>
    • What is love, truly?<br>
    • Speak of revenge and forgiveness<br>
    • Tell me of a king who lost everything<br>
    • What is the nature of fate?<br>
    • Describe a storm at sea
    </small>
    """, unsafe_allow_html=True)

# ---------- MAIN UI ----------
st.title("THE BARD'S CHAMBER")
st.caption("Converse with William Shakespeare, drawn from his complete works")
st.markdown("---")

# Display messages for active conversation
for msg in active_conv["messages"]:
    avatar = "🪶" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Chat input
if user_input := st.chat_input("Speak thy mind..."):
    active_conv["messages"].append({"role": "user", "content": user_input})

    # Auto-title the conversation from the first message
    if active_conv["title"] == "New Conversation":
        active_conv["title"] = user_input[:30] + ("..." if len(user_input) > 30 else "")

    with st.chat_message("user", avatar="🧑"):
        st.write(user_input)

    with st.chat_message("assistant", avatar="🪶"):
        with st.spinner("Composing a response in iambic pentameter..."):
            response = chat(user_input, active_conv["messages"][:-1])
        st.write(response)

    active_conv["messages"].append({"role": "assistant", "content": response})
    st.rerun()