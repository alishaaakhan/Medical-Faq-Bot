import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from pypdf import PdfReader

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="MediCare AI Assistant",
    page_icon="🏥",
    layout="wide"
)

# ---------------- BACKGROUND IMAGE ----------------

def add_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1588776814546-1ffcf47267a5");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* GLASS EFFECT */
        .glass {
            background: rgba(255,255,255,0.75);
            backdrop-filter: blur(14px);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }

        /* HERO TITLE */
        .title {
            font-size: 60px;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(90deg,#2563eb,#06b6d4,#22c55e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #1f2937;
            margin-bottom: 25px;
        }

        /* CARDS */
        .card {
            background: rgba(255,255,255,0.85);
            padding: 18px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            transition: 0.3s;
        }

        .card:hover {
            transform: translateY(-6px);
        }

        /* BUTTON */
        .stButton>button {
            background: linear-gradient(90deg,#2563eb,#06b6d4);
            color: white;
            padding: 12px;
            border-radius: 12px;
            font-weight: 600;
            border: none;
            width: 100%;
        }

        .stButton>button:hover {
            transform: scale(1.03);
        }

        /* ANSWER BOX */
        .answer {
            background: rgba(255,255,255,0.9);
            border-left: 6px solid #22c55e;
            padding: 20px;
            border-radius: 15px;
            font-size: 16px;
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg,#0f172a,#1e293b);
        }

        section[data-testid="stSidebar"] * {
            color: white !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

add_bg()

# ---------------- TITLE ----------------

st.markdown("""
<div class="glass">
    <div class="title">🏥 MediCare AI Assistant</div>
    <div class="subtitle">Smart Medical FAQ Bot using Gemini + FAISS + RAG System</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Control Panel")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.sidebar.success("Gemini Connected")
except:
    st.sidebar.error("API Key Missing")
    st.stop()

st.sidebar.markdown("""
📌 Upload Medical Documents:
- Admission Details
- Fee Structure
- Hospital FAQ
- WHO Guidelines
""")

# ---------------- FEATURE CARDS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>📄 Medical Docs</h3>
        <p>Upload PDF knowledge base</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>🧠 AI Engine</h3>
        <p>Gemini + FAISS RAG system</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3>⚕️ Safe Answers</h3>
        <p>Only trusted medical info</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- UPLOAD ----------------

st.markdown("## 📤 Upload Medical PDF")

uploaded_file = st.file_uploader("", type="pdf")

# ---------------- PROCESS ----------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)

    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    def chunk_text(text, size=700, overlap=120):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+size])
            start += size - overlap
        return chunks

    chunks = chunk_text(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    st.success(f"Medical Knowledge Loaded: {len(chunks)} chunks")

    # ---------------- QUESTION ----------------

    st.markdown("## 💬 Ask Medical Question")

    question = st.text_input("", placeholder="Example: What are MBBS admission fees?")

    if st.button("Get Answer 🚀"):

        q_emb = model.encode([question]).astype("float32")
        _, idx = index.search(q_emb, k=4)

        context = "\n\n".join([chunks[i] for i in idx[0]])

        prompt = f"""
You are a Medical FAQ AI Assistant.

Rules:
- Answer ONLY from context
- If not found say "Not available in documents"
- No hallucination
- Add medical disclaimer

Context:
{context}

Question:
{question}

Answer:
"""

        llm = genai.GenerativeModel("gemini-2.5-flash-lite")

        with st.spinner("Analyzing medical data..."):
            res = llm.generate_content(prompt)

        st.markdown("## 🩺 AI Answer")

        st.markdown(f"""
        <div class="answer">
        {res.text}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📚 Source Context"):
            st.write(context[:2000])

        st.warning("""
⚠️ Medical Disclaimer:
This bot is for informational purposes only.
Always consult a certified healthcare professional.
""")

# ---------------- FOOTER ----------------

st.markdown("""
<div style='text-align:center; padding:20px; color:white;'>
Made with ❤️ using Streamlit • Gemini • FAISS • RAG
</div>
""", unsafe_allow_html=True)
