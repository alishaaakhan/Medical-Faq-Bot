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
    page_title="MediAI - Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- ULTRA MODERN UI ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* BACKGROUND */
.stApp {
    background: radial-gradient(circle at top, #dbeafe, #f8fafc, #ffffff);
}

/* HIDE STREAMLIT UI */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* HERO */
.hero {
    text-align: center;
    padding: 25px 10px;
}

.hero h1 {
    font-size: 58px;
    font-weight: 800;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    font-size: 18px;
    color: #475569;
}

/* GLASS CARD */
.glass {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    border: 1px solid rgba(255,255,255,0.4);
}

/* FEATURE CARD */
.card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-6px);
}

/* UPLOAD BOX */
.upload {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    padding: 22px;
    border-radius: 18px;
    color: white;
    text-align: center;
    font-weight: 600;
}

/* INPUT */
.stTextInput input {
    padding: 14px;
    border-radius: 12px;
    border: 2px solid #e2e8f0;
}

/* BUTTON */
.stButton>button {
    width: 100%;
    padding: 14px;
    border-radius: 12px;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: white;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
}

/* ANSWER CARD */
.answer {
    background: linear-gradient(135deg, #dcfce7, #f0fdf4);
    border-left: 6px solid #22c55e;
    padding: 20px;
    border-radius: 16px;
    font-size: 16px;
    margin-top: 10px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.06);
}

/* QUESTION BOX */
.qbox {
    background: white;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.05);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 40px;
    color: #64748b;
    font-size: 14px;
}

/* CHAT STYLE */
.chat-title {
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------

st.markdown("""
<div class="hero">
    <h1>🏥 MediAI Assistant</h1>
    <p>Smart Medical FAQ Bot powered by Gemini + FAISS + RAG System</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Control Panel")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.sidebar.success("Gemini Connected")
except:
    st.sidebar.error("Missing API Key")
    st.stop()

st.sidebar.markdown("### 📌 Upload Medical Data")
st.sidebar.info("""
Admission • Fees • Hospital FAQ • WHO Docs
""")

# ---------------- FEATURE ROW ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>📄 PDF Docs</h3>
        <p>Upload medical knowledge base</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>🧠 AI Search</h3>
        <p>FAISS + Gemini RAG engine</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3>⚕️ Safe Answers</h3>
        <p>Only trusted document-based responses</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- UPLOAD ----------------

st.markdown("""
<div class="upload">
📤 Upload Medical PDF (Admission / Fees / Hospital FAQ)
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type="pdf")

# ---------------- PROCESS PDF ----------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)

    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    def chunk_text(text, size=700, overlap=120):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+size])
            start += size - overlap
        return chunks

    texts = chunk_text(full_text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    st.success(f"Knowledge Base Ready: {len(texts)} chunks loaded")

    # ---------------- QUESTION ----------------

    st.markdown('<div class="qbox">', unsafe_allow_html=True)

    st.markdown("### 💬 Ask Your Medical Question")

    question = st.text_input("", placeholder="Example: What are MBBS admission fees?")

    if st.button("Get Answer 🚀"):

        if question.strip() == "":
            st.warning("Please enter a question")
        else:

            q_emb = model.encode([question]).astype("float32")
            _, idx = index.search(q_emb, k=4)

            context = "\n\n".join([texts[i] for i in idx[0]])

            prompt = f"""
You are a Medical FAQ AI Assistant.

Rules:
- Answer ONLY from context
- If not found say "Not available in documents"
- No hallucination
- Add safety disclaimer

Context:
{context}

Question:
{question}

Answer:
"""

            llm = genai.GenerativeModel("gemini-2.5-flash-lite")

            with st.spinner("Thinking like a doctor... 🧠"):
                response = llm.generate_content(prompt)

            st.markdown("### 🩺 AI Answer")

            st.markdown(f"""
            <div class="answer">
            {response.text}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📚 Source Context"):
                st.write(context[:2000])

            st.warning("""
⚠️ Medical Disclaimer:
This bot is for informational purposes only.
Always consult a qualified doctor.
""")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
Made with ❤️ using Streamlit • Gemini • FAISS • RAG Architecture
</div>
""", unsafe_allow_html=True)
