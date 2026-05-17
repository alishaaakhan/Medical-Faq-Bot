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
    page_title="Medical FAQ Bot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CSS (PREMIUM UI) ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top, #e0f2fe, #f8fafc, #ffffff);
}

/* Hide default UI */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* HERO */
.hero {
    text-align: center;
    padding: 20px;
}

.hero h1 {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(90deg, #0ea5e9, #2563eb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    font-size: 18px;
    color: #475569;
}

/* CARDS */
.card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-6px);
}

/* UPLOAD */
.upload {
    background: linear-gradient(135deg, #2563eb, #0ea5e9);
    padding: 20px;
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
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.02);
}

/* ANSWER */
.answer {
    background: linear-gradient(135deg, #dcfce7, #ecfdf5);
    border-left: 6px solid #22c55e;
    padding: 20px;
    border-radius: 15px;
    font-size: 16px;
    margin-top: 10px;
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
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------

st.markdown("""
<div class="hero">
    <h1>🏥 Medical FAQ AI Bot</h1>
    <p>AI-powered Healthcare Assistant using RAG + Gemini + FAISS</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Settings")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.sidebar.success("Gemini API Connected")
except:
    st.sidebar.error("Missing API Key")
    st.stop()

st.sidebar.markdown("### 📌 Upload Medical PDFs")
st.sidebar.info("""
Admission, Fees, Hospital FAQ, WHO Guidelines
""")

# ---------------- FEATURE CARDS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>📄 Documents</h3>
        <p>Upload medical PDFs</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>🧠 AI Search</h3>
        <p>Gemini + FAISS RAG system</p>
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
📤 Upload Medical FAQ / Admission / Hospital PDF
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type="pdf")

# ---------------- PROCESS ----------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)

        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        # chunking
        def chunk_text(text, chunk_size=700, overlap=100):
            chunks = []
            start = 0
            while start < len(text):
                chunks.append(text[start:start+chunk_size])
                start += chunk_size - overlap
            return chunks

        texts = chunk_text(full_text)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        st.success(f"Processed {len(texts)} medical chunks")

        # ---------------- QUESTION ----------------

        st.markdown("### 💬 Ask Your Question")

        question = st.text_input("", placeholder="e.g. What is MBBS fee structure?")

        if st.button("Get Answer"):

            if question.strip() == "":
                st.warning("Enter a question")
            else:

                q_emb = model.encode([question]).astype("float32")
                distances, indices = index.search(q_emb, k=4)

                context = "\n\n".join([texts[i] for i in indices[0]])

                prompt = f"""
You are a Medical FAQ AI Assistant.

Rules:
- Answer only from context
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

                with st.spinner("Generating answer..."):
                    response = llm.generate_content(prompt)

                st.markdown("### ✅ Answer")
                st.markdown(f"""
                <div class="answer">
                {response.text}
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📚 Source Context"):
                    st.write(context[:2000])

                st.warning("""
⚠️ Medical Disclaimer:
This bot provides information only from uploaded documents.
Always consult a certified doctor.
""")

    finally:
        os.unlink(tmp_path)

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
Made with ❤️ using Streamlit + Gemini + FAISS
</div>
""", unsafe_allow_html=True)
