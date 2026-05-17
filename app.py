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
    page_title="Medical FAQ AI Bot",
    page_icon="🏥",
    layout="wide"
)

# ---------------- MODERN MEDICAL UI ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* MAIN BACKGROUND */

.stApp {
    background-image: linear-gradient(
        rgba(240,248,255,0.88),
        rgba(240,248,255,0.88)
    ),
    url("https://images.unsplash.com/photo-1576091160550-2173dba999ef?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* HIDE STREAMLIT DEFAULT */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* HERO SECTION */

.hero {
    text-align: center;
    padding-top: 20px;
    padding-bottom: 25px;
}

.hero h1 {
    font-size: 62px;
    font-weight: 800;
    background: linear-gradient(90deg,#2563eb,#0891b2,#22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    font-size: 20px;
    color: #1e293b;
    font-weight: 500;
}

/* GLASS EFFECT */

.glass {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(14px);
    border-radius: 22px;
    padding: 28px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
}

/* FEATURE CARDS */

.card {
    background: rgba(255,255,255,0.88);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-6px);
}

.card h3 {
    color: #0f172a;
}

.card p {
    color: #475569;
}

/* INPUT BOX */

.stTextInput input {
    background-color: white;
    color: black;
    border-radius: 14px;
    padding: 15px;
    border: 2px solid #dbeafe;
    font-size: 16px;
}

/* BUTTON */

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#2563eb,#06b6d4);
    color: white;
    border: none;
    padding: 14px;
    border-radius: 14px;
    font-size: 17px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
}

/* ANSWER BOX */

.answer-box {
    background: rgba(255,255,255,0.92);
    border-left: 7px solid #22c55e;
    padding: 22px;
    border-radius: 16px;
    color: #111827;
    font-size: 16px;
    line-height: 1.8;
    box-shadow: 0 8px 18px rgba(0,0,0,0.08);
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f172a,#1e293b);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #334155;
    margin-top: 35px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------

st.markdown("""
<div class="hero">
    <h1>🏥 Medical FAQ AI Bot</h1>
    <p>AI-powered healthcare assistant using Gemini + FAISS + RAG</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Medical AI Panel")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.sidebar.success("✅ Gemini API Connected")

except:
    st.sidebar.error("❌ API Key Missing")
    st.stop()

st.sidebar.markdown("""
### 📌 Upload Medical Documents

Supported PDFs:
- Medical FAQs
- WHO Guidelines
- Hospital Policies
- Healthcare Manuals
- Medical Encyclopedia
- Treatment Information
""")

# ---------------- FEATURE CARDS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>📄 Medical Knowledge</h3>
        <p>Upload trusted healthcare PDFs and documents</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>🧠 AI Medical Search</h3>
        <p>Semantic search powered by Gemini and FAISS</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3>⚕️ Safe Healthcare Answers</h3>
        <p>Reliable responses only from uploaded documents</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- MAIN GLASS CONTAINER ----------------

st.markdown('<div class="glass">', unsafe_allow_html=True)

st.markdown("## 📤 Upload Medical PDF")

uploaded_file = st.file_uploader(
    "Upload healthcare FAQ or medical documents",
    type="pdf"
)

# ---------------- PDF PROCESS ----------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:

        with st.spinner("📚 Reading medical documents..."):

            reader = PdfReader(tmp_path)

            full_text = ""

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    full_text += text + "\n"

        # ---------------- CHUNKING ----------------

        def chunk_text(text, chunk_size=700, overlap=120):

            chunks = []

            start = 0

            while start < len(text):

                end = start + chunk_size

                chunks.append(text[start:end])

                start += chunk_size - overlap

            return chunks

        texts = chunk_text(full_text)

        # ---------------- EMBEDDINGS ----------------

        with st.spinner("🧠 Building medical knowledge base..."):

            model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

            embeddings = model.encode(
                texts
            ).astype("float32")

            index = faiss.IndexFlatL2(
                embeddings.shape[1]
            )

            index.add(embeddings)

        st.success(
            f"✅ Medical knowledge base ready with {len(texts)} chunks"
        )

        # ---------------- QUESTION ----------------

        st.markdown("## 💬 Ask Your Medical Question")

        question = st.text_input(
            "",
            placeholder="Example: What are the symptoms of diabetes?"
        )

        # ---------------- ANSWER BUTTON ----------------

        if st.button("🔍 Generate Medical Answer"):

            if question.strip() == "":
                st.warning("Please enter a medical question.")

            else:

                q_emb = model.encode(
                    [question]
                ).astype("float32")

                distances, indices = index.search(
                    q_emb,
                    k=4
                )

                context = "\n\n".join(
                    [texts[i] for i in indices[0]]
                )

                # ---------------- PROMPT ----------------

                prompt = f"""
You are an AI Medical FAQ Assistant.

STRICT RULES:
1. Answer ONLY using the provided context.
2. If answer not found say:
"The information is not available in the uploaded medical documents."
3. Never generate fake medical advice.
4. Keep answers safe and professional.
5. Add a short medical disclaimer.
6. Mention source-based response style.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

                try:

                    llm = genai.GenerativeModel(
                        "gemini-2.5-flash-lite"
                    )

                    with st.spinner("🤖 AI is analyzing medical information..."):

                        response = llm.generate_content(prompt)

                    st.markdown("## 🩺 AI Medical Answer")

                    st.markdown(
                        f"""
                        <div class="answer-box">
                        {response.text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # ---------------- SOURCE ----------------

                    with st.expander("📚 View Source Context"):

                        st.write(context[:2000])

                    # ---------------- DISCLAIMER ----------------

                    st.warning("""
⚠️ Medical Disclaimer:
This chatbot provides information only from uploaded documents.
It does not replace professional medical consultation.
Always consult certified healthcare professionals.
""")

                except Exception as e:

                    st.error(f"Gemini Error: {e}")

    finally:

        os.unlink(tmp_path)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
Made with ❤️ using Streamlit • Gemini AI • FAISS • RAG Architecture
</div>
""", unsafe_allow_html=True)
