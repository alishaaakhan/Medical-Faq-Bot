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
    page_title="Health Medical FAQ Bot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #dbeafe, #eff6ff, #ffffff);
}

/* Hide Streamlit Menu */

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Hero Section */

.hero {
    text-align: center;
    padding-top: 10px;
    padding-bottom: 25px;
}

.hero-title {
    font-size: 55px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 20px;
    color: #475569;
    margin-bottom: 30px;
}

/* Main Card */

.main-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    padding: 30px;
    border-radius: 25px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.18);
    border: 1px solid rgba(255,255,255,0.4);
}

/* Upload Box */

.upload-box {
    background: linear-gradient(135deg,#0ea5e9,#2563eb);
    padding: 25px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

/* Question Box */

.question-box {
    background: #ffffff;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-top: 20px;
}

/* Answer Box */

.answer-box {
    background: linear-gradient(135deg,#dcfce7,#f0fdf4);
    border-left: 8px solid #22c55e;
    padding: 25px;
    border-radius: 18px;
    color: #111827;
    font-size: 17px;
    line-height: 1.8;
    margin-top: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
}

/* Source Box */

.source-box {
    background: #f8fafc;
    padding: 20px;
    border-radius: 18px;
    color: #111827;
    border: 1px solid #cbd5e1;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f172a,#1e293b);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Buttons */

.stButton>button {
    width: 100%;
    background: linear-gradient(135deg,#2563eb,#0ea5e9);
    color: white;
    border: none;
    padding: 14px;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(135deg,#1d4ed8,#0284c7);
}

/* Input */

.stTextInput>div>div>input {
    background-color: white;
    color: black;
    border-radius: 12px;
    padding: 14px;
    border: 2px solid #cbd5e1;
}

/* Footer */

.footer {
    text-align: center;
    margin-top: 40px;
    color: #64748b;
    font-size: 15px;
}

/* Medical Cards */

.feature-card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
    transition: 0.3s;
}

.feature-card:hover {
    transform: translateY(-5px);
}

.feature-icon {
    font-size: 40px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------

st.markdown("""
<div class="hero">
    <div class="hero-title">🏥 Health Medical FAQ Bot</div>
    <div class="hero-subtitle">
        AI-powered Healthcare Assistant using RAG + Gemini + FAISS
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE CARDS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📄</div>
        <h3>Medical PDFs</h3>
        <p>Upload hospital, admission, fees or healthcare documents.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <h3>AI Assistant</h3>
        <p>Get accurate answers powered by Gemini AI & semantic search.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🛡️</div>
        <h3>Safe Responses</h3>
        <p>Answers only from uploaded documents with safety disclaimer.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Medical Bot Settings")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.sidebar.success("✅ Gemini API Connected")

except:
    st.sidebar.error("❌ API Key Missing")
    st.stop()

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 📌 Upload Documents

Supported PDFs:
- Admission Brochure
- Hospital FAQ
- Medical Guidelines
- Fee Structure
- WHO Documents
- Healthcare Policies
""")

st.sidebar.markdown("---")

st.sidebar.info("""
💡 Example Questions:
- What is the admission process?
- What are the MBBS fees?
- What are OPD timings?
- Hostel facility available?
""")

# ---------------- MAIN CONTAINER ----------------

st.markdown('<div class="main-card">', unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------

st.markdown("""
<div class="upload-box">
    <h2>📤 Upload Medical PDF</h2>
    <p>Drag and drop your medical FAQ or hospital document here</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type="pdf"
)

# ---------------- PDF PROCESS ----------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:

        # ---------------- READ PDF ----------------

        with st.spinner("📚 Extracting medical information..."):

            reader = PdfReader(tmp_path)

            full_text = ""

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    full_text += text + "\n"

        # ---------------- CHUNKING ----------------

        def chunk_text(text, chunk_size=700, overlap=100):

            chunks = []

            start = 0

            while start < len(text):

                end = start + chunk_size

                chunks.append(text[start:end])

                start += chunk_size - overlap

            return chunks

        texts = chunk_text(full_text)

        # ---------------- EMBEDDINGS ----------------

        with st.spinner("🧠 Building AI medical knowledge base..."):

            model_emb = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

            embeddings = model_emb.encode(
                texts
            ).astype("float32")

            index = faiss.IndexFlatL2(
                embeddings.shape[1]
            )

            index.add(embeddings)

        st.success(
            f"✅ Successfully processed PDF with {len(texts)} knowledge chunks."
        )

        # ---------------- QUESTION SECTION ----------------

        st.markdown("""
        <div class="question-box">
            <h3>💬 Ask Your Medical Question</h3>
        </div>
        """, unsafe_allow_html=True)

        question = st.text_input(
            "",
            placeholder="Example: What are the admission fees for MBBS?"
        )

        if st.button("🔍 Generate Answer"):

            if question.strip() == "":
                st.warning("Please enter a question.")

            else:

                # ---------------- SEARCH ----------------

                q_emb = model_emb.encode(
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
You are an advanced Medical FAQ Assistant.

IMPORTANT RULES:
1. Answer ONLY from provided context.
2. If answer not found say:
"The information is not available in the uploaded medical documents."
3. Never generate fake medical advice.
4. Keep answers concise and professional.
5. Add short medical disclaimer.
6. Mention source context reference.

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

                    with st.spinner("🤖 AI is analyzing documents..."):

                        response = llm.generate_content(prompt)

                    st.markdown("""
                    <h2 style='margin-top:20px; color:#0f172a;'>
                    ✅ AI Generated Answer
                    </h2>
                    """, unsafe_allow_html=True)

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

                        st.markdown(
                            f"""
                            <div class="source-box">
                            {context[:2000]}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

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
Made by Alisha Khan❤️ 
</div>
""", unsafe_allow_html=True)
