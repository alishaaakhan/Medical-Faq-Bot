import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from pypdf import PdfReader

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="MediCare AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= GEMINI API =================

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("❌ Gemini API Key Missing")
    st.stop()

# ================= PREMIUM DARK MEDICAL UI =================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* ================= BACKGROUND ================= */

.stApp {

    background-image:
    linear-gradient(
        rgba(2,6,23,0.88),
        rgba(15,23,42,0.92)
    ),
    url("https://images.unsplash.com/photo-1587351021759-3e566b6af7cc?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;

    background-position: center;

    background-attachment: fixed;

    background-repeat: no-repeat;
}

/* ================= HIDE STREAMLIT ================= */

#MainMenu,
header,
footer {
    visibility: hidden;
}

/* ================= HERO ================= */

.hero {
    text-align: center;
    padding-top: 10px;
    padding-bottom: 35px;
}

.hero h1 {

    font-size: 88px;

    font-weight: 800;

    background: linear-gradient(
        90deg,
        #60a5fa,
        #22d3ee,
        #2dd4bf,
        #4ade80
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    text-shadow:
    0 0 35px rgba(34,211,238,0.35);

    margin-bottom: 10px;
}

.hero p {

    color: #dbeafe;

    font-size: 23px;

    letter-spacing: 0.5px;

    font-weight: 500;
}

/* ================= GLASS EFFECT ================= */

.glass {

    background: rgba(15,23,42,0.58);

    backdrop-filter: blur(26px);

    border-radius: 34px;

    padding: 40px;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
    0 10px 45px rgba(0,0,0,0.45),
    0 0 25px rgba(34,211,238,0.08);
}

/* ================= FEATURE CARDS ================= */

.card {

    background: rgba(15,23,42,0.88);

    border: 1px solid rgba(255,255,255,0.06);

    border-radius: 28px;

    padding: 30px;

    text-align: center;

    box-shadow:
    0 10px 35px rgba(0,0,0,0.35);

    transition: 0.4s;

    height: 270px;
}

.card:hover {

    transform: translateY(-12px);

    border: 1px solid rgba(34,211,238,0.35);

    box-shadow:
    0 15px 40px rgba(34,211,238,0.22);
}

.icon {

    font-size: 60px;

    margin-bottom: 15px;
}

.card h3 {

    color: #f8fafc;

    font-size: 27px;

    margin-bottom: 10px;
}

.card p {

    color: #cbd5e1;

    line-height: 1.8;

    font-size: 15px;
}

/* ================= UPLOAD SECTION ================= */

.upload-box {

    background: linear-gradient(
        135deg,
        rgba(37,99,235,0.95),
        rgba(14,165,233,0.95),
        rgba(6,182,212,0.95),
        rgba(20,184,166,0.95)
    );

    border-radius: 28px;

    padding: 35px;

    text-align: center;

    color: white;

    margin-bottom: 28px;

    box-shadow:
    0 12px 35px rgba(34,211,238,0.25);
}

.upload-box h2 {

    font-size: 36px;

    margin-bottom: 10px;
}

/* ================= QUESTION BOX ================= */

.question-box {

    background: rgba(15,23,42,0.88);

    border: 1px solid rgba(255,255,255,0.06);

    border-radius: 24px;

    padding: 28px;

    margin-top: 20px;

    box-shadow:
    0 10px 25px rgba(0,0,0,0.28);
}

/* ================= INPUT ================= */

.stTextInput input {

    background-color: rgba(15,23,42,0.95) !important;

    color: #ffffff !important;

    border: 2px solid rgba(34,211,238,0.18) !important;

    border-radius: 18px !important;

    padding: 18px !important;

    font-size: 17px !important;
}

.stTextInput input:focus {

    border: 2px solid #22d3ee !important;

    box-shadow:
    0 0 20px rgba(34,211,238,0.25) !important;
}

/* ================= FILE UPLOADER ================= */

[data-testid="stFileUploader"] {

    background: rgba(15,23,42,0.78);

    border-radius: 22px;

    padding: 18px;

    border: 1px dashed rgba(34,211,238,0.35);
}

/* ================= BUTTON ================= */

.stButton>button {

    width: 100%;

    padding: 16px;

    border-radius: 18px;

    background: linear-gradient(
        90deg,
        #2563eb,
        #0ea5e9,
        #06b6d4,
        #14b8a6
    );

    color: white;

    border: none;

    font-size: 18px;

    font-weight: 700;

    transition: 0.35s;

    box-shadow:
    0 10px 30px rgba(34,211,238,0.22);
}

.stButton>button:hover {

    transform: scale(1.03);

    box-shadow:
    0 15px 40px rgba(34,211,238,0.35);
}

/* ================= ANSWER BOX ================= */

.answer-box {

    background: rgba(15,23,42,0.96);

    color: #f8fafc;

    border-left: 8px solid #22d3ee;

    border-radius: 24px;

    padding: 30px;

    line-height: 2;

    margin-top: 20px;

    box-shadow:
    0 10px 35px rgba(0,0,0,0.35);
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {

    background: linear-gradient(
        180deg,
        #020617,
        #0f172a,
        #111827
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* ================= SECTION TITLE ================= */

.section-title {

    color: #f8fafc;

    font-size: 30px;

    font-weight: 700;
}

/* ================= FOOTER ================= */

.footer {

    text-align: center;

    margin-top: 40px;

    color: #dbeafe;

    font-weight: 600;

    font-size: 16px;
}

/* ================= FLOATING BOT ================= */

.avatar {

    position: fixed;

    bottom: 25px;

    right: 25px;

    width: 95px;

    height: 95px;

    border-radius: 50%;

    background: linear-gradient(
        135deg,
        #2563eb,
        #0ea5e9,
        #06b6d4,
        #14b8a6
    );

    display: flex;

    justify-content: center;

    align-items: center;

    font-size: 45px;

    box-shadow:
    0 0 35px rgba(34,211,238,0.45);

    animation: float 3s ease infinite;

    z-index: 999;
}

@keyframes float {

    0% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(-12px);
    }

    100% {
        transform: translateY(0px);
    }
}

/* ================= SCROLLBAR ================= */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #0f172a;
}

::-webkit-scrollbar-thumb {

    background: linear-gradient(
        #2563eb,
        #06b6d4
    );

    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

# ================= HERO =================

st.markdown("""
<div class="hero">
    <h1>🩺 MediCare AI</h1>
    <p>Next Generation Medical FAQ Assistant</p>
</div>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================

st.sidebar.title("⚙️ MediVerse Control Center")

st.sidebar.success("✅ Gemini API Connected")

st.sidebar.markdown("""
### 📌 Supported Healthcare PDFs

- Medical FAQ Documents
- WHO Guidelines
- Treatment Information
- Hospital Manuals
- Healthcare Policies
- Medical Encyclopedia
""")

st.sidebar.info("""
💡 Example Questions

• What are diabetes symptoms?
• What causes hypertension?
• What is asthma treatment?
• What precautions are used for COVID-19?
""")

# ================= FEATURE CARDS =================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="icon">📄</div>
        <h3>Medical Knowledge Base</h3>
        <p>
        Upload trusted healthcare PDFs and create an intelligent AI-powered medical assistant.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="icon">🧠</div>
        <h3>Semantic AI Search</h3>
        <p>
        Powered by Gemini AI and FAISS for highly accurate medical document retrieval.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="icon">⚕️</div>
        <h3>Safe Healthcare Answers</h3>
        <p>
        Provides professional responses strictly from uploaded medical documents.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ================= MAIN CONTAINER =================

st.markdown('<div class="glass">', unsafe_allow_html=True)

# ================= UPLOAD SECTION =================

st.markdown("""
<div class="upload-box">
    <h2>📤 Upload Medical PDF</h2>
    <p>
    Upload healthcare FAQs, WHO guidelines, treatment documents,
    medical encyclopedia or healthcare manuals.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["pdf"]
)

# ================= PDF PROCESS =================

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

        tmp.write(uploaded_file.read())

        tmp_path = tmp.name

    try:

        with st.spinner("📚 Extracting medical knowledge..."):

            reader = PdfReader(tmp_path)

            full_text = ""

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    full_text += text + "\n"

        # ================= CHUNKING =================

        def chunk_text(text, chunk_size=700, overlap=120):

            chunks = []

            start = 0

            while start < len(text):

                end = start + chunk_size

                chunks.append(text[start:end])

                start += chunk_size - overlap

            return chunks

        texts = chunk_text(full_text)

        # ================= EMBEDDINGS =================

        with st.spinner("🧠 Building AI medical knowledge base..."):

            model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

            embeddings = model.encode(texts)

            embeddings = np.array(embeddings).astype("float32")

            index = faiss.IndexFlatL2(
                embeddings.shape[1]
            )

            index.add(embeddings)

        st.success(
            f"✅ Medical knowledge base ready with {len(texts)} intelligent chunks."
        )

        # ================= QUESTION =================

        st.markdown("""
        <div class="question-box">
            <h2 class="section-title">💬 Ask Your Medical Question</h2>
        </div>
        """, unsafe_allow_html=True)

        question = st.text_input(
            "",
            placeholder="Example: What are the symptoms of pneumonia?"
        )

        # ================= ANSWER BUTTON =================

        if st.button("🚀 Generate AI Medical Answer"):

            if question.strip() == "":
                st.warning("Please enter a medical question.")

            else:

                q_emb = model.encode([question])

                q_emb = np.array(q_emb).astype("float32")

                distances, indices = index.search(
                    q_emb,
                    k=4
                )

                context = "\n\n".join(
                    [texts[i] for i in indices[0]]
                )

                prompt = f"""
You are an advanced AI Medical FAQ Assistant.

STRICT RULES:
1. Answer ONLY using provided context.
2. If answer not found say:
"The information is not available in the uploaded medical documents."
3. Never generate fake medical advice.
4. Keep responses professional and safe.
5. Add a short healthcare disclaimer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

                try:

                    llm = genai.GenerativeModel(
                        "gemini-1.5-flash"
                    )

                    with st.spinner("🤖 AI is analyzing medical information..."):

                        response = llm.generate_content(prompt)

                    answer = getattr(
                        response,
                        "text",
                        "No response generated."
                    )

                    st.markdown("## 🩺 AI Medical Answer")

                    st.markdown(
                        f"""
                        <div class="answer-box">
                        {answer}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    with st.expander("📚 View Source Context"):

                        st.write(context[:2000])

                    st.warning("""
⚠️ Medical Disclaimer:
This chatbot provides information only from uploaded documents.
It does not replace professional medical consultation.
Always consult certified healthcare professionals.
""")

                except Exception as e:

                    st.error(f"Gemini Error: {e}")

    finally:

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

st.markdown("</div>", unsafe_allow_html=True)

# ================= FLOATING BOT =================

st.markdown("""
<div class='avatar'>
🤖
</div>
""", unsafe_allow_html=True)

# ================= FOOTER =================

st.markdown("""
<div class="footer">
Made by Alisha Khan ❤️
</div>
""", unsafe_allow_html=True)
