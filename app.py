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

# ================= PREMIUM ULTRA UI =================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* ================= MAIN BACKGROUND ================= */

.stApp {

    background:
    linear-gradient(
        rgba(2,6,23,0.90),
        rgba(15,23,42,0.93)
    ),

    url("https://images.unsplash.com/photo-1666214280557-f1b5022eb634?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;

    background-position: center;

    background-attachment: fixed;

    background-repeat: no-repeat;
}

/* ================= REMOVE STREAMLIT ================= */

#MainMenu,
header,
footer {
    visibility: hidden;
}

/* ================= HERO SECTION ================= */

.hero {

    text-align: center;

    padding-top: 20px;

    padding-bottom: 40px;
}

.hero h1 {

    font-size: 95px;

    font-weight: 800;

    line-height: 1;

    background: linear-gradient(
        90deg,
        #60a5fa,
        #38bdf8,
        #22d3ee,
        #2dd4bf,
        #4ade80
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    text-shadow:
    0 0 45px rgba(34,211,238,0.4);

    margin-bottom: 12px;
}

.hero p {

    font-size: 24px;

    color: #dbeafe;

    font-weight: 500;

    letter-spacing: 1px;
}

/* ================= GLASS EFFECT ================= */

.glass {

    background: rgba(15,23,42,0.58);

    backdrop-filter: blur(28px);

    border-radius: 36px;

    padding: 42px;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
    0 12px 45px rgba(0,0,0,0.45),
    0 0 25px rgba(34,211,238,0.08);
}

/* ================= FEATURE CARDS ================= */

.card {

    background: rgba(15,23,42,0.90);

    border-radius: 30px;

    padding: 32px;

    text-align: center;

    height: 280px;

    border: 1px solid rgba(255,255,255,0.06);

    transition: 0.4s;

    box-shadow:
    0 10px 30px rgba(0,0,0,0.35);
}

.card:hover {

    transform: translateY(-12px) scale(1.02);

    border: 1px solid rgba(34,211,238,0.35);

    box-shadow:
    0 18px 45px rgba(34,211,238,0.22);
}

.icon {

    font-size: 65px;

    margin-bottom: 18px;

    filter: drop-shadow(0 0 10px rgba(34,211,238,0.4));
}

.card h3 {

    color: #f8fafc;

    font-size: 28px;

    margin-bottom: 12px;
}

.card p {

    color: #cbd5e1;

    font-size: 15px;

    line-height: 1.9;
}

/* ================= UPLOAD BOX ================= */

.upload-box {

    background: linear-gradient(
        135deg,
        rgba(37,99,235,0.96),
        rgba(14,165,233,0.96),
        rgba(6,182,212,0.96),
        rgba(20,184,166,0.96)
    );

    border-radius: 30px;

    padding: 38px;

    text-align: center;

    color: white;

    margin-bottom: 30px;

    box-shadow:
    0 15px 40px rgba(34,211,238,0.28);
}

.upload-box h2 {

    font-size: 38px;

    margin-bottom: 12px;
}

.upload-box p {

    font-size: 16px;

    line-height: 1.8;
}

/* ================= QUESTION BOX ================= */

.question-box {

    background: rgba(15,23,42,0.88);

    border-radius: 28px;

    padding: 30px;

    margin-top: 20px;

    border: 1px solid rgba(255,255,255,0.06);

    box-shadow:
    0 10px 30px rgba(0,0,0,0.30);
}

/* ================= SECTION TITLE ================= */

.section-title {

    color: #f8fafc;

    font-size: 32px;

    font-weight: 700;
}

/* ================= INPUT ================= */

.stTextInput input {

    background-color: rgba(15,23,42,0.95) !important;

    color: white !important;

    border-radius: 20px !important;

    padding: 20px !important;

    border: 2px solid rgba(34,211,238,0.18) !important;

    font-size: 17px !important;

    transition: 0.3s;
}

.stTextInput input:focus {

    border: 2px solid #22d3ee !important;

    box-shadow:
    0 0 22px rgba(34,211,238,0.28) !important;
}

/* ================= FILE UPLOADER ================= */

[data-testid="stFileUploader"] {

    background: rgba(15,23,42,0.82);

    border-radius: 24px;

    padding: 22px;

    border: 2px dashed rgba(34,211,238,0.30);

    box-shadow:
    0 10px 25px rgba(0,0,0,0.2);
}

/* ================= BUTTON ================= */

.stButton>button {

    width: 100%;

    padding: 18px;

    border-radius: 20px;

    background: linear-gradient(
        90deg,
        #2563eb,
        #0ea5e9,
        #06b6d4,
        #14b8a6
    );

    color: white;

    border: none;

    font-size: 19px;

    font-weight: 700;

    transition: 0.35s;

    box-shadow:
    0 12px 35px rgba(34,211,238,0.25);
}

.stButton>button:hover {

    transform: scale(1.03);

    box-shadow:
    0 18px 45px rgba(34,211,238,0.35);
}

/* ================= ANSWER BOX ================= */

.answer-box {

    background: rgba(15,23,42,0.96);

    border-left: 8px solid #22d3ee;

    border-radius: 28px;

    padding: 32px;

    color: #f8fafc;

    font-size: 17px;

    line-height: 2;

    margin-top: 22px;

    box-shadow:
    0 12px 35px rgba(0,0,0,0.35);
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        #020617,
        #0f172a,
        #111827
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* ================= SUCCESS/WARNING ================= */

.stSuccess,
.stWarning,
.stInfo {

    border-radius: 18px;
}

/* ================= FLOATING BOT ================= */

.avatar {

    position: fixed;

    bottom: 25px;

    right: 25px;

    width: 100px;

    height: 100px;

    border-radius: 50%;

    background: linear-gradient(
        135deg,
        #2563eb,
        #0ea5e9,
        #06b6d4,
        #14b8a6
    );

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 50px;

    box-shadow:
    0 0 40px rgba(34,211,238,0.45);

    animation: float 3s ease-in-out infinite;

    z-index: 999;
}

@keyframes float {

    0% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(-14px);
    }

    100% {
        transform: translateY(0px);
    }
}

/* ================= FOOTER ================= */

.footer {

    text-align: center;

    margin-top: 40px;

    color: #dbeafe;

    font-size: 16px;

    font-weight: 600;
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

# ================= UPLOAD BOX =================

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
                    full_text += text + "\\n"

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

                context = "\\n\\n".join(
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
Made with ❤️ by Alisha Khan
</div>
""", unsafe_allow_html=True)
