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
    page_title="MediVerse AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PREMIUM CSS ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* MAIN BACKGROUND */

.stApp {

    background-image:

    linear-gradient(
        rgba(240,248,255,0.82),
        rgba(240,248,255,0.86)
    ),

    url("https://images.unsplash.com/photo-1579684385127-1ef15d508118?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* HIDE STREAMLIT */

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* TOP BAR */

.topbar {

    width: 100%;

    padding: 16px;

    border-radius: 24px;

    background: linear-gradient(
        90deg,
        rgba(37,99,235,0.95),
        rgba(6,182,212,0.95),
        rgba(20,184,166,0.95)
    );

    color: white;

    text-align: center;

    font-size: 16px;

    font-weight: 700;

    margin-bottom: 25px;

    box-shadow: 0 12px 30px rgba(37,99,235,0.28);
}

/* HERO */

.hero-container {

    background: rgba(255,255,255,0.55);

    backdrop-filter: blur(20px);

    border-radius: 35px;

    padding: 50px;

    text-align: center;

    box-shadow: 0 14px 45px rgba(0,0,0,0.12);

    border: 1px solid rgba(255,255,255,0.35);

    margin-bottom: 30px;
}

.hero-badge {

    display: inline-block;

    padding: 12px 26px;

    border-radius: 30px;

    background: linear-gradient(
        90deg,
        #2563eb,
        #06b6d4,
        #14b8a6
    );

    color: white;

    font-size: 15px;

    font-weight: 700;

    margin-bottom: 28px;

    box-shadow: 0 10px 25px rgba(37,99,235,0.28);
}

.hero-title {

    font-size: 100px;

    font-weight: 900;

    line-height: 1;

    margin-bottom: 15px;

    background: linear-gradient(
        90deg,
        #2563eb,
        #0891b2,
        #06b6d4,
        #14b8a6,
        #22c55e
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {

    margin: auto;

    width: 82%;

    background: rgba(255,255,255,0.72);

    padding: 20px;

    border-radius: 24px;

    color: #0f172a;

    font-size: 22px;

    font-weight: 600;

    box-shadow: 0 10px 25px rgba(0,0,0,0.06);
}

/* MAIN GLASS */

.glass {

    background: rgba(255,255,255,0.66);

    backdrop-filter: blur(24px);

    border-radius: 36px;

    padding: 40px;

    border: 1px solid rgba(255,255,255,0.35);

    box-shadow: 0 14px 45px rgba(0,0,0,0.14);
}

/* FEATURE CARDS */

.card {

    background: rgba(255,255,255,0.94);

    border-radius: 30px;

    padding: 34px;

    text-align: center;

    height: 290px;

    transition: 0.4s;

    box-shadow: 0 14px 35px rgba(0,0,0,0.08);
}

.card:hover {

    transform: translateY(-14px) scale(1.03);
}

.icon {

    font-size: 72px;

    margin-bottom: 14px;
}

.card h3 {

    font-size: 30px;

    color: #0f172a;

    margin-bottom: 10px;
}

.card p {

    color: #475569;

    line-height: 1.9;

    font-size: 15px;
}

/* UPLOAD BOX */

.upload-box {

    background: linear-gradient(
        135deg,
        #2563eb,
        #0891b2,
        #06b6d4,
        #14b8a6
    );

    padding: 40px;

    border-radius: 30px;

    text-align: center;

    color: white;

    margin-bottom: 30px;

    box-shadow: 0 16px 40px rgba(37,99,235,0.30);
}

.upload-box h2 {

    font-size: 42px;

    margin-bottom: 12px;
}

/* QUESTION BOX */

.question-box {

    background: rgba(255,255,255,0.94);

    padding: 30px;

    border-radius: 26px;

    margin-top: 20px;

    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
}

/* INPUT */

.stTextInput input {

    background: white;

    color: #111827;

    border-radius: 18px;

    padding: 22px;

    border: 2px solid #bfdbfe;

    font-size: 18px;
}

/* BUTTON */

.stButton>button {

    width: 100%;

    padding: 18px;

    border-radius: 18px;

    border: none;

    color: white;

    font-size: 18px;

    font-weight: 700;

    background: linear-gradient(
        90deg,
        #2563eb,
        #0891b2,
        #06b6d4,
        #14b8a6
    );

    box-shadow: 0 10px 25px rgba(37,99,235,0.30);

    transition: 0.35s;
}

.stButton>button:hover {

    transform: scale(1.03);
}

/* ANSWER BOX */

.answer-box {

    background: rgba(255,255,255,0.96);

    border-left: 10px solid #22c55e;

    border-radius: 28px;

    padding: 35px;

    margin-top: 20px;

    font-size: 17px;

    line-height: 2;

    color: #111827;

    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
}

/* SIDEBAR */

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

/* MEDICAL SHOWCASE */

.medical-showcase {

    margin-top: 60px;

    padding: 40px;

    border-radius: 35px;

    background: rgba(255,255,255,0.15);

    backdrop-filter: blur(18px);

    border: 1px solid rgba(255,255,255,0.25);

    box-shadow: 0 10px 40px rgba(0,0,0,0.12);
}

.showcase-title {

    text-align: center;

    font-size: 48px;

    font-weight: 800;

    margin-bottom: 10px;

    background: linear-gradient(
        90deg,
        #2563eb,
        #06b6d4,
        #14b8a6
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.showcase-subtitle {

    text-align: center;

    color: #334155;

    font-size: 18px;

    margin-bottom: 40px;
}

/* GRID */

.medical-grid {

    display: grid;

    grid-template-columns: repeat(4, 1fr);

    gap: 25px;
}

/* MEDICAL CARDS */

.medical-card {

    background: rgba(255,255,255,0.92);

    border-radius: 28px;

    padding: 30px;

    text-align: center;

    transition: 0.4s;

    box-shadow: 0 12px 25px rgba(0,0,0,0.08);

    position: relative;

    overflow: hidden;
}

.medical-card:hover {

    transform: translateY(-12px) scale(1.03);

    box-shadow: 0 18px 40px rgba(37,99,235,0.22);
}

.medical-card::before {

    content: "";

    position: absolute;

    width: 200px;
    height: 200px;

    background: rgba(37,99,235,0.10);

    border-radius: 50%;

    top: -60px;
    right: -60px;
}

.medical-icon {

    font-size: 65px;

    margin-bottom: 15px;

    animation: float 3s ease-in-out infinite;
}

.medical-card h3 {

    font-size: 24px;

    color: #0f172a;

    margin-bottom: 10px;
}

.medical-card p {

    color: #475569;

    font-size: 15px;

    line-height: 1.8;
}

/* FLOAT ANIMATION */

@keyframes float {

    0% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(-10px);
    }

    100% {
        transform: translateY(0px);
    }
}

/* FOOTER */

.footer {

    text-align: center;

    margin-top: 45px;

    font-size: 16px;

    font-weight: 700;

    color: #0f172a;
}

.section-title {

    color: #0f172a;

    font-size: 34px;

    font-weight: 800;
}

/* RESPONSIVE */

@media(max-width: 1000px){

    .medical-grid {

        grid-template-columns: repeat(2,1fr);
    }
}

@media(max-width: 650px){

    .medical-grid {

        grid-template-columns: 1fr;
    }

    .hero-title {

        font-size: 58px;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- TOP BAR ----------------

st.markdown("""
<div class="topbar">
🚀 Next Generation Medical FAQ Assistant • Gemini AI • FAISS • RAG Architecture
</div>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------

st.markdown("""
<div class="hero-container">

    <div class="hero-badge">
    🚀 NEXT GENERATION AI HEALTHCARE ASSISTANT
    </div>

    <div class="hero-title">
    🩺 MediVerse AI
    </div>

    <div class="hero-subtitle">

    ✨ Intelligent Medical FAQ Assistant powered by
    <span style="color:#2563eb;font-weight:800;">Gemini AI</span>
    +
    <span style="color:#0891b2;font-weight:800;">FAISS</span>
    +
    <span style="color:#14b8a6;font-weight:800;">RAG Technology</span>

    </div>

</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ MediVerse Control Center")

try:

    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    st.sidebar.success("✅ Gemini API Connected")

except:

    st.sidebar.error("❌ API Key Missing")

    st.stop()

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

# ---------------- FEATURE CARDS ----------------

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="card">
        <div class="icon">📄</div>
        <h3>Medical Knowledge</h3>
        <p>
        Upload trusted healthcare PDFs and create your intelligent AI medical assistant.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="card">
        <div class="icon">🧠</div>
        <h3>AI Semantic Search</h3>
        <p>
        Advanced semantic retrieval powered by Gemini AI and FAISS vector search.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="card">
        <div class="icon">⚕️</div>
        <h3>Safe Healthcare Answers</h3>
        <p>
        Provides safe and professional healthcare responses from uploaded documents.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- MAIN GLASS ----------------

st.markdown('<div class="glass">', unsafe_allow_html=True)

# ---------------- UPLOAD SECTION ----------------

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
    type="pdf"
)

# ---------------- PDF PROCESS ----------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

        tmp.write(uploaded_file.read())

        tmp_path = tmp.name

    try:

        with st.spinner("📚 Extracting medical knowledge from PDF..."):

            reader = PdfReader(tmp_path)

            full_text = ""

            for page in reader.pages:

                text = page.extract_text()

                if text:

                    full_text += text + "\\n"

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

        with st.spinner("🧠 Building AI medical knowledge base..."):

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
            f"✅ Medical knowledge base ready with {len(texts)} intelligent chunks."
        )

        # ---------------- QUESTION ----------------

        st.markdown("""
        <div class="question-box">
            <h2 class="section-title">💬 Ask Your Medical Question</h2>
        </div>
        """, unsafe_allow_html=True)

        question = st.text_input(
            "",
            placeholder="Example: What are the symptoms of pneumonia?"
        )

        # ---------------- ANSWER BUTTON ----------------

        if st.button("🚀 Generate AI Medical Answer"):

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

                context = "\\n\\n".join(
                    [texts[i] for i in indices[0]]
                )

                prompt = f'''
You are an advanced AI Medical FAQ Assistant.

STRICT RULES:
1. Answer ONLY from the provided context.
2. If answer not found say:
"The information is not available in the uploaded medical documents."
3. Never generate fake medical advice.
4. Keep responses professional and safe.
5. Add a short medical disclaimer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
'''

                try:

                    llm = genai.GenerativeModel(
                        "gemini-2.5-flash-lite"
                    )

                    with st.spinner("🤖 AI is analyzing medical information..."):

                        response = llm.generate_content(prompt)

                    st.markdown("## 🩺 AI Medical Answer")

                    st.markdown(
                        f'''
                        <div class="answer-box">
                        {response.text}
                        </div>
                        ''',
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

        os.unlink(tmp_path)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ANIMATED MEDICAL SHOWCASE ----------------

st.markdown("""
<div class="medical-showcase">

    <div class="showcase-title">
    🩺 AI Healthcare Intelligence
    </div>

    <div class="showcase-subtitle">
    Explore powerful AI-driven healthcare capabilities with futuristic medical assistance
    </div>

    <div class="medical-grid">

        <div class="medical-card">

            <div class="medical-icon">🧬</div>

            <h3>Disease Analysis</h3>

            <p>
            AI-powered healthcare understanding using uploaded medical documents and guidelines.
            </p>

        </div>

        <div class="medical-card">

            <div class="medical-icon">💊</div>

            <h3>Treatment Guidance</h3>

            <p>
            Retrieve trusted treatment information from healthcare PDFs and WHO guidelines.
            </p>

        </div>

        <div class="medical-card">

            <div class="medical-icon">🩻</div>

            <h3>Medical Knowledge</h3>

            <p>
            Smart semantic retrieval using Gemini AI and FAISS vector intelligence.
            </p>

        </div>

        <div class="medical-card">

            <div class="medical-icon">⚕️</div>

            <h3>Safe Healthcare AI</h3>

            <p>
            Professional and secure medical FAQ assistant with reliable document-based answers.
            </p>

        </div>

    </div>

</div>
""", unsafe_allow_html=True)

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
Made by Alisha Khan ❤️
</div>
""", unsafe_allow_html=True)
