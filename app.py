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
    page_title="MediSphere AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- ULTIMATE UI ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* MAIN BACKGROUND */

.stApp {

    background-image:
    linear-gradient(
        rgba(240,248,255,0.82),
        rgba(240,248,255,0.86)
    ),

    url("https://images.unsplash.com/photo-1666214280391-8ff5bd3c0bf0?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* REMOVE STREAMLIT DEFAULT */

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* HERO SECTION */

.hero {

    text-align: center;

    padding-top: 15px;
    padding-bottom: 30px;
}

.hero h1 {

    font-size: 90px;
    font-weight: 800;

    line-height: 1;

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

    margin-bottom: 12px;
}

.hero p {

    font-size: 24px;
    color: #0f172a;
    font-weight: 500;
}

/* TOP BANNER */

.top-banner {

    background: linear-gradient(
        90deg,
        rgba(37,99,235,0.9),
        rgba(6,182,212,0.9),
        rgba(20,184,166,0.9)
    );

    color: white;

    padding: 14px;

    border-radius: 18px;

    text-align: center;

    margin-bottom: 20px;

    font-weight: 600;

    box-shadow: 0 10px 25px rgba(37,99,235,0.25);
}

/* GLASS MAIN */

.glass {

    background: rgba(255,255,255,0.68);

    backdrop-filter: blur(22px);

    border-radius: 34px;

    padding: 35px;

    box-shadow: 0 10px 40px rgba(0,0,0,0.14);

    border: 1px solid rgba(255,255,255,0.35);
}

/* FEATURE CARDS */

.card {

    background: rgba(255,255,255,0.94);

    border-radius: 28px;

    padding: 30px;

    text-align: center;

    box-shadow: 0 14px 30px rgba(0,0,0,0.08);

    transition: 0.4s;

    height: 280px;
}

.card:hover {

    transform: translateY(-12px) scale(1.03);
}

.icon {

    font-size: 68px;

    margin-bottom: 12px;
}

.card h3 {

    font-size: 28px;

    color: #0f172a;

    margin-bottom: 10px;
}

.card p {

    color: #475569;

    line-height: 1.8;

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

    padding: 35px;

    border-radius: 28px;

    text-align: center;

    color: white;

    margin-bottom: 25px;

    box-shadow: 0 14px 35px rgba(37,99,235,0.3);
}

.upload-box h2 {

    font-size: 38px;

    margin-bottom: 10px;
}

/* QUESTION BOX */

.question-box {

    background: rgba(255,255,255,0.94);

    padding: 28px;

    border-radius: 24px;

    margin-top: 25px;

    box-shadow: 0 10px 28px rgba(0,0,0,0.08);
}

/* INPUT */

.stTextInput input {

    background-color: white;

    color: #111827;

    border-radius: 18px;

    padding: 20px;

    border: 2px solid #bfdbfe;

    font-size: 17px;
}

/* BUTTON */

.stButton>button {

    width: 100%;

    padding: 18px;

    border-radius: 18px;

    background: linear-gradient(
        90deg,
        #2563eb,
        #0891b2,
        #06b6d4,
        #14b8a6
    );

    color: white;

    border: none;

    font-size: 18px;

    font-weight: 700;

    transition: 0.35s;

    box-shadow: 0 10px 22px rgba(37,99,235,0.28);
}

.stButton>button:hover {

    transform: scale(1.03);
}

/* ANSWER BOX */

.answer-box {

    background: rgba(255,255,255,0.96);

    border-left: 9px solid #22c55e;

    border-radius: 26px;

    padding: 32px;

    color: #111827;

    font-size: 17px;

    line-height: 2;

    margin-top: 18px;

    box-shadow: 0 12px 28px rgba(0,0,0,0.08);
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

/* FOOTER */

.footer {

    text-align: center;

    margin-top: 40px;

    color: #0f172a;

    font-weight: 700;

    font-size: 16px;
}

/* TITLES */

.section-title {

    color: #0f172a;

    font-size: 32px;

    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------

st.markdown("""
<div class="top-banner">
🚀 AI-Powered Medical Knowledge Assistant • Gemini + FAISS + RAG Technology
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🩺 MediSphere AI</h1>
    <p>Intelligent Medical FAQ Assistant for Modern Healthcare</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ MediSphere Control Panel")

try:

    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    st.sidebar.success("✅ Gemini API Connected")

except:

    st.sidebar.error("❌ API Key Missing")

    st.stop()

st.sidebar.markdown("""
### 📌 Supported Medical PDFs

- WHO Guidelines
- Medical FAQ Documents
- Healthcare Policies
- Treatment Information
- Hospital Manuals
- Medical Encyclopedia
""")

st.sidebar.info("""
💡 Example Questions

• What are symptoms of pneumonia?
• What causes hypertension?
• What precautions prevent dengue?
• What are asthma treatments?
""")

# ---------------- FEATURE CARDS ----------------

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="card">
        <div class="icon">📄</div>
        <h3>Medical Knowledge</h3>
        <p>
        Upload healthcare PDFs and create your intelligent AI medical knowledge system.
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
        <h3>Safe Medical Answers</h3>
        <p>
        Provides safe and professional healthcare responses from trusted uploaded documents.
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
    Upload healthcare FAQs, WHO guidelines, medical encyclopedia,
    treatment manuals, or healthcare policy documents.
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

                # ---------------- PROMPT ----------------

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
