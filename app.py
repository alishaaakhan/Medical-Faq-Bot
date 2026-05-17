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

# ---------------- ULTRA PREMIUM UI ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* MAIN BACKGROUND */

.stApp {

    background-image:
    linear-gradient(
        rgba(240,248,255,0.82),
        rgba(240,248,255,0.85)
    ),
    url("https://images.unsplash.com/photo-1631815588090-d4bfec5b1ccb?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
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
    padding-top: 10px;
    padding-bottom: 30px;
}

.hero h1 {

    font-size: 82px;

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

    margin-bottom: 10px;
}

.hero p {

    font-size: 22px;

    color: #0f172a;

    font-weight: 500;
}

/* GLASS CONTAINER */

.glass {

    background: rgba(255,255,255,0.68);

    backdrop-filter: blur(20px);

    border-radius: 30px;

    padding: 35px;

    box-shadow: 0 10px 40px rgba(0,0,0,0.15);

    border: 1px solid rgba(255,255,255,0.35);
}

/* FEATURE CARDS */

.card {

    background: rgba(255,255,255,0.92);

    border-radius: 24px;

    padding: 30px;

    text-align: center;

    box-shadow: 0 12px 25px rgba(0,0,0,0.08);

    transition: 0.4s;

    height: 260px;
}

.card:hover {

    transform: translateY(-10px) scale(1.02);
}

.icon {

    font-size: 58px;

    margin-bottom: 12px;
}

.card h3 {

    font-size: 26px;

    color: #0f172a;

    margin-bottom: 10px;
}

.card p {

    color: #475569;

    font-size: 15px;

    line-height: 1.7;
}

/* IMAGE SECTION */

[data-testid="stImage"] {

    display: flex;

    justify-content: center;

    align-items: center;
}

[data-testid="stImage"] img {

    width: 100%;

    height: 260px;

    object-fit: cover;

    border-radius: 24px;

    box-shadow:
    0 10px 30px rgba(0,0,0,0.12);

    transition: 0.4s;
}

[data-testid="stImage"] img:hover {

    transform: scale(1.03);
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

    border-radius: 24px;

    padding: 30px;

    text-align: center;

    color: white;

    margin-bottom: 25px;

    box-shadow: 0 10px 25px rgba(37,99,235,0.28);
}

.upload-box h2 {

    font-size: 34px;
}

/* QUESTION BOX */

.question-box {

    background: rgba(255,255,255,0.92);

    padding: 25px;

    border-radius: 22px;

    margin-top: 20px;

    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}

/* INPUT */

.stTextInput input {

    background-color: white;

    color: #111827;

    border-radius: 16px;

    padding: 18px;

    border: 2px solid #dbeafe;

    font-size: 17px;
}

/* BUTTON */

.stButton>button {

    width: 100%;

    padding: 16px;

    border-radius: 16px;

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

    font-weight: 600;

    transition: 0.35s;
}

.stButton>button:hover {

    transform: scale(1.02);
}

/* ANSWER BOX */

.answer-box {

    background: rgba(255,255,255,0.95);

    border-left: 8px solid #22c55e;

    border-radius: 22px;

    padding: 30px;

    color: #111827;

    font-size: 17px;

    line-height: 2;

    margin-top: 18px;

    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
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

/* FLOATING BOT */

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
        #06b6d4,
        #14b8a6
    );

    display: flex;

    justify-content: center;

    align-items: center;

    font-size: 45px;

    box-shadow:
    0 0 35px rgba(37,99,235,0.6);

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

/* FOOTER */

.footer {

    text-align: center;

    margin-top: 40px;

    color: #0f172a;

    font-weight: 600;

    font-size: 16px;
}

.section-title {

    color: #0f172a;

    font-size: 30px;

    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------

st.markdown("""
<div class="hero">
    <h1>🩺 MediVerse AI</h1>
    <p>Next Generation Medical FAQ Assistant</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MEDICAL IMAGE SECTION ----------------

st.markdown("""
<style>

/* IMAGE ALIGNMENT */

[data-testid="stImage"] {

    display: flex;

    justify-content: center;

    align-items: center;
}

[data-testid="stImage"] img {

    width: 100%;

    height: 260px;

    object-fit: cover;

    border-radius: 24px;

    box-shadow:
    0 10px 30px rgba(0,0,0,0.12);

    transition: 0.4s;
}

[data-testid="stImage"] img:hover {

    transform: scale(1.03);
}

/* EXTRA SECTION */

.extra-section {

    margin-top: 60px;

    background: rgba(255,255,255,0.75);

    backdrop-filter: blur(18px);

    border-radius: 28px;

    padding: 40px;

    box-shadow: 0 10px 30px rgba(0,0,0,0.10);
}

.extra-title {

    text-align: center;

    font-size: 42px;

    font-weight: 800;

    color: #0f172a;

    margin-bottom: 15px;
}

.extra-text {

    text-align: center;

    font-size: 18px;

    color: #334155;

    line-height: 1.9;

    margin-bottom: 35px;
}

.feature-mini {

    background: rgba(255,255,255,0.92);

    border-radius: 22px;

    padding: 25px;

    text-align: center;

    box-shadow: 0 8px 22px rgba(0,0,0,0.08);

    transition: 0.3s;
}

.feature-mini:hover {

    transform: translateY(-8px);
}

.feature-mini h3 {

    color: #0f172a;

    margin-top: 12px;

    margin-bottom: 10px;
}

.feature-mini p {

    color: #475569;

    font-size: 15px;

    line-height: 1.7;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown("""
<div style="
text-align:center;
margin-top:20px;
margin-bottom:30px;
">

<h2 style="
font-size:42px;
font-weight:800;
color:#0f172a;
margin-bottom:10px;
">
🏥 Smart Healthcare AI Platform
</h2>

<p style="
font-size:18px;
color:#334155;
margin-bottom:35px;
">
AI-powered medical FAQ assistant using RAG, Gemini AI and trusted healthcare documents.
</p>

</div>
""", unsafe_allow_html=True)

# ---------------- IMAGES ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.image(
        "https://images.unsplash.com/photo-1584982751601-97dcc096659c?q=80&w=1200&auto=format&fit=crop",
        use_container_width=True
    )

with col2:
    st.image(
        "https://images.unsplash.com/photo-1576091160550-2173dba999ef?q=80&w=1200&auto=format&fit=crop",
        use_container_width=True
    )

with col3:
    st.image(
        "https://images.unsplash.com/photo-1581595219315-a187dd40c322?q=80&w=1200&auto=format&fit=crop",
        use_container_width=True
    )

# ---------------- GAP ----------------

st.markdown("<br><br><br>", unsafe_allow_html=True)

# ---------------- EXTRA SECTION ----------------

st.markdown("""
<div class="extra-section">

    <div class="extra-title">
        ⚕️ Why Choose MediVerse AI?
    </div>

    <div class="extra-text">
        MediVerse AI combines Retrieval-Augmented Generation (RAG),
        Gemini AI, FAISS semantic search, and trusted healthcare
        documents to deliver reliable medical FAQ responses with
        intelligent document understanding.
    </div>

</div>
""", unsafe_allow_html=True)

mini1, mini2, mini3 = st.columns(3)

with mini1:
    st.markdown("""
    <div class="feature-mini">
        <div style="font-size:52px;">🧠</div>
        <h3>AI Intelligence</h3>
        <p>
        Advanced semantic understanding using Gemini AI and FAISS vector search.
        </p>
    </div>
    """, unsafe_allow_html=True)

with mini2:
    st.markdown("""
    <div class="feature-mini">
        <div style="font-size:52px;">📚</div>
        <h3>Trusted Sources</h3>
        <p>
        Answers generated strictly from uploaded healthcare documents.
        </p>
    </div>
    """, unsafe_allow_html=True)

with mini3:
    st.markdown("""
    <div class="feature-mini">
        <div style="font-size:52px;">🛡️</div>
        <h3>Safe Responses</h3>
        <p>
        Designed to provide professional and responsible medical information.
        </p>
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

# ---------------- FEATURE CARDS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="icon">📄</div>
        <h3>Medical Knowledge Base</h3>
        <p>
        Upload trusted healthcare PDFs and create intelligent AI medical assistance.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="icon">🧠</div>
        <h3>Semantic AI Search</h3>
        <p>
        Powered by Gemini AI and FAISS for accurate medical retrieval.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="icon">⚕️</div>
        <h3>Safe Healthcare Answers</h3>
        <p>
        Professional answers strictly from uploaded documents.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="glass">', unsafe_allow_html=True)

# ---------------- UPLOAD BOX ----------------

st.markdown("""
<div class="upload-box">
    <h2>📤 Upload Medical PDF</h2>
    <p>
    Upload healthcare FAQs, WHO guidelines or treatment documents.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type="pdf")

# ---------------- PDF PROCESS ----------------

if uploaded_file:

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

        def chunk_text(text, chunk_size=700, overlap=120):

            chunks = []

            start = 0

            while start < len(text):

                end = start + chunk_size

                chunks.append(text[start:end])

                start += chunk_size - overlap

            return chunks

        texts = chunk_text(full_text)

        with st.spinner("🧠 Building AI medical knowledge base..."):

            model = SentenceTransformer("all-MiniLM-L6-v2")

            embeddings = model.encode(texts).astype("float32")

            index = faiss.IndexFlatL2(embeddings.shape[1])

            index.add(embeddings)

        st.success(
            f"✅ Medical knowledge base ready with {len(texts)} intelligent chunks."
        )

        st.markdown("""
        <div class="question-box">
            <h2 class="section-title">💬 Ask Your Medical Question</h2>
        </div>
        """, unsafe_allow_html=True)

        question = st.text_input(
            "",
            placeholder="Example: What are the symptoms of pneumonia?"
        )

        if st.button("🚀 Generate AI Medical Answer"):

            if question.strip() == "":

                st.warning("Please enter a medical question.")

            else:

                q_emb = model.encode([question]).astype("float32")

                distances, indices = index.search(q_emb, k=4)

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

                with st.expander("📚 View Source Context"):

                    st.write(context[:2000])

                st.warning("""
⚠️ Medical Disclaimer:
This chatbot provides information only from uploaded documents.
It does not replace professional medical consultation.
Always consult certified healthcare professionals.
""")

    finally:

        os.unlink(tmp_path)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FLOATING BOT ----------------

st.markdown("""
<div class='avatar'>
🤖
</div>
""", unsafe_allow_html=True)

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
Made by Alisha Khan ❤️
</div>
""", unsafe_allow_html=True)
