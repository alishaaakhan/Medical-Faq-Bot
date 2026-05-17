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
    page_title="MediAura AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Urbanist:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"]  {
    font-family: 'Urbanist', sans-serif;
}

/* MAIN APP */

.stApp {

    background:
    linear-gradient(
        rgba(240,248,255,0.78),
        rgba(240,248,255,0.84)
    ),

    url("https://images.unsplash.com/photo-1579684385127-1ef15d508118?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
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

/* TOP NAV */

.topbar {

    width: 100%;

    padding: 16px;

    border-radius: 22px;

    background: linear-gradient(
        90deg,
        rgba(37,99,235,0.92),
        rgba(6,182,212,0.92),
        rgba(20,184,166,0.92)
    );

    color: white;

    text-align: center;

    font-size: 17px;

    font-weight: 700;

    box-shadow: 0 10px 30px rgba(37,99,235,0.30);

    margin-bottom: 20px;
}

/* HERO */

.hero {

    text-align: center;

    padding-top: 10px;
    padding-bottom: 25px;
}

.hero h1 {

    font-size: 96px;

    font-weight: 900;

    line-height: 1;

    margin-bottom: 10px;

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

.hero p {

    font-size: 24px;

    color: #0f172a;

    font-weight: 600;
}

/* GLASS CONTAINER */

.glass {

    background: rgba(255,255,255,0.68);

    backdrop-filter: blur(24px);

    border-radius: 36px;

    padding: 40px;

    border: 1px solid rgba(255,255,255,0.30);

    box-shadow: 0 14px 45px rgba(0,0,0,0.15);
}

/* FEATURE CARDS */

.card {

    background: rgba(255,255,255,0.94);

    border-radius: 30px;

    padding: 34px;

    text-align: center;

    height: 300px;

    transition: 0.4s;

    box-shadow: 0 14px 35px rgba(0,0,0,0.08);
}

.card:hover {

    transform: translateY(-14px) scale(1.03);
}

.icon {

    font-size: 76px;

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

.upload-box p {

    font-size: 17px;
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

</style>
""", unsafe_allow_html=True)

# ---------------- TOPBAR ----------------

st.markdown("""
<div class="topbar">
🚀 Next Generation Medical FAQ Assistant • Gemini AI • FAISS • RAG Architecture
</div>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------

st.markdown("""
<div class="hero">
    <h1>🩺 MediAura AI</h1>
    <p>Smart Healthcare Intelligence Powered by Artificial Intelligence</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ MediAura Dashboard")

try:

    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    st.sidebar.success("✅ Gemini API Connected")

except:

    st.sidebar.error("❌ API Key Missing")

    st.stop()

st.sidebar.markdown("""
### 📌 Supported Documents

- WHO Guidelines
- Healthcare Manuals
- Medical FAQ PDFs
- Treatment Information
- Medical Encyclopedia
- Hospital Policies
""")

st.sidebar.info("""
💡 Example Questions

• What are symptoms of diabetes?
• What causes asthma?
• How to prevent dengue?
• What are COVID precautions?
""")

# ---------------- FEATURE CARDS ----------------

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="card">
        <div class="icon">📄</div>
        <h3>Medical Knowledge</h3>
        <p>
        Upload trusted medical PDFs and build your intelligent healthcare assistant.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="card">
        <div class="icon">🧠</div>
        <h3>AI Semantic Search</h3>
        <p>
        Advanced AI retrieval powered by Gemini AI and FAISS vector search.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="card">
        <div class="icon">⚕️</div>
        <h3>Reliable Answers</h3>
        <p>
        Provides professional healthcare answers strictly from uploaded documents.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- MAIN CONTAINER ----------------

st.markdown('<div class="glass">', unsafe_allow_html=True)

# ---------------- UPLOAD SECTION ----------------

st.markdown("""
<div class="upload-box">
    <h2>📤 Upload Medical PDF</h2>
    <p>
    Upload WHO guidelines, treatment documents,
    healthcare manuals or medical encyclopedia PDFs.
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
1. Answer ONLY from provided context.
2. If answer is unavailable say:
"The information is not available in uploaded medical documents."
3. Never generate fake medical advice.
4. Keep responses professional and safe.
5. Add a short healthcare disclaimer.

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

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
Made with ❤️ using Streamlit • Gemini AI • FAISS • RAG Architecture
</div>
""", unsafe_allow_html=True)
