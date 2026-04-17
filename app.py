import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
import PyPDF2
import pandas as pd



nltk.download('stopwords')
stop_words = set(stopwords.words("english"))



from dataset.dataset_loader import load_dataset
from preprocessing.text_cleaner import clean_text
import numpy as np
import pandas as pd



st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

STRONG_THRESHOLD = 0.70  
WEAK_LOW = 0.60
WEAK_HIGH = 0.70

df = pd.read_csv("dataset/cleaned_train.csv")

q1_emb = np.load("dataset/q1_emb.npy")
q2_emb = np.load("dataset/q2_emb.npy")


# STEP 3: APPLY CLEANING
df = load_dataset("dataset/reference/train.csv")
df.to_csv("dataset/cleaned_train.csv", index=False)


df['q1_clean'] = df['question1'].apply(clean_text)
df['q2_clean'] = df['question2'].apply(clean_text)
df = df[:300] 


# def check_single_doc(doc, model, df, q1_emb, q2_emb):
#     sentences = sent_tokenize(doc)

#     total_score = 0
#     matches = []

#     for sent in sentences:
#         emb = model.encode([sent])

#         sim1 = cosine_similarity(emb, q1_emb)[0]
#         sim2 = cosine_similarity(emb, q2_emb)[0]

#         max_score = max(max(sim1), max(sim2))
#         total_score += max_score

#         if max_score > 0.75:
#             matches.append((sent, max_score))

#     if len(sentences) == 0:
#         return 0, []

#     final_score = total_score / len(sentences)
#     # percent = final_score * 100
#     plag_count = 0

#     for sent in sentences:
#         emb = model.encode([sent])

#         sim1 = cosine_similarity(emb, q1_emb)[0]
#         sim2 = cosine_similarity(emb, q2_emb)[0]

#         max_score = max(max(sim1), max(sim2))

#         if max_score > 0.60:
#             plag_count += 1
#             matches.append((sent, max_score))

#     if len(sentences) == 0:
#         return 0, []

#     percent = (plag_count / len(sentences)) * 100

#     return percent, matches
def check_single_doc(doc, model, df, q1_emb, q2_emb):
    sentences = sent_tokenize(doc)

    matches = []
    plag_count = 0

    for sent in sentences:
        emb = model.encode([sent])

        sim1 = cosine_similarity(emb, q1_emb)[0]
        sim2 = cosine_similarity(emb, q2_emb)[0]

        max_score = max(max(sim1), max(sim2))

        # ❌ Ignore weak similarity
        if WEAK_LOW < max_score < WEAK_HIGH:
            continue

        # ✅ Only strong plagiarism
        if max_score >= STRONG_THRESHOLD:
            plag_count += 1
            matches.append((sent, max_score))

    if len(sentences) == 0:
        return 0, []

    percent = (plag_count / len(sentences)) * 100

    return percent, matches
#--------------
def get_plagiarized_sentences(doc, model, q1_emb, q2_emb, threshold=0.60):
    sentences = sent_tokenize(doc)

    plagiarized = []

    for sent in sentences:
        emb = model.encode([sent])

        sim1 = cosine_similarity(emb, q1_emb)[0]
        sim2 = cosine_similarity(emb, q2_emb)[0]

        max_score = max(max(sim1), max(sim2))

        if max_score > threshold:
            plagiarized.append((sent, max_score))

    return plagiarized

#---------------
def highlight_text(doc, plag_sentences):
    for sent, _ in plag_sentences:
        doc = doc.replace(
            sent,
            f"<span style='background-color:yellow'>{sent}</span>"
        )
    return doc
# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    
    return text

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---------------- SIMILARITY ----------------
def compare_documents(doc1, doc2):
    doc1 = clean_text(doc1)
    doc2 = clean_text(doc2)

    emb1 = model.encode([doc1])
    emb2 = model.encode([doc2])

    score = cosine_similarity(emb1, emb2)[0][0]
    return score


nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def highlight_similar_sentences(doc1, doc2, threshold=0.7):
    sentences1 = sent_tokenize(doc1)
    sentences2 = sent_tokenize(doc2)

    results = []

    for s1 in sentences1:
        for s2 in sentences2:
            score = compare_documents(s1, s2)
            if score > threshold:
                results.append((s1, s2, score))

    return results




def plagiarism_percentage(doc1, doc2):
    sentences1 = sent_tokenize(doc1)
    matches = highlight_similar_sentences(doc1, doc2)

    if len(sentences1) == 0:
        return 0

    percent = (len(matches) / len(sentences1)) * 100
    return percent

def compare_multiple(doc, reference_docs):
    results = []

    for ref in reference_docs:
        score = compare_documents(doc, ref)
        results.append(score)

    return results

from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import tempfile

# def speech_to_text():
#     audio = mic_recorder(
#         start_prompt="🎤 Start Recording",
#         stop_prompt="⏹ Stop",
#         just_once=True,
#         use_container_width=True,
#         key="mic1"   # ✅ VERY IMPORTANT (unique key)
#     )

#     if audio and "bytes" in audio:
#         recognizer = sr.Recognizer()

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#             f.write(audio["bytes"])
#             temp_path = f.name

#         with sr.AudioFile(temp_path) as source:
#             audio_data = recognizer.record(source)

#         try:
#             text = recognizer.recognize_google(audio_data)
#             return text
#         except sr.UnknownValueError:
#             return "❌ Could not understand"
#         except sr.RequestError:
#             return "❌ API error"

#     return ""


def generate_pdf(score, percent):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Plagiarism Report", ln=True)

    pdf.cell(200, 10, txt=f"Similarity Score: {round(score,2)}", ln=True)
    pdf.cell(200, 10, txt=f"Plagiarism %: {round(percent,2)}%", ln=True)

    file_path = "report.pdf"
    pdf.output(file_path)

    return file_path

# # ---------------- UI ----------------
# st.set_page_config(page_title="Plagiarism Checker", layout="centered")

# st.title("📄 NLP Plagiarism Detection System")

# st.markdown("### Select Mode")

# # SESSION STATE for mode
# if "mode" not in st.session_state:
#     st.session_state.mode = None

# col1, col2 = st.columns(2)

# with col1:
#     if st.button("📑 Compare Two PDFs"):
#         st.session_state.mode = "compare"

# with col2:
#     if st.button("📄 Single PDF Check"):
#         st.session_state.mode = "single"


# # ---------------- MODE 1 ----------------
# if st.session_state.mode == "compare":

#     st.markdown("## 🔍 Compare Two PDFs")

#     file1 = st.file_uploader("Upload First PDF", type=["pdf"], key="file1")
#     file2 = st.file_uploader("Upload Second PDF", type=["pdf"], key="file2")

#     if st.button("Check Plagiarism", key="compare_btn"):

#         if file1 and file2:

#             doc1 = extract_text_from_pdf(file1)
#             doc2 = extract_text_from_pdf(file2)

#             score = compare_documents(doc1, doc2)

#             st.success(f"Similarity Score: {score:.2f}")

#             percent = plagiarism_percentage(doc1, doc2)
#             st.info(f"Plagiarism %: {percent:.2f}")

#             matches = highlight_similar_sentences(doc1, doc2)

#             st.subheader("📌 Top Matches")

#             for s1, s2, sc in matches[:5]:
#                 st.write(f"🔸 {s1}")
#                 st.write(f"Score: {sc:.2f}")
#                 st.markdown("---")

#         else:
#             st.warning("Upload both PDFs")


# # ---------------- MODE 2 ----------------
# elif st.session_state.mode == "single":

#     st.markdown("## 📄 Single PDF Plagiarism Check")

#     file = st.file_uploader("Upload PDF", type=["pdf"], key="single_file")

#     if st.button("Check Plagiarism", key="single_btn"):

#         if file:

#             doc = extract_text_from_pdf(file)

#             plag_sentences = get_plagiarized_sentences(
#                 doc, model, q1_emb, q2_emb
#             )

#             percent, matches = check_single_doc(
#                 doc, model, df, q1_emb, q2_emb
#             )

#             highlighted_doc = highlight_text(doc, plag_sentences)

#             st.subheader("📄 Highlighted Document")
#             st.markdown(highlighted_doc, unsafe_allow_html=True)

#             st.success(f"Plagiarism: {percent:.2f}%")

#             plag_sentences = get_plagiarized_sentences(
#                 doc, model, q1_emb, q2_emb
#             )

#             st.subheader("🚨 Plagiarized Sentences")

#             if percent == 0:
#                 st.success("✅ No plagiarism detected (content is original)")
#             elif percent < 20:
#                 st.info("ℹ Low plagiarism detected")
#             elif percent < 50:
#                 st.warning("⚠ Moderate plagiarism detected")
#             else:
#                 st.error("🚨 High plagiarism detected")

#             for sent, score in plag_sentences:
#                 st.markdown(
#                     f"<span style='color:red'><b>⚠ {sent}</b></span>",
#                     unsafe_allow_html=True
#                 )
#                 st.write(f"Similarity: {score:.2f}")
#                 st.markdown("---")

#         else:
#             st.warning("Upload a PDF")

import streamlit as st
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Plagiarism Detector",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Title Animation */
.title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    animation: fadeIn 2s ease-in-out;
}

/* Buttons */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 55px;
    font-size: 18px;
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #24c6dc, #514a9d);
}

/* Cards */
.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.1);
    box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

/* Animations */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>AI Plagiarism Detection System</div>", unsafe_allow_html=True)

st.markdown("### 🚀 Powered by BERT & Semantic Search")

# ---------------- MODE SELECT ----------------
st.markdown("## ⚡ Choose Mode")

if "mode" not in st.session_state:
    st.session_state.mode = None

col1, col2 = st.columns(2)

with col1:
    if st.button("📑 Compare Two PDFs"):
        st.session_state.mode = "compare"

with col2:
    if st.button("📄 Single PDF Check"):
        st.session_state.mode = "single"

# ---------------- LOADER FUNCTION ----------------
def show_loader(text="Processing..."):
    with st.spinner(text):
        time.sleep(1)

# ---------------- MODE 1 ----------------
if st.session_state.mode == "compare":

    st.markdown("## 🔍 Compare Two Documents")

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        file1 = st.file_uploader("Upload PDF", type=["pdf"], key="pdf1")
        file2 = st.file_uploader("Upload PDF", type=["pdf"], key="pdf2")

    

        # ✅ BUTTON OUTSIDE COLUMNS (IMPORTANT)
    if st.button("🚀 Run Plagiarism Check"):

            # ✅ CHECK INPUT (PDF OR TEXT)
            if (file1 or text1) and (file2 or text2):

                show_loader("Analyzing documents...")

                # ✅ DOC1
                if file1:
                    doc1 = extract_text_from_pdf(file1)
                else:
                    doc1 = text1

                # ✅ DOC2
                if file2:
                    doc2 = extract_text_from_pdf(file2)
                else:
                    doc2 = text2

                # ✅ PROCESS
                score = compare_documents(doc1, doc2)
                percent = plagiarism_percentage(doc1, doc2)

                # Animated Result
                st.markdown("## 📊 Results")

                st.progress(int(score * 100))
                st.success(f"Similarity Score: {score:.2f}")
                st.info(f"Plagiarism: {percent:.2f}%")

                if percent > 60:
                    st.error("🚨 High Plagiarism Detected")
                elif percent > 30:
                    st.warning("⚠ Moderate Similarity")
                else:
                    st.success("✅ Content is mostly original")

                matches = highlight_similar_sentences(doc1, doc2)

                st.markdown("## 📌 Matched Sentences")

                for s1, s2, sc in matches[:5]:
                    st.markdown(f"""
                    <div class="card">
                        <b>Score:</b> {sc:.2f}<br>
                        <b>Doc1:</b> {s1}<br>
                        <b>Doc2:</b> {s2}
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.warning("⚠ Upload both PDFs")

            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- MODE 2 ----------------
elif st.session_state.mode == "single":

    st.markdown("## 📄 Single Document Detection")

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        file = st.file_uploader("Upload PDF", type=["pdf"])


        # st.markdown("### 🎤 OR Speak Your Content")

        # speech_input = speech_to_text()

        # text_input = st.text_area(
        #     "Your Text",
        #     value=speech_input if speech_input else ""
        # )

        if st.button("🚀 Detect Plagiarism"):

            if file:
                show_loader("Scanning document...")

                if file:
                    doc = extract_text_from_pdf(file)
                else:
                    doc = text_input

                plag_sentences = get_plagiarized_sentences(
                    doc, model, q1_emb, q2_emb
                )

                percent, matches = check_single_doc(
                    doc, model, df, q1_emb, q2_emb
                )

                st.markdown("## 📊 Result")

                st.progress(int(percent))
                st.success(f"Plagiarism: {percent:.2f}%")

                # Status
                if percent == 0:
                    st.success("✅ Fully Original")
                elif percent < 20:
                    st.info("ℹ Low plagiarism")
                elif percent < 50:
                    st.warning("⚠ Moderate plagiarism")
                else:
                    st.error("🚨 High plagiarism")

                # Highlighted text
                highlighted_doc = highlight_text(doc, plag_sentences)

                st.markdown("## 📄 Highlighted Content")
                st.markdown(highlighted_doc, unsafe_allow_html=True)

                st.markdown("## 🚨 Detected Sentences")

                for sent, score in plag_sentences:
                    st.markdown(f"""
                    <div class="card">
                        <span style="color:red"><b>⚠ {sent}</b></span><br>
                        Similarity: {score:.2f}
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.warning("⚠ Upload a PDF")

        st.markdown("</div>", unsafe_allow_html=True)