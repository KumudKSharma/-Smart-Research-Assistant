import os
import streamlit as st
import fitz  # Provided by PyMuPDF
from openai import OpenAI

# === API KEY HANDLING ===
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    st.sidebar.markdown("### 🔑 OpenAI API Key required")
    API_KEY = st.sidebar.text_input("Enter your OpenAI API key", type="password")

if not API_KEY:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
    st.stop()

# ✅ Initialize OpenAI client AFTER key is guaranteed
client = OpenAI(api_key=API_KEY)


# === FUNCTIONS ===

def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF file using PyMuPDF."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def extract_text_from_txt(uploaded_file):
    """Read text content from an uploaded TXT file."""
    return uploaded_file.read().decode("utf-8")


def summarize_text(text):
    """Return a ≤150‑word summary of the provided text."""
    prompt = (
        "Summarize the following text in under 150 words:\n\n" + text[:3000]
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def answer_question(text, question):
    """Answer a user question using only the document content with justification."""
    prompt = f"""
You are a helpful assistant. Use the document content below to answer the user's question.
Only answer using the document. Justify with paragraph or section reference.

Document:
{text[:6000]}

Question:
{question}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def generate_logic_questions(text):
    """Generate three logic‑based comprehension questions from the document."""
    prompt = (
        "Generate 3 logic-based or comprehension questions based on this document:\n\n"
        + text[:3000]
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return [q.strip("- ") for q in response.choices[0].message.content.strip().split("\n") if q]


def evaluate_answer(doc_text, question, user_answer):
    """Evaluate the user's answer and provide feedback citing the document."""
    prompt = f"""
Evaluate the following user answer based on the document.
Provide clear feedback and reference supporting parts from the document.

Document:
{doc_text[:3000]}

Question: {question}
User Answer: {user_answer}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

# === STREAMLIT UI ===

st.title("📄 Smart Research Assistant")

uploaded_file = st.file_uploader(
    "Upload PDF or TXT Document", type=["pdf", "txt"], label_visibility="visible"
)

if uploaded_file:
    # Extract text depending on file type
    if uploaded_file.type == "application/pdf":
        document_text = extract_text_from_pdf(uploaded_file)
    else:
        document_text = extract_text_from_txt(uploaded_file)

    # Auto summary section
    st.subheader("🔍 Auto Summary")
    with st.spinner("Generating summary..."):
        summary = summarize_text(document_text)
    st.info(summary)

    # Interaction mode selection
    mode = st.radio(
        "Select Interaction Mode:", ["Ask Anything", "Challenge Me"], index=0
    )

    # === Ask Anything Mode ===
    if mode == "Ask Anything":
        user_question = st.text_input("Ask a question based on the document:")
        if user_question:
            with st.spinner("Thinking..."):
                answer = answer_question(document_text, user_question)
            st.markdown(f"**Answer:** {answer}")

    # === Challenge Me Mode ===
    elif mode == "Challenge Me":
        st.subheader("🧠 Answer These Questions")
        with st.spinner("Generating questions..."):
            questions = generate_logic_questions(document_text)
        if questions:
            for i, q in enumerate(questions):
                user_answer = st.text_input(f"Q{i+1}: {q}", key=f"q{i}")
                if user_answer:
                    with st.spinner("Evaluating..."):
                        feedback = evaluate_answer(document_text, q, user_answer)
                    st.markdown(f"**Feedback:** {feedback}")
