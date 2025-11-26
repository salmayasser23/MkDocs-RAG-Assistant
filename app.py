# streamlit_app
"""
Streamlit UI for the MkDocs RAG assistant.

- Reuses ask_mkdocs() from query_cli.py (same RAG pipeline)
- Simple chat-like interface
- Shows both the answer and the context chunks used
"""

import streamlit as st
from query_cli import ask_mkdocs  # reuse your working RAG function


st.set_page_config(
    page_title="MkDocs RAG Assistant",
    layout="wide",
)

st.title("MkDocs RAG Assistant")
st.caption("RAG system over MkDocs user guides (text + images).")

# Initialise chat history
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dicts: {question, answer, context}


with st.form("question_form"):
    question = st.text_input(
        "Ask a question about MkDocs:",
        placeholder="e.g. How do I install MkDocs?",
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    try:
        answer, context_chunks = ask_mkdocs(question.strip())
        st.session_state["history"].append(
            {
                "question": question.strip(),
                "answer": answer,
                "context": context_chunks,
            }
        )
    except Exception as e:
        st.error(f"Error while answering: {e}")

# Show chat history (latest on top)
for item in reversed(st.session_state["history"]):
    st.markdown(f"**You:** {item['question']}")
    st.markdown(f"**Assistant:** {item['answer']}")

    with st.expander("Show context chunks used"):
        for i, c in enumerate(item["context"], start=1):
            st.markdown(f"**[{i}]** {c}")

    st.markdown("---")
