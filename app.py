import streamlit as st
from src.retrieve import retrieve
from src.generate import generate_answer
from src.guard import check_grounding
from src.eval import evaluate
import time

st.set_page_config(page_title="Talk to My Notes", page_icon="🔠", layout="wide")

# ---- Custom styling ----
st.markdown("""
<style>
.main-header {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    margin-bottom: 0 !important;
    line-height: 1.1 !important;
    color : #3d1f26;
}
.sub-header {
    color: #6b7280;
    font-size: 1rem;
    margin-top: 0;
    margin-bottom: 1.5rem;
}
.answer-card {
    background-color: ##f6dce1;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-top: 0.6rem;
    margin-bottom: 0.8rem;
}
.badge-grounded {
    display: inline-block;
    background-color: #dcfce7;
    color: #15803d;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}
.badge-flagged {
    display: inline-block;
    background-color: #fef3c7;
    color: #b45309;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}
.chunk-box {
    background-color: #ffffff;
    border: 1px solid #eef0f3;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}
.chunk-score {
    color: #7c3aed;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header"> Welcome to my notes world</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Here what you see - RAG over my own college notes — with retrieval evaluation (Hit Rate, MRR) '
    'and a hallucination guard that checks every answer is actually grounded.</p>',
    unsafe_allow_html=True,
)

# session_state persists across Streamlit's reruns (it reruns the whole script
# on every interaction) -- without it, history would reset every time.
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("▥  Retrieval evaluation")
    st.caption("Runs the labeled eval set through the retriever right now.")
    if st.button("▶ Run eval harness", use_container_width=True):
        with st.spinner("Evaluating retrieval..."):
            report = evaluate(k=5)
        col1, col2 = st.columns(2)
        col1.metric("Hit Rate@5", report["hit_rate"])
        col2.metric("MRR", report["mrr"])
        st.caption(f"Across {report['num_questions']} labeled questions")

    st.divider()
    st.header("⏲︎  Chat history")
    if not st.session_state.history:
        st.caption("Questions you have searched so far...")
    else:
        for idx, item in enumerate(st.session_state.history):
            label = item["question"]
            if len(label) > 40:
                 label = label[:40] + "..."

            is_selected = (idx == st.session_state.selected_idx)
            button_type = "primary" if is_selected else "secondary"

            if st.button(f" {label}", key=f"hist_{idx}", use_container_width=True, type=button_type):
                st.session_state.selected_idx = idx
                
        st.divider()
        if st.button("🗑 Clear chat history", use_container_width=True):
            st.session_state.history = []
            st.session_state.selected_idx = None
            st.rerun()

    st.divider()
    
    st.header("ℹ️ About this app")
    st.caption(
        "Your questions meet my notes here. Every document is chunked , embedded, and indexed "
        "behind the scenes to make searching feel like chatting."
        
        "\n\n This isn't a multi-user upload tool — it answers from my own notes."
    )

# ---------------- MAIN AREA ----------------
with st.form(key="ask_form", clear_on_submit=True):
    question = st.text_input(
        "Ask something:",
        placeholder="e.g. What is encapsulation in OOPs?",
    )
    submitted = st.form_submit_button("Ask")

if submitted and question:
    start_time = time.time()

    with st.spinner("Retrieving relevant chunks..."):
        chunks = retrieve(question, k=5)

    with st.spinner("Generating answer..."):
        answer = generate_answer(question, chunks)

    with st.spinner("Checking for hallucinations..."):
        verdict = check_grounding(answer, chunks)

    elapsed = time.time() - start_time

    st.session_state.history.insert(0, {
        "question": question,
        "answer": answer,
        "verdict": verdict,
        "chunks": chunks,
        "elapsed": elapsed,
    })
    st.session_state.selected_idx = 0  # show the just-asked question

display_item = None
if st.session_state.selected_idx is not None and st.session_state.history:
    display_item = st.session_state.history[st.session_state.selected_idx]

if display_item:
    st.markdown(f"**⌕ You asked: {display_item['question']}**")

    badge = (
        '<span class="badge-grounded"> Grounded</span>'
        if display_item["verdict"]["grounded"]
        else '<span class="badge-flagged">⚠️ Possibly unsupported</span>'
    )

    st.markdown(
        f'<div class="answer-card">{badge}<br><br>{display_item["answer"]}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"⏱ Answered in {display_item['elapsed']:.1f}s")

    if not display_item["verdict"]["grounded"] and display_item["verdict"].get("unsupported_claims"):
        with st.expander("Unsupported claims flagged by the guard"):
            for claim in display_item["verdict"]["unsupported_claims"]:
                st.write(f"- {claim}")

    with st.expander("Retrieved chunks (debug view)"):
        for c in display_item["chunks"]:
            st.markdown(
                f'<div class="chunk-box"><span class="chunk-score">[{c["score"]:.3f}]</span> '
                f'{c["id"]}<br><span style="color:#6b7280">{c["text"][:250]}...</span></div>',
                unsafe_allow_html=True,
            )
else:
    st.info("Ask a question to get started.")
