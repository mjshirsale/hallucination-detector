import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from pipeline import run_pipeline
from core.pdf_parser import parse_pdf_bytes
from core.vector_store import VectorStore

st.set_page_config(
    page_title="HalluciDetect",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #ffffff; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

.halluci-header { text-align: center; padding: 3rem 0 1.5rem 0; }
.halluci-header h1 { font-size: 2.8rem; font-weight: 700; color: #0f0f0f; line-height: 1.2; margin-bottom: 0.5rem; }
.halluci-header p { font-size: 1.1rem; color: #6b7280; font-weight: 400; max-width: 520px; margin: 0 auto; }
.accent { color: #2563eb; }

.stats-bar { display: flex; justify-content: center; gap: 3rem; padding: 1.5rem 0; border-top: 1px solid #f3f4f6; border-bottom: 1px solid #f3f4f6; margin: 1.5rem 0 2rem 0; }
.stat-item { text-align: center; }
.stat-num { font-size: 1.6rem; font-weight: 700; color: #2563eb; }
.stat-label { font-size: 0.8rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }

.stButton > button[kind="primary"] { background: #0f0f0f !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 0.75rem 2rem !important; font-size: 1rem !important; font-weight: 600 !important; transition: all 0.2s ease !important; }
.stButton > button[kind="primary"]:hover { background: #2563eb !important; transform: translateY(-1px) !important; }

.metric-card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.25rem; text-align: center; }
.metric-card .num { font-size: 2rem; font-weight: 700; color: #0f0f0f; }
.metric-card .label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem; }
.metric-card.red .num   { color: #dc2626; }
.metric-card.amber .num { color: #d97706; }
.metric-card.green .num { color: #16a34a; }
.metric-card.blue .num  { color: #2563eb; }

.claim-card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.25rem 1.5rem; margin: 0.75rem 0; transition: box-shadow 0.2s; }
.claim-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
.claim-card.hallucinated { border-left: 4px solid #dc2626; background: #fff5f5; }
.claim-card.supported    { border-left: 4px solid #16a34a; background: #f0fdf4; }
.claim-card.uncertain    { border-left: 4px solid #d97706; background: #fffbeb; }
.claim-title { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.claim-title.red   { color: #dc2626; }
.claim-title.green { color: #16a34a; }
.claim-title.amber { color: #d97706; }
.claim-text  { font-size: 0.95rem; color: #374151; line-height: 1.6; }
.claim-score { display: inline-block; font-size: 0.75rem; background: #f3f4f6; color: #6b7280; border-radius: 20px; padding: 0.2rem 0.75rem; margin-top: 0.5rem; }

.density-cell { border-radius: 10px; padding: 1rem; text-align: center; border: 1px solid transparent; }
.density-cell.high { background: #fff5f5; border-color: #fecaca; }
.density-cell.mid  { background: #fffbeb; border-color: #fde68a; }
.density-cell.low  { background: #f0fdf4; border-color: #bbf7d0; }
.density-pct { font-size: 1.4rem; font-weight: 700; }
.density-cell.high .density-pct { color: #dc2626; }
.density-cell.mid  .density-pct { color: #d97706; }
.density-cell.low  .density-pct { color: #16a34a; }
.density-sec { font-size: 0.7rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }

.section-header { font-size: 1.1rem; font-weight: 600; color: #0f0f0f; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #f3f4f6; }

[data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #e5e7eb; }
.stTabs [aria-selected="true"] { color: #2563eb !important; }
/* ── Fix text areas ── */
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #0f0f0f !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
.stTextArea textarea::placeholder {
    color: #9ca3af !important;
}

/* ── Fix file uploader ── */
[data-testid="stFileUploader"] {
    background: #f9fafb !important;
    border: 1px dashed #d1d5db !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

/* ── Fix tabs emoji ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #f3f4f6 !important;
}            
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 HalluciDetect")
    st.markdown("<p style='color:#6b7280;font-size:0.85rem'>Production-grade hallucination detection for LLM outputs.</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**⚙️ Settings**")
    top_k = st.slider("Chunks per claim", 1, 6, 3,
                      help="Kitne relevant chunks retrieve karein per claim")
    st.divider()

    st.markdown("**📂 Document History**")
    try:
        store = VectorStore()
        stored_docs = store.list_documents()
        if stored_docs:
            st.success(f"{len(stored_docs)} document(s) cached")
            for doc_id in stored_docs:
                col1, col2 = st.columns([3, 1])
                col1.caption(f"📄 {doc_id}")
                if col2.button("🗑", key=f"del_{doc_id}"):
                    store.delete_document(doc_id)
                    st.rerun()
        else:
            st.caption("No documents cached yet.")
    except Exception:
        st.caption("Initializing vector store...")

    st.divider()
    st.markdown("**📊 Pipeline**")
    st.markdown("""
<p style='font-size:0.82rem;color:#6b7280;line-height:1.8'>
1. PDF / Text parse<br>
2. Semantic chunking<br>
3. ChromaDB + BM25 retrieval<br>
4. DeBERTa-large NLI (GPU)<br>
5. Llama 3.3 70B CoT judge<br>
6. Weighted aggregation
</p>
""", unsafe_allow_html=True)
    st.divider()
    st.caption("RTX 4060 · Groq free tier · ChromaDB")


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="halluci-header">
    <h1>Hallucination Detector<br><span class="accent">for Reliable AI.</span></h1>
    <p>Verify LLM outputs against source documents using NLI models,
       Chain-of-Thought reasoning, and hybrid semantic search.</p>
</div>

<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-num">2x</div>
        <div class="stat-label">Model Signals</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">50+</div>
        <div class="stat-label">Pages Supported</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">GPU</div>
        <div class="stat-label">Accelerated</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">Free</div>
        <div class="stat-label">Open Source</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Input ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📄  Text Input", "📁  PDF Upload"])

source_text = ""
llm_response = ""

with tab1:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**Source Document**")
        st.caption("Paste the trusted / original content")
        source_text_input = st.text_area(
            label="source_text_label",
            label_visibility="collapsed",
            height=220,
            placeholder="The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel...",
            key="source_text"
        )
    with col2:
        st.markdown("**LLM Response**")
        st.caption("Paste the AI-generated response to verify")
        llm_response_input = st.text_area(
            label="llm_response_label",
            label_visibility="collapsed",
            height=220,
            placeholder="The Eiffel Tower is located in Berlin. It was built in 1850 by Napoleon...",
            key="llm_response"
        )
    source_text  = source_text_input
    llm_response = llm_response_input

with tab2:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**Source Document PDF**")
        source_pdf = st.file_uploader("Upload source", type=["pdf"], key="source_pdf")
        if source_pdf:
            source_text = parse_pdf_bytes(source_pdf.read())
            st.success(f"✅ {len(source_text):,} characters extracted")
            with st.expander("Preview"):
                st.write(source_text[:400] + "...")
    with col2:
        st.markdown("**LLM Response PDF**")
        response_pdf = st.file_uploader("Upload response", type=["pdf"], key="response_pdf")
        if response_pdf:
            llm_response = parse_pdf_bytes(response_pdf.read())
            st.success(f"✅ {len(llm_response):,} characters extracted")
            with st.expander("Preview"):
                st.write(llm_response[:400] + "...")

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
analyze_btn = st.button("Detect →", type="primary", use_container_width=True)


# ── Results ──────────────────────────────────────────────────────────────────
if analyze_btn:
    if not source_text or not llm_response:
        st.error("Please fill both fields before analyzing.")
    else:
        status_box  = st.empty()
        progress_bar = st.progress(0)

        def progress_callback(msg: str, pct: float):
            status_box.markdown(
                f"<p style='color:#6b7280;font-size:0.9rem'>⏳ {msg}</p>",
                unsafe_allow_html=True
            )
            progress_bar.progress(pct)

        summary = run_pipeline(
            source_text, llm_response,
            top_k=top_k,
            progress_callback=progress_callback
        )
        progress_bar.empty()
        status_box.empty()

        # ── Summary metrics ───────────────────────────────────────────────
        st.markdown("<div class='section-header'>Analysis Summary</div>",
                    unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        cards = [
            (c1, summary["total_claims"],       "Total Claims",    "blue"),
            (c2, summary["hallucinated"],        "Hallucinated",    "red"),
            (c3, summary["uncertain"],           "Uncertain",       "amber"),
            (c4, summary["supported"],           "Supported",       "green"),
            (c5, f"{summary['hallucination_rate']}%", "Hal. Rate", "red"),
        ]
        for col, num, label, color in cards:
            col.markdown(
                f"<div class='metric-card {color}'>"
                f"<div class='num'>{num}</div>"
                f"<div class='label'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # ── Density map ───────────────────────────────────────────────────
        st.markdown("<div class='section-header'>Hallucination Density Map</div>",
                    unsafe_allow_html=True)
        st.caption("Section-wise hallucination distribution across the document")

        dcols = st.columns(len(summary["density_map"]))
        for col, sec in zip(dcols, summary["density_map"]):
            d   = sec["density"]
            css = "high" if d >= 60 else "mid" if d >= 30 else "low"
            icon = "🔴" if d >= 60 else "⚠️" if d >= 30 else "✅"
            col.markdown(
                f"<div class='density-cell {css}'>"
                f"<div class='density-pct'>{d}%</div>"
                f"<div class='density-sec'>{icon} {sec['section']}</div>"
                f"<div class='density-sec'>{sec['claims']} claims</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # ── API efficiency ────────────────────────────────────────────────
        st.markdown("<div class='section-header'>API Efficiency</div>",
                    unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        e1.metric("Groq Calls Made",  summary["llm_calls_made"])
        e2.metric("Calls Saved",      summary["llm_calls_saved"])
        e3.metric("Doc ID",           summary["doc_id"])

        # ── Claim breakdown ───────────────────────────────────────────────
        st.markdown("<div class='section-header'>Claim-by-Claim Breakdown</div>",
                    unsafe_allow_html=True)

        for i, r in enumerate(summary["results"]):
            if r["verdict"] == "Hallucinated":
                css, title_css, icon, label = "hallucinated", "red", "🔴", "HALLUCINATED"
            elif r["verdict"] == "Uncertain":
                css, title_css, icon, label = "uncertain", "amber", "⚠️", "UNCERTAIN"
            else:
                css, title_css, icon, label = "supported", "green", "✅", "SUPPORTED"

            st.markdown(
                f"<div class='claim-card {css}'>"
                f"<div class='claim-title {title_css}'>{icon} {label}</div>"
                f"<div class='claim-text'>{r['claim']}</div>"
                f"<span class='claim-score'>score: {r['hallucination_score']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            with st.expander(f"View details — Claim {i+1}"):
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**AI Explanation**")
                    st.write(r["explanation"] or
                             "High confidence verdict — explanation not required.")
                    if r["correction"] and r["correction"] != "N/A":
                        st.markdown("**Suggested Correction**")
                        st.info(r["correction"])
                with d2:
                    st.markdown("**Evidence from Source**")
                    st.write(r["evidence"] or "N/A")
                    st.markdown("**Model Details**")
                    st.write(f"LLM invoked: "
                             f"{'Yes' if r['llm_used'] else 'No — DeBERTa confident'}")
                    st.write(f"Confidence: {r['confidence']}")

        # ── Self correction ───────────────────────────────────────────────
        hallucinated_list = [r for r in summary["results"]
                             if r["verdict"] == "Hallucinated"]
        if hallucinated_list:
            st.markdown("<div class='section-header'>Self-Correction Mode</div>",
                        unsafe_allow_html=True)
            st.caption("Auto-correct hallucinated claims based on source document")
            if st.button("✨ Auto-Correct", type="secondary"):
                corrected = llm_response
                for r in hallucinated_list:
                    if r["correction"] and r["correction"] != "N/A":
                        corrected = corrected.replace(
                            r["claim"],
                            f"**[CORRECTED: {r['correction']}]**"
                        )
                st.markdown("**Corrected Response:**")
                st.markdown(corrected)

        # ── Export ────────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>Export Report</div>",
                    unsafe_allow_html=True)

        lines = [
            "HALLUCIDETECT — ANALYSIS REPORT",
            "=" * 50,
            f"Doc ID          : {summary['doc_id']}",
            f"Total Claims    : {summary['total_claims']}",
            f"Hallucinated    : {summary['hallucinated']}",
            f"Uncertain       : {summary['uncertain']}",
            f"Supported       : {summary['supported']}",
            f"Hallucination % : {summary['hallucination_rate']}%",
            f"Groq API Calls  : {summary['llm_calls_made']}",
            f"API Calls Saved : {summary['llm_calls_saved']}",
            "=" * 50,
            "\nCLAIM-BY-CLAIM BREAKDOWN\n"
        ]
        for i, r in enumerate(summary["results"]):
            lines += [
                f"Claim {i+1}: {r['verdict']} (score: {r['hallucination_score']})",
                f"  Text       : {r['claim']}",
                f"  Explanation: {r['explanation']}",
                f"  Evidence   : {r['evidence']}",
                f"  Correction : {r['correction']}",
                ""
            ]

        st.download_button(
            label="📥 Download Report (.txt)",
            data="\n".join(lines),
            file_name="hallucidetect_report.txt",
            mime="text/plain",
            use_container_width=True
        )