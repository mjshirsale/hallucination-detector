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
.verdict-hallucinated {
    background: #FAECE7;
    border-left: 4px solid #D85A30;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
}
.verdict-supported {
    background: #EAF3DE;
    border-left: 4px solid #639922;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
}
.verdict-uncertain {
    background: #FAEEDA;
    border-left: 4px solid #BA7517;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
}
.density-high { background: #FAECE7; border-radius: 8px; padding: 8px; text-align: center; }
.density-mid  { background: #FAEEDA; border-radius: 8px; padding: 8px; text-align: center; }
.density-low  { background: #EAF3DE; border-radius: 8px; padding: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 HalluciDetect")
    st.markdown("Production-grade hallucination detection")
    st.divider()

    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks per claim", 1, 6, 3,
                      help="Kitne relevant chunks retrieve karein")

    st.divider()
    st.subheader("📂 Document History")
    try:
        store = VectorStore()
        stored_docs = store.list_documents()
        if stored_docs:
            st.success(f"{len(stored_docs)} document(s) stored")
            for doc_id in stored_docs:
                col1, col2 = st.columns([3, 1])
                col1.caption(f"📄 {doc_id}")
                if col2.button("🗑", key=f"del_{doc_id}"):
                    store.delete_document(doc_id)
                    st.rerun()
        else:
            st.caption("No documents stored yet")
    except Exception as e:
        st.caption("Vector store initializing...")

    st.divider()
    st.subheader("📊 How it works")
    st.markdown("""
    1. **PDF/Text** parse hota hai
    2. **Smart chunking** — semantic + overlap
    3. **ChromaDB** — persistent vector store
    4. **Hybrid search** — ChromaDB + BM25
    5. **DeBERTa-large** — NLI on RTX 4060
    6. **Llama 3.3 70B** — CoT reasoning
    7. **Smart aggregation** — final verdict
    """)
    st.divider()
    st.caption("Built with RTX 4060 + Groq free tier")


# ── Main UI ───────────────────────────────────────────────
st.title("🔍 HalluciDetect")
st.markdown("*Detect hallucinations in LLM outputs — NLI + Chain-of-Thought + Hybrid Search*")
st.divider()

tab1, tab2 = st.tabs(["📄 Text Input", "📁 PDF Upload"])

source_text = ""
llm_response = ""

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Source Document")
        source_text_input = st.text_area(
            "Paste trusted content",
            height=250,
            placeholder="The Eiffel Tower is located in Paris...",
            key="source_text"
        )
    with col2:
        st.subheader("🤖 LLM Response")
        llm_response_input = st.text_area(
            "Paste LLM response to verify",
            height=250,
            placeholder="The Eiffel Tower is located in Berlin...",
            key="llm_response"
        )
    source_text = source_text_input
    llm_response = llm_response_input

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 Source PDF")
        source_pdf = st.file_uploader("Upload source document",
                                       type=["pdf"], key="source_pdf")
        if source_pdf:
            source_text = parse_pdf_bytes(source_pdf.read())
            st.success(f"✅ {len(source_text)} characters extracted")
            st.text_area("Preview", source_text[:300] + "...", height=100)

    with col2:
        st.subheader("🤖 LLM Response PDF")
        response_pdf = st.file_uploader("Upload LLM response",
                                         type=["pdf"], key="response_pdf")
        if response_pdf:
            llm_response = parse_pdf_bytes(response_pdf.read())
            st.success(f"✅ {len(llm_response)} characters extracted")
            st.text_area("Preview", llm_response[:300] + "...", height=100)

st.divider()
analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)


# ── Analysis ─────────────────────────────────────────────
if analyze_btn:
    if not source_text or not llm_response:
        st.error("❌ Dono fields bharo pehle!")
    else:
        status_text = st.empty()
        progress_bar = st.progress(0)

        def progress_callback(msg: str, pct: float):
            status_text.markdown(f"⏳ **{msg}**")
            progress_bar.progress(pct)

        summary = run_pipeline(
            source_text,
            llm_response,
            top_k=top_k,
            progress_callback=progress_callback
        )

        progress_bar.empty()
        status_text.empty()
        st.success("✅ Analysis complete!")
        st.divider()

        # ── Summary metrics ───────────────────────────────
        st.subheader("📊 Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Claims",     summary["total_claims"])
        m2.metric("🔴 Hallucinated",  summary["hallucinated"])
        m3.metric("⚠️ Uncertain",     summary["uncertain"])
        m4.metric("✅ Supported",     summary["supported"])
        m5.metric("Hallucination %",  f"{summary['hallucination_rate']}%")

        st.divider()

        # ── Density map ───────────────────────────────────
        st.subheader("🗺️ Hallucination Density Map")
        st.caption("Document ke har section mein kitni hallucinations hain")
        density_cols = st.columns(len(summary["density_map"]))
        for col, section in zip(density_cols, summary["density_map"]):
            d = section["density"]
            css = "density-high" if d >= 60 else "density-mid" if d >= 30 else "density-low"
            icon = "🔴" if d >= 60 else "⚠️" if d >= 30 else "✅"
            col.markdown(
                f"<div class='{css}'>"
                f"<strong>{section['section']}</strong><br>"
                f"{icon} {d}%<br>"
                f"<small>{section['claims']} claims</small>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.divider()

        # ── API efficiency ────────────────────────────────
        st.subheader("⚡ API Efficiency")
        e1, e2, e3 = st.columns(3)
        e1.metric("Groq API Calls",   summary["llm_calls_made"])
        e2.metric("API Calls Saved",  summary["llm_calls_saved"])
        e3.metric("Doc ID",           summary["doc_id"])

        st.divider()

        # ── Claim breakdown ───────────────────────────────
        st.subheader("📋 Claim-by-Claim Breakdown")
        for i, r in enumerate(summary["results"]):
            if r["verdict"] == "Hallucinated":
                css, icon = "verdict-hallucinated", "🔴"
            elif r["verdict"] == "Uncertain":
                css, icon = "verdict-uncertain", "⚠️"
            else:
                css, icon = "verdict-supported", "✅"

            st.markdown(
                f"<div class='{css}'>"
                f"<strong>{icon} Claim {i+1}: {r['verdict']}</strong> "
                f"(score: {r['hallucination_score']})<br>"
                f"<em>{r['claim']}</em>"
                f"</div>",
                unsafe_allow_html=True
            )

            with st.expander(f"Details — Claim {i+1}"):
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**🤖 AI Explanation**")
                    st.write(r["explanation"] or
                             "High confidence — no explanation needed.")
                    if r["correction"] and r["correction"] != "N/A":
                        st.markdown("**✏️ Suggested Correction**")
                        st.info(r["correction"])
                with d2:
                    st.markdown("**📄 Evidence from Source**")
                    st.write(r["evidence"] or "N/A")
                    st.markdown("**🔧 Models Used**")
                    st.write(f"LLM called: "
                             f"{'Yes' if r['llm_used'] else 'No (DeBERTa confident)'}")
                    st.write(f"Confidence: {r['confidence']}")

        st.divider()

        # ── Self correction ───────────────────────────────
        hallucinated_claims = [r for r in summary["results"]
                               if r["verdict"] == "Hallucinated"]
        if hallucinated_claims:
            st.subheader("🔧 Self-Correction Mode")
            if st.button("✨ Auto-Correct Hallucinated Claims", type="secondary"):
                corrected = llm_response
                for r in hallucinated_claims:
                    if r["correction"] and r["correction"] != "N/A":
                        corrected = corrected.replace(
                            r["claim"],
                            f"**[CORRECTED: {r['correction']}]**"
                        )
                st.markdown("**📝 Corrected Response:**")
                st.markdown(corrected)

        st.divider()

        # ── Export ────────────────────────────────────────
        st.subheader("📥 Export Report")
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