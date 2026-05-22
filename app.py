import streamlit as st
from core.splitter import split_into_claims
from core.nli_checker import check_claim
from core.llm_judge import judge_claim
from core.aggregator import aggregate

st.set_page_config(
    page_title="LLM Hallucination Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 LLM Hallucination Detector")
st.markdown("Paste a **source document** and an **LLM response** — ye tool check karega kitna sach hai.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Source Document")
    source_doc = st.text_area(
        "Paste karo original/trusted content yahan",
        height=250,
        placeholder="The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel..."
    )

with col2:
    st.subheader("🤖 LLM Response")
    llm_response = st.text_area(
        "Paste karo LLM ka response yahan",
        height=250,
        placeholder="The Eiffel Tower is located in Berlin. It was built in 1850 by Napoleon..."
    )

st.divider()

if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if not source_doc or not llm_response:
        st.error("Bhai dono fields bharo pehle!")
    else:
        claims = split_into_claims(llm_response)
        st.info(f"Found **{len(claims)} claims** to analyze...")

        results = []
        progress = st.progress(0)

        for i, claim in enumerate(claims):
            nli = check_claim(claim, source_doc)
            llm = judge_claim(claim, source_doc)
            agg = aggregate(nli, llm)
            results.append(agg)
            progress.progress((i + 1) / len(claims))

        progress.empty()
        st.divider()
        st.subheader("📊 Results")

        hallucinated = [r for r in results if r["verdict"] == "Hallucinated"]
        uncertain = [r for r in results if r["verdict"] == "Uncertain"]
        supported = [r for r in results if r["verdict"] == "Supported"]

        m1, m2, m3 = st.columns(3)
        m1.metric("✅ Supported", len(supported))
        m2.metric("⚠️ Uncertain", len(uncertain))
        m3.metric("🔴 Hallucinated", len(hallucinated))

        st.divider()

        for r in results:
            if r["verdict"] == "Hallucinated":
                color = "red"
                icon = "🔴"
            elif r["verdict"] == "Uncertain":
                color = "orange"
                icon = "⚠️"
            else:
                color = "green"
                icon = "✅"

            st.markdown(
                f"{icon} **{r['verdict']}** (score: `{r['hallucination_score']}`)"
                f" — :{color}[{r['claim']}]"
            )
            with st.expander("Details"):
                st.write(f"**NLI Label:** {r['nli_label']}")
                st.write(f"**LLM Verdict:** {r['llm_verdict']}")
                st.write(f"**Final Score:** {r['hallucination_score']}")