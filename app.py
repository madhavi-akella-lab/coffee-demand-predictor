import streamlit as st
from matcher import get_models, compute_match_score, analyze_gaps, keyword_gap

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Job Matcher",
    page_icon="🎯",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .score-box {
        text-align: center;
        padding: 28px 20px;
        border-radius: 16px;
        margin-bottom: 20px;
    }
    .score-high { background: #e6f4ea; border: 2px solid #34a853; }
    .score-mid  { background: #fff8e1; border: 2px solid #fbbc04; }
    .score-low  { background: #fce8e6; border: 2px solid #ea4335; }
    .score-number { font-size: 56px; font-weight: 800; line-height: 1; }
    .score-high .score-number { color: #1a7340; }
    .score-mid  .score-number { color: #7b4f00; }
    .score-low  .score-number { color: #c0392b; }
    .score-label { font-size: 15px; color: #555; margin-top: 6px; }
    .chip-green {
        display: inline-block;
        background: #e6f4ea; color: #1a7340;
        border: 1px solid #a8d5b5;
        border-radius: 20px; padding: 3px 12px;
        font-size: 13px; font-weight: 500; margin: 3px;
    }
    .chip-red {
        display: inline-block;
        background: #fce8e6; color: #c0392b;
        border: 1px solid #f5a9a9;
        border-radius: 20px; padding: 3px 12px;
        font-size: 13px; font-weight: 500; margin: 3px;
    }
    .suggestion-box {
        background: #f0f7ff;
        border-left: 4px solid #2d7dd2;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 14px;
        line-height: 1.7;
        color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 AI Resume Job Matcher")
st.markdown("**Paste your resume and a job description. Get an instant match score, skill gap analysis, and improvement suggestions.**")
st.markdown("*100% free — no API key needed. Powered by Sentence Transformers + Flan-T5.*")
st.divider()

# ── Load models ───────────────────────────────────────────────────────────────
with st.spinner("Loading AI models (first load takes ~30 seconds)..."):
    embedder, llm = get_models(st)
st.success("✅ Models ready!")

# ── Input columns ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Your Resume")
    resume_text = st.text_area(
        "Paste your resume text here",
        height=320,
        placeholder="Copy and paste your full resume text here...",
        label_visibility="collapsed",
    )

with col2:
    st.subheader("💼 Job Description")
    jd_text = st.text_area(
        "Paste the job description here",
        height=320,
        placeholder="Copy and paste the job description you're applying for...",
        label_visibility="collapsed",
    )

st.markdown("")
analyze_btn = st.button("🔍 Analyze Match", type="primary", use_container_width=True)

# ── Analysis ───────────────────────────────────────────────────────────────────
if analyze_btn:
    if not resume_text.strip():
        st.error("Please paste your resume text.")
    elif not jd_text.strip():
        st.error("Please paste the job description.")
    elif len(resume_text.strip()) < 100:
        st.error("Resume text is too short. Please paste the full resume.")
    elif len(jd_text.strip()) < 50:
        st.error("Job description is too short. Please paste the full job description.")
    else:
        st.divider()
        st.subheader("📊 Results")

        with st.spinner("Running analysis..."):
            # 1 — Semantic match score
            display_score, raw_score = compute_match_score(resume_text, jd_text, embedder)

            # 2 — Keyword gap (fast)
            matched_kw, missing_kw = keyword_gap(resume_text, jd_text)

            # 3 — LLM analysis
            analysis = analyze_gaps(resume_text, jd_text, llm)

        # ── Score display ──────────────────────────────────────────────────
        score_class = "score-high" if display_score >= 65 else ("score-mid" if display_score >= 40 else "score-low")
        score_emoji = "🟢" if display_score >= 65 else ("🟡" if display_score >= 40 else "🔴")
        score_verdict = (
            "Strong match — your profile aligns well with this role."
            if display_score >= 65
            else "Moderate match — some gaps to address before applying."
            if display_score >= 40
            else "Low match — significant gaps identified. Review suggestions below."
        )

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.markdown(f"""
            <div class="score-box {score_class}">
                <div class="score-number">{display_score}%</div>
                <div class="score-label">Overall Match Score</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="score-box {'score-high' if len(matched_kw) > len(missing_kw) else 'score-mid'}">
                <div class="score-number">{len(matched_kw)}/{len(matched_kw)+len(missing_kw)}</div>
                <div class="score-label">Keywords Matched</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"### {score_emoji} {score_verdict}")
            st.markdown(f"*Semantic similarity score: {raw_score:.1f}%*")

        st.divider()

        # ── Keyword chips ──────────────────────────────────────────────────
        kw_col1, kw_col2 = st.columns(2)

        with kw_col1:
            st.markdown("#### ✅ Matched Keywords")
            if matched_kw:
                chips = "".join([f'<span class="chip-green">{kw}</span>' for kw in matched_kw])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.markdown("*No common tech keywords detected.*")

        with kw_col2:
            st.markdown("#### ❌ Missing Keywords")
            if missing_kw:
                chips = "".join([f'<span class="chip-red">{kw}</span>' for kw in missing_kw])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.markdown("*No missing tech keywords detected — great!*")

        st.divider()

        # ── LLM Analysis ──────────────────────────────────────────────────
        st.markdown("#### 🧠 AI-Powered Analysis")

        tab1, tab2, tab3 = st.tabs(["💡 Improvement Suggestions", "❌ Skill Gaps", "✅ Your Strengths"])

        with tab1:
            suggestions_text = analysis.get("suggestions", "")
            if suggestions_text:
                # Try to split numbered suggestions
                lines = [l.strip() for l in suggestions_text.split("\n") if l.strip()]
                for line in lines:
                    st.markdown(f'<div class="suggestion-box">{line}</div>', unsafe_allow_html=True)
            else:
                st.info("No specific suggestions generated. Try with a longer job description.")

        with tab2:
            gaps_text = analysis.get("gaps", "No gaps identified.")
            st.markdown(f'<div class="suggestion-box">{gaps_text}</div>', unsafe_allow_html=True)

        with tab3:
            strengths_text = analysis.get("strengths", "No strengths identified.")
            st.markdown(f'<div class="suggestion-box">{strengths_text}</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown(
            "Built by [Madhavi Akella](https://linkedin.com/in/madhavi-akella-Lab-2b8213114) · "
            "[GitHub](https://github.com/madhavi-akella-Lab) · "
            "Powered by 🤗 Hugging Face"
        )
