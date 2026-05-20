import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# ── Model loaders (called from app.py with st.cache_resource) ───────────────
def get_models(st):
    @st.cache_resource
    def _load_embedder():
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    @st.cache_resource
    def _load_llm():
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=300,
        )

    return _load_embedder(), _load_llm()


# ── Semantic similarity score ────────────────────────────────────────────────
def compute_match_score(resume_text, jd_text, embedder):
    """
    Encode both texts and return cosine similarity as a 0–100 score.
    """
    vecs = embedder.encode([resume_text, jd_text], show_progress_bar=False)
    score = cosine_similarity([vecs[0]], [vecs[1]])[0][0]
    # Scale: cosine similarity for text is typically 0.3–0.95
    # Map to a friendlier 0–100 range
    normalized = max(0.0, min(1.0, (score - 0.2) / 0.7))
    return round(float(normalized) * 100, 1), round(float(score) * 100, 1)


# ── Keyword extraction helper ────────────────────────────────────────────────
def extract_keywords(text):
    """
    Simple keyword extractor — pulls capitalized terms, tech words, and
    common skill patterns. Not NLP-perfect but works well for resumes/JDs.
    """
    # Common tech / skill tokens
    text_lower = text.lower()
    tech_terms = [
        "python", "sql", "java", "scala", "r", "spark", "pyspark", "kafka",
        "airflow", "dbt", "databricks", "snowflake", "redshift", "bigquery",
        "aws", "azure", "gcp", "terraform", "docker", "kubernetes", "git",
        "mlflow", "sagemaker", "bedrock", "langchain", "openai", "huggingface",
        "rag", "llm", "nlp", "machine learning", "deep learning", "etl", "elt",
        "data warehouse", "data lake", "lakehouse", "power bi", "tableau",
        "informatica", "talend", "alteryx", "streamlit", "fastapi", "flask",
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "faiss",
        "pinecone", "chroma", "vector database", "embeddings", "transformers",
        "medallion", "delta lake", "ci/cd", "devops", "agile", "scrum",
        "restful", "api", "microservices", "hadoop", "hive", "presto",
        "looker", "dax", "azure data factory", "adf", "adls",
    ]
    found = set()
    for term in tech_terms:
        if term in text_lower:
            found.add(term)
    return found


# ── Gap analysis via LLM ─────────────────────────────────────────────────────
def analyze_gaps(resume_text, jd_text, llm):
    """
    Use flan-t5 to identify skill gaps and generate suggestions.
    Returns a dict with keys: gaps, strengths, suggestions.
    """
    # Truncate inputs to fit model context
    resume_short = " ".join(resume_text.split()[:400])
    jd_short = " ".join(jd_text.split()[:400])

    # --- Gaps prompt ---
    gaps_prompt = (
        f"Job description requires these skills and qualifications:\n{jd_short}\n\n"
        f"The candidate's resume shows:\n{resume_short}\n\n"
        f"List only the skills and qualifications required by the job that are MISSING from the resume. "
        f"Be specific. List as comma separated values."
    )
    gaps_result = llm(gaps_prompt)[0]["generated_text"].strip()

    # --- Strengths prompt ---
    strengths_prompt = (
        f"Job description requires:\n{jd_short}\n\n"
        f"Resume shows:\n{resume_short}\n\n"
        f"List only the skills and qualifications that are present in BOTH the job description and the resume. "
        f"Be specific. List as comma separated values."
    )
    strengths_result = llm(strengths_prompt)[0]["generated_text"].strip()

    # --- Suggestions prompt ---
    suggestions_prompt = (
        f"A candidate is applying for this role:\n{jd_short}\n\n"
        f"Their resume currently says:\n{resume_short}\n\n"
        f"Give 3 specific, actionable suggestions to improve the resume for this job. "
        f"Number each suggestion 1. 2. 3."
    )
    suggestions_result = llm(suggestions_prompt)[0]["generated_text"].strip()

    return {
        "gaps": gaps_result,
        "strengths": strengths_result,
        "suggestions": suggestions_result,
    }


# ── Keyword-level gap (fast, no LLM) ────────────────────────────────────────
def keyword_gap(resume_text, jd_text):
    resume_kw = extract_keywords(resume_text)
    jd_kw = extract_keywords(jd_text)
    missing = sorted(jd_kw - resume_kw)
    matched = sorted(jd_kw & resume_kw)
    return matched, missing
