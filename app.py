# app.py
import io, re, string, json
from typing import List, Tuple
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------- Simple, portable extractors (no big models) -------
def read_file(upload) -> str:
    name = (upload.name or "").lower()
    data = upload.read()
    if name.endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
            text = ""
            reader = PdfReader(io.BytesIO(data))
            for p in reader.pages:
                text += (p.extract_text() or "") + "\n"
            return text
        except Exception:
            pass
    if name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            pass
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ------- Lightweight cleaning -------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text

# ------- Skills catalog -------
SKILLS = {
    "programming": [
        "python","java","javascript","typescript","c","c++","c#","go","rust","scala","kotlin","bash",
    ],
    "data & ml": [
        "pandas","numpy","scikit-learn","tensorflow","pytorch","matplotlib","seaborn","xgboost",
        "lightgbm","nlp","computer vision","opencv","mlops","airflow","spark","hadoop","sql",
    ],
    "web & backend": [
        "flask","django","fastapi","node.js","express","spring","rest api","graphql","docker","kubernetes",
        "aws","azure","gcp","redis","postgresql","mysql","mongodb","ci/cd","git","linux",
    ],
    "testing & qa": [
        "pytest","unittest","selenium","playwright","cypress","jira",
    ],
    "soft": [
        "communication","leadership","problem solving","teamwork","agile","scrum",
    ],
}
ALL_SKILLS = sorted({s for group in SKILLS.values() for s in group})

def find_skills(text: str) -> List[str]:
    t = " " + normalize(text) + " "
    hits = []
    for sk in ALL_SKILLS:
        if f" {sk} " in t or re.search(rf"\b{re.escape(sk)}\b", t):
            hits.append(sk)
    return sorted(set(hits))

# ------- Similarity -------
def compute_match(resume_text: str, jd_text: str) -> Tuple[float, List[Tuple[str,float]]]:
    docs = [normalize(resume_text), normalize(jd_text)]
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english")
    X = vec.fit_transform(docs)
    score = float(cosine_similarity(X[0], X[1])[0,0])
    feats = vec.get_feature_names_out()
    w_res = X[0].toarray()[0]
    w_jd  = X[1].toarray()[0]
    contrib = (w_res * w_jd)
    top_idx = contrib.argsort()[-15:][::-1]
    top_terms = [(feats[i], float(contrib[i])) for i in top_idx if contrib[i] > 0]
    return score, top_terms

# ------- Suggestions -------
def suggest_bullets(missing_sk: List[str], jd_text: str) -> List[str]:
    suggestions = []
    if missing_sk:
        for sk in missing_sk[:6]:
            suggestions.append(f"Implemented {sk} in a production project; improved reliability/performance by X%.")
    if "api" in jd_text.lower():
        suggestions.append("Designed and documented REST APIs; reduced response time by 35% using caching and async IO.")
    if any(w in jd_text.lower() for w in ["ml","machine learning","model"]):
        suggestions.append("Built ML pipeline; achieved +X% accuracy and automated retraining.")
    suggestions.append("Owned end-to-end features in an Agile team; wrote tests and CI/CD to ship confidently.")
    return suggestions[:8]

# ================== UI ==================
st.set_page_config(page_title="AI Resume Analyzer & Job Matcher", page_icon="🧠", layout="wide")
st.title("🧠 AI Resume Analyzer & Job Matcher")
st.caption("Upload your resume, paste a job description, and get a match score, skill gaps, and improvement tips.")

col1, col2 = st.columns([1,1])
with col1:
    resume_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
    resume_text = ""
    if resume_file:
        resume_text = read_file(resume_file)

with col2:
    jd_text = st.text_area("Paste Job Description", height=240, placeholder="Paste the JD here...")

run = st.button("Analyze Match")

if run:
    if not resume_text.strip():
        st.error("Please upload a readable resume file.")
    elif not jd_text.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Analyzing…"):
            score, top_terms = compute_match(resume_text, jd_text)
            r_sk = set(find_skills(resume_text))
            j_sk = set(find_skills(jd_text))
            matching_sk = sorted(r_sk & j_sk)
            missing_sk = sorted(j_sk - r_sk)

        # Score card
        st.subheader("📊 Match Score")
        st.metric("Cosine Similarity", f"{score*100:.1f}%")

        # Skills
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Matching Skills")
            st.write(", ".join(matching_sk) if matching_sk else "_None detected_")
        with c2:
            st.subheader("❗ Missing Skills (from JD)")
            st.write(", ".join(missing_sk) if missing_sk else "_None_")

        # 🔎 Key overlap terms
        st.subheader("🔎 Key Overlap Terms")
        if top_terms:
            st.dataframe([{"term": t, "weight": f"{w:.4f}"} for t, w in top_terms], hide_index=True)
        else:
            st.write("_No significant n-gram overlap_")

        # 🎯 Improved Skill Heatmap 
        st.subheader("🎯 Skill Coverage Heatmap")
        if j_sk:
            skills_list = sorted(j_sk)
            data = [[1 if sk in r_sk else 0 for sk in skills_list]]

            coverage_pct = int((sum(data[0])/len(skills_list))*100)
            st.caption(f"Resume covers {coverage_pct}% of the JD skills")

            fig, ax = plt.subplots(figsize=(max(len(skills_list)*0.7, 8), 3))
            sns.heatmap(
                data,
                annot=False,
                cmap="RdYlGn",
                cbar=False,
                xticklabels=skills_list,
                yticklabels=["Resume"],
                linewidths=0.8,
                linecolor='gray'
            )

            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
            st.pyplot(fig)
        else:
            st.write("_No skills found in JD to visualize._")

        # Suggestions
        st.subheader("💡 Suggested Resume Bullets")
        for s in suggest_bullets(missing_sk, jd_text):
            st.markdown(f"- {s}")

        # Export JSON summary
        summary = {
            "score": round(score, 4),
            "matching_skills": matching_sk,
            "missing_skills": missing_sk,
            "top_terms": top_terms,
        }
        st.download_button(
            "⬇️ Download Analysis (JSON)",
            data=json.dumps(summary, indent=2),
            file_name="resume_job_match_summary.json",
            mime="application/json",
        )

# Sidebar Tips
st.sidebar.header("ℹ️ Tips")
st.sidebar.write(
    "- Tune skills in code (`SKILLS`) for your domain.\n"
    "- Add weights per skill or required/optional tags.\n"
    "- Swap TF-IDF for transformer embeddings later."
)
