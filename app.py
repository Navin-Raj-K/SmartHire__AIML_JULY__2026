import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
from pathlib import Path
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SMART HIRE - AI-POWERED JOB RECOMMENDATION SYSTEM
# ============================================================

st.set_page_config(
    page_title="SmartHire",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 20px;
        color: #9aa0a6;
        margin-bottom: 28px;
    }

    .section-note {
        color: #9aa0a6;
        font-size: 14px;
    }

    .match-excellent {
        color: #21c77a;
        font-weight: 700;
    }

    .match-good {
        color: #f5c542;
        font-weight: 700;
    }

    .match-moderate {
        color: #ff9f43;
        font-weight: 700;
    }

    .match-low {
        color: #ff5c5c;
        font-weight: 700;
    }

    .skill-chip {
        display: inline-block;
        padding: 5px 10px;
        margin: 3px 4px 3px 0;
        border-radius: 14px;
        background: rgba(33, 150, 243, 0.14);
        border: 1px solid rgba(33, 150, 243, 0.35);
        font-size: 13px;
    }

    .score-breakdown {
        padding: 14px 16px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin: 10px 0;
    }

    .score-note {
        color: #9aa0a6;
        font-size: 13px;
    }

    .reason-box {
        padding: 12px 14px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.035);
        border-left: 3px solid rgba(33, 150, 243, 0.8);
        margin-top: 8px;
    }

    div[data-testid="stMetric"] {
        padding: 8px 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models"

JOBS_FILE = MODEL_PATH / "jobs_clean.pkl"
TFIDF_MATRIX_FILE = MODEL_PATH / "tfidf_matrix.pkl"
TFIDF_VECTORIZER_FILE = MODEL_PATH / "tfidf_vectorizer.pkl"


# ============================================================
# LOAD SAVED ML FILES
# ============================================================

@st.cache_resource
def load_models():
    jobs = pd.read_pickle(JOBS_FILE)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_FILE)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_FILE)

    return jobs.copy(), tfidf_vectorizer, tfidf_matrix


try:
    jobs, tfidf_vectorizer, tfidf_matrix = load_models()

except Exception as e:
    st.error("❌ Unable to load SmartHire model files.")
    st.info(
        "Make sure these files exist inside the project's models folder: "
        "jobs_clean.pkl, tfidf_matrix.pkl and tfidf_vectorizer.pkl."
    )
    st.code(str(e))
    st.stop()


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "title",
    "company",
    "location",
    "experience",
    "salary_min",
    "salary_max",
    "pay_period",
    "url",
]

missing_columns = [c for c in required_columns if c not in jobs.columns]

if missing_columns:
    st.error(f"❌ Missing required job columns: {missing_columns}")
    st.stop()

if len(jobs) != tfidf_matrix.shape[0]:
    st.error(
        "❌ The number of jobs does not match the number of rows "
        "in the saved TF-IDF matrix."
    )
    st.write(f"Jobs: {len(jobs):,}")
    st.write(f"TF-IDF rows: {tfidf_matrix.shape[0]:,}")
    st.stop()


# ============================================================
# SALARY NORMALIZATION
# ============================================================

def convert_salary_to_yearly(salary, pay_period):
    """Convert salary into an estimated yearly value."""

    if pd.isna(salary) or pd.isna(pay_period):
        return np.nan

    try:
        salary = float(salary)
    except (ValueError, TypeError):
        return np.nan

    period = str(pay_period).strip().upper()

    multipliers = {
        "YEARLY": 1,
        "ANNUAL": 1,
        "MONTHLY": 12,
        "WEEKLY": 52,
        "BIWEEKLY": 26,
        "HOURLY": 2080,
    }

    multiplier = multipliers.get(period)

    if multiplier is None:
        return np.nan

    return salary * multiplier


jobs["salary_min_yearly"] = [
    convert_salary_to_yearly(s, p)
    for s, p in zip(jobs["salary_min"], jobs["pay_period"])
]

jobs["salary_max_yearly"] = [
    convert_salary_to_yearly(s, p)
    for s, p in zip(jobs["salary_max"], jobs["pay_period"])
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_text(value, default="Not specified"):
    if pd.isna(value):
        return default

    text = str(value).strip()
    return text if text else default


def format_salary(row):
    salary_min = row.get("salary_min")
    salary_max = row.get("salary_max")
    pay_period = row.get("pay_period")

    period = (
        str(pay_period).title()
        if pd.notna(pay_period) and str(pay_period).strip()
        else ""
    )

    if pd.notna(salary_min) and pd.notna(salary_max):
        text = f"${float(salary_min):,.0f} – ${float(salary_max):,.0f}"
        return f"{text} ({period})" if period else text

    if pd.notna(salary_min):
        text = f"From ${float(salary_min):,.0f}"
        return f"{text} ({period})" if period else text

    if pd.notna(salary_max):
        text = f"Up to ${float(salary_max):,.0f}"
        return f"{text} ({period})" if period else text

    return "Salary not specified"


def get_score_label(score):
    if score >= 70:
        return "🟢 Excellent Match", "match-excellent"
    if score >= 50:
        return "🟡 Good Match", "match-good"
    if score >= 30:
        return "🟠 Moderate Match", "match-moderate"
    return "🔴 Low Match", "match-low"


def get_resume_keywords(resume_text, top_n=12):
    try:
        vector = tfidf_vectorizer.transform([resume_text])
        scores = vector.toarray()[0]
        feature_names = tfidf_vectorizer.get_feature_names_out()

        indices = np.argsort(scores)[::-1]
        terms = []

        for idx in indices:
            if scores[idx] <= 0:
                break

            term = str(feature_names[idx]).strip()

            if term and term not in terms:
                terms.append(term)

            if len(terms) >= top_n:
                break

        return terms

    except Exception:
        return []


def split_skill_text(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    parts = re.split(r"[,;|\\n/•]+", text)

    cleaned = []
    seen = set()

    for part in parts:
        skill = re.sub(r"\s+", " ", part).strip(" .:-")
        key = skill.lower()

        if len(skill) >= 2 and key not in seen:
            seen.add(key)
            cleaned.append(skill)

    return cleaned


def normalize_skill(skill):
    """Normalize a skill phrase for comparison."""
    skill = re.sub(r"[^a-z0-9+#.\- ]+", " ", str(skill).lower())
    skill = re.sub(r"\s+", " ", skill).strip()
    return skill


def get_job_skills(row):
    """Return unique normalized job skills."""
    if "skills" not in row.index:
        return []

    skills = split_skill_text(row.get("skills"))
    result = []
    seen = set()

    for skill in skills:
        normalized = normalize_skill(skill)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(skill)

    return result


def skill_match_score(resume_text, row):
    """
    Calculate a skill-match percentage from the job's explicit skills.
    Falls back to important resume terms when the job has no skills field.
    """
    resume_lower = normalize_skill(resume_text)
    job_skills = get_job_skills(row)

    if job_skills:
        matched = 0
        for skill in job_skills:
            normalized = normalize_skill(skill)
            if normalized and normalized in resume_lower:
                matched += 1
        return min(100.0, (matched / len(job_skills)) * 100.0), [
            skill for skill in job_skills
            if normalize_skill(skill) in resume_lower
        ]

    # Fallback: compare important resume TF-IDF terms with job text.
    resume_terms = get_resume_keywords(resume_text, top_n=20)
    job_text = " ".join(
        str(row.get(column))
        for column in ["title", "description", "combined_text"]
        if column in row.index and pd.notna(row.get(column))
    ).lower()

    if not resume_terms or not job_text.strip():
        return 0.0, []

    matched_terms = [term for term in resume_terms if term.lower() in job_text]
    return min(100.0, (len(matched_terms) / min(len(resume_terms), 10)) * 100.0), matched_terms[:8]


def get_matching_skills(resume_text, row, top_n=8):
    """Return skills/terms used in the explanation UI."""
    _, matches = skill_match_score(resume_text, row)
    return matches[:top_n]


def extract_experience_years(resume_text):
    """Infer years of professional experience from common resume wording."""
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional\s+)?experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+in\s+(?:software|development|engineering|data|technology|it)",
    ]

    candidates = []
    lower = resume_text.lower()

    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            try:
                candidates.append(float(match.group(1)))
            except (ValueError, TypeError):
                pass

    return max(candidates) if candidates else np.nan


def experience_match_score(resume_text, job_experience):
    """
    Compare inferred resume years with a job's experience level.
    Missing/unclear experience returns a neutral score.
    """
    if pd.isna(job_experience):
        return 50.0

    resume_years = extract_experience_years(resume_text)

    if pd.isna(resume_years):
        return 50.0

    level = str(job_experience).lower()

    if "intern" in level:
        target_min, target_max = 0.0, 1.0
    elif "entry" in level or "associate" in level or "junior" in level:
        target_min, target_max = 0.0, 2.0
    elif "mid" in level:
        target_min, target_max = 2.0, 5.0
    elif "senior" in level:
        target_min, target_max = 5.0, 10.0
    elif "director" in level or "executive" in level or "lead" in level:
        target_min, target_max = 7.0, 20.0
    else:
        return 50.0

    if target_min <= resume_years <= target_max:
        return 100.0

    if resume_years < target_min:
        gap = target_min - resume_years
    else:
        gap = resume_years - target_max

    return max(0.0, 100.0 - gap * 20.0)


def location_match_score(resume_text, job_location):
    """
    Compare the resume's location wording with the job location.
    This is a soft signal, not a hard requirement.
    """
    if pd.isna(job_location):
        return 50.0

    job_loc = normalize_skill(job_location)
    if not job_loc:
        return 50.0

    resume_lower = normalize_skill(resume_text)

    # Exact full-location phrase.
    if job_loc in resume_lower:
        return 100.0

    # Compare meaningful location tokens.
    tokens = [
        token for token in job_loc.split()
        if len(token) >= 3
        and token not in {"united", "states", "area", "metropolitan"}
    ]

    if not tokens:
        return 50.0

    matches = sum(token in resume_lower for token in tokens)
    if matches == len(tokens):
        return 100.0
    if matches > 0:
        return 60.0

    return 0.0


def extract_salary_expectation(resume_text):
    """
    Infer an annual salary expectation only when the resume explicitly
    contains salary/compensation wording. Otherwise return NaN.
    """
    lower = resume_text.lower()

    salary_pattern = re.compile(
        r"(?:salary|compensation|expected salary|desired salary|"
        r"salary expectation)[^$0-9]{0,40}"
        r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
        r"(k|thousand|lakh|lakhs)?",
        re.IGNORECASE,
    )

    match = salary_pattern.search(lower)
    if not match:
        return np.nan

    try:
        value = float(match.group(1).replace(",", ""))
    except (ValueError, TypeError):
        return np.nan

    suffix = (match.group(2) or "").lower()

    if suffix == "k":
        value *= 1000
    elif suffix in {"thousand"}:
        value *= 1000
    elif suffix in {"lakh", "lakhs"}:
        value *= 100000

    # Avoid treating implausibly small values as annual expectations.
    if value < 1000:
        return np.nan

    return value


def salary_match_score(resume_text, row):
    """
    Score salary compatibility only if the resume states an expectation.
    If no expectation is found, return a neutral score and mark it as
    unavailable so the final weighted score can ignore this component.
    """
    expectation = extract_salary_expectation(resume_text)

    if pd.isna(expectation):
        return 50.0, False

    salary_min = row.get("salary_min_yearly")
    salary_max = row.get("salary_max_yearly")

    if pd.isna(salary_min) and pd.isna(salary_max):
        return 50.0, False

    if pd.notna(salary_min) and pd.notna(salary_max):
        low = float(salary_min)
        high = float(salary_max)
    elif pd.notna(salary_min):
        low = float(salary_min)
        high = low
    else:
        low = float(salary_max)
        high = low

    if low <= expectation <= high:
        return 100.0, True

    distance = min(abs(expectation - low), abs(expectation - high))
    score = max(0.0, 100.0 - (distance / max(expectation, 1.0)) * 100.0)

    return score, True


def calculate_smart_score(resume_text, row, cosine_score):
    """
    Combine multiple explainable signals into one overall score.

    Base weights:
      TF-IDF similarity : 50%
      Skills            : 25%
      Experience        : 15%
      Location          : 5%
      Salary            : 5%

    Missing salary information is excluded and the remaining weights are
    renormalized, so a missing salary field does not unfairly lower a job.
    """
    skill_score, matched_skills = skill_match_score(resume_text, row)
    exp_score = experience_match_score(
        resume_text,
        row.get("experience"),
    )
    loc_score = location_match_score(
        resume_text,
        row.get("location"),
    )
    salary_score, salary_available = salary_match_score(
        resume_text,
        row,
    )

    components = {
        "tfidf_score": float(np.clip(cosine_score, 0, 100)),
        "skills_score": float(np.clip(skill_score, 0, 100)),
        "experience_score": float(np.clip(exp_score, 0, 100)),
        "location_score": float(np.clip(loc_score, 0, 100)),
        "salary_score": float(np.clip(salary_score, 0, 100)),
    }

    weights = {
        "tfidf_score": 0.50,
        "skills_score": 0.25,
        "experience_score": 0.15,
        "location_score": 0.05,
        "salary_score": 0.05,
    }

    if not salary_available:
        weights["salary_score"] = 0.0

    weight_total = sum(weights.values())

    overall = sum(
        components[key] * weights[key]
        for key in components
    ) / weight_total

    components["overall_score"] = float(np.clip(overall, 0, 100))
    components["salary_available"] = salary_available
    components["matched_skills"] = matched_skills

    return components


def get_matching_terms(resume_text, row, top_n=6):
    resume_terms = get_resume_keywords(resume_text, top_n=30)

    text_parts = []

    for column in ["title", "company", "description", "skills", "combined_text"]:
        if column in row.index and pd.notna(row.get(column)):
            text_parts.append(str(row.get(column)).lower())

    job_text = " ".join(text_parts)

    matching = [
        term for term in resume_terms
        if term.lower() in job_text
    ]

    return matching[:top_n]


def render_skill_chips(skills):
    if not skills:
        return

    chips = " ".join(
        f'<span class="skill-chip">✓ {skill}</span>'
        for skill in skills
    )

    st.markdown(chips, unsafe_allow_html=True)


# ============================================================
# RESUME TEXT EXTRACTION
# ============================================================

def extract_resume_text(uploaded_file):
    document = Document(uploaded_file)

    text_parts = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            text_parts.append(text)

    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                text = cell.text.strip()

                if text:
                    text_parts.append(text)

    return " ".join(text_parts)


# ============================================================
# FILTER ENGINE
# ============================================================

def build_filter_mask(location, experience, minimum_salary):
    mask = np.ones(len(jobs), dtype=bool)

    if location != "All Locations":
        location_mask = (
            jobs["location"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(str(location).strip().lower())
            .to_numpy()
        )
        mask &= location_mask

    if experience != "All Experience Levels":
        experience_mask = (
            jobs["experience"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(str(experience).strip().lower())
            .to_numpy()
        )
        mask &= experience_mask

    if minimum_salary > 0:
        salary_values = jobs["salary_min_yearly"].fillna(-1).to_numpy()
        mask &= salary_values >= minimum_salary

    return mask


# ============================================================
# JOB RECOMMENDATION ENGINE
# ============================================================

def recommend_jobs(
    resume_text,
    location="All Locations",
    experience="All Experience Levels",
    minimum_salary=0,
    top_n=10,
):
    mask = build_filter_mask(
        location,
        experience,
        minimum_salary,
    )

    filtered_indices = np.flatnonzero(mask)

    if len(filtered_indices) == 0:
        return jobs.iloc[0:0].copy()

    resume_vector = tfidf_vectorizer.transform([resume_text])
    filtered_tfidf_matrix = tfidf_matrix[filtered_indices]

    similarity_scores = cosine_similarity(
        resume_vector,
        filtered_tfidf_matrix,
    ).ravel() * 100.0

    result = jobs.iloc[filtered_indices].copy()

    smart_scores = []
    tfidf_scores = []
    skills_scores = []
    experience_scores = []
    location_scores = []
    salary_scores = []
    salary_available = []
    matched_skills_list = []

    for row, cosine_score in zip(
        result.to_dict("records"),
        similarity_scores,
    ):
        row_series = pd.Series(row)

        components = calculate_smart_score(
            resume_text,
            row_series,
            cosine_score,
        )

        smart_scores.append(components["overall_score"])
        tfidf_scores.append(components["tfidf_score"])
        skills_scores.append(components["skills_score"])
        experience_scores.append(components["experience_score"])
        location_scores.append(components["location_score"])
        salary_scores.append(components["salary_score"])
        salary_available.append(components["salary_available"])
        matched_skills_list.append(components["matched_skills"])

    result["match_score"] = smart_scores
    result["tfidf_score"] = tfidf_scores
    result["skills_score"] = skills_scores
    result["experience_score"] = experience_scores
    result["location_score"] = location_scores
    result["salary_score"] = salary_scores
    result["salary_available"] = salary_available
    result["_matched_skills"] = matched_skills_list

    result = result.sort_values(
        "match_score",
        ascending=False,
        kind="stable",
    )

    duplicate_columns = [
        c
        for c in ["title", "company", "location", "url"]
        if c in result.columns
    ]

    if duplicate_columns:
        result = result.drop_duplicates(
            subset=duplicate_columns,
            keep="first",
        )

    score_columns = [
        "match_score",
        "tfidf_score",
        "skills_score",
        "experience_score",
        "location_score",
        "salary_score",
    ]

    for column in score_columns:
        result[column] = result[column].round(2)

    return result.head(top_n)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

with st.sidebar:
    st.header("⚙️ Job Filters")
    st.write("Customize your recommendations.")

    locations = sorted(
        jobs["location"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    selected_location = st.selectbox(
        "📍 Location",
        ["All Locations"] + locations,
    )

    experiences = sorted(
        jobs["experience"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    selected_experience = st.selectbox(
        "💼 Experience",
        ["All Experience Levels"] + experiences,
    )

    salary_values = jobs["salary_min_yearly"].dropna()

    if len(salary_values):
        # Keep the slider practical while still data-driven.
        p99 = float(salary_values.quantile(0.99))
        salary_max = int(min(max(p99, 250000), 1000000))
    else:
        salary_max = 250000

    selected_salary = st.slider(
        "💰 Minimum Salary",
        min_value=0,
        max_value=salary_max,
        value=0,
        step=5000,
        format="$%d",
    )

    if selected_salary > 0:
        st.caption(f"Yearly salary ≥ ${selected_salary:,.0f}")
    else:
        st.caption("Any salary")

    filtered_count = int(
        build_filter_mask(
            selected_location,
            selected_experience,
            selected_salary,
        ).sum()
    )

    st.divider()

    st.info(
        f"📊 **{len(jobs):,} jobs** in database.\n\n"
        f"🔎 **{filtered_count:,} jobs** match current filters."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💼 SmartHire</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Resume & Job Recommendation System'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Jobs", f"{len(jobs):,}")

with col2:
    st.metric(
        "TF-IDF Features",
        f"{len(tfidf_vectorizer.get_feature_names_out()):,}",
    )

with col3:
    st.metric("Model", "TF-IDF")

with col4:
    st.metric("Similarity", "Cosine")


st.divider()


# ============================================================
# RESUME UPLOAD
# ============================================================

st.header("📄 Upload Your Resume")

uploaded_file = st.file_uploader(
    "Upload your resume in DOCX format",
    type=["docx"],
    help="Upload a .docx resume to receive personalized recommendations.",
)

if uploaded_file is not None:
    current_signature = (
        uploaded_file.name,
        uploaded_file.size,
    )

    previous_signature = st.session_state.get("resume_signature")

    if current_signature != previous_signature:
        st.session_state["resume_signature"] = current_signature
        st.session_state.pop("recommendations", None)
        st.session_state.pop("resume_text", None)


# ============================================================
# PROCESS RESUME
# ============================================================

if uploaded_file is not None:

    try:
        resume_text = extract_resume_text(uploaded_file)

    except Exception as e:
        st.error("❌ Could not read this DOCX file.")
        st.code(str(e))
        st.stop()

    if not resume_text.strip():
        st.error("❌ No readable text was found in this resume.")

    else:
        st.session_state["resume_text"] = resume_text

        st.success("✅ Resume uploaded and processed successfully!")

        st.header("🧠 Resume Insights")

        word_count = len(resume_text.split())
        character_count = len(resume_text)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Words", f"{word_count:,}")

        with col2:
            st.metric("Characters", f"{character_count:,}")

        with col3:
            st.metric("Format", "DOCX")

        with st.expander("📖 View Extracted Resume Text"):
            st.text_area(
                "Extracted resume content",
                resume_text,
                height=220,
                label_visibility="collapsed",
            )

        with st.expander("🔑 Key Resume Terms"):
            terms = get_resume_keywords(resume_text, top_n=15)

            if terms:
                st.write(
                    "Strongest TF-IDF terms detected in your resume:"
                )
                st.write(" • ".join(terms))
            else:
                st.write("No significant TF-IDF terms were detected.")

        st.divider()

        if st.button(
            "🚀 Find My Best Job Matches",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                "🤖 Comparing your resume with available jobs..."
            ):
                recommendations = recommend_jobs(
                    resume_text=resume_text,
                    location=selected_location,
                    experience=selected_experience,
                    minimum_salary=selected_salary,
                    top_n=10,
                )

            st.session_state["recommendations"] = recommendations


# ============================================================
# SHOW RECOMMENDATIONS
# ============================================================

if "recommendations" in st.session_state:

    recommendations = st.session_state["recommendations"]

    st.divider()
    st.header("🎯 Your Top Job Matches")

    if len(recommendations) == 0:
        st.warning(
            "No jobs matched the selected filters. "
            "Try relaxing the location, experience, or salary filters."
        )

    else:
        best_score = float(recommendations["match_score"].max())
        average_score = float(recommendations["match_score"].mean())

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Jobs Found", len(recommendations))

        with col2:
            st.metric("Best Match", f"{best_score:.2f}%")

        with col3:
            st.metric("Average Match", f"{average_score:.2f}%")

        with col4:
            strong_matches = int(
                (recommendations["match_score"] >= 50).sum()
            )
            st.metric("50%+ Matches", strong_matches)

        st.caption(
            "Match score is the cosine similarity between the resume "
            "TF-IDF vector and each job's TF-IDF vector."
        )

        # --------------------------------------------------------
        # BEST MATCH
        # --------------------------------------------------------

        best_job = recommendations.iloc[0]

        best_title = safe_text(
            best_job.get("title"),
            "Job Title Not Available",
        )

        best_company = safe_text(
            best_job.get("company"),
            "Company Not Specified",
        )

        best_location = safe_text(
            best_job.get("location"),
            "Location Not Specified",
        )

        best_label, _ = get_score_label(best_score)

        st.subheader("🏆 Best Match")

        with st.container(border=True):
            top_col1, top_col2 = st.columns([4, 1])

            with top_col1:
                st.markdown(f"### {best_title.title()}")
                st.write(f"🏢 **{best_company}**")
                st.write(f"📍 {best_location}")
                st.write(
                    f"💼 {safe_text(best_job.get('experience'))}"
                )
                st.write(f"💰 {format_salary(best_job)}")

            with top_col2:
                st.metric(
                    "Match Score",
                    f"{best_score:.2f}%",
                )
                st.write(best_label)

            st.progress(
                min(best_score / 100, 1.0),
                text=f"Resume Match: {best_score:.2f}%",
            )

            st.markdown("#### 🧠 Why this job matched")

            breakdown_cols = st.columns(5)

            breakdown = [
                ("TF-IDF", best_job.get("tfidf_score", 0)),
                ("Skills", best_job.get("skills_score", 0)),
                ("Experience", best_job.get("experience_score", 0)),
                ("Location", best_job.get("location_score", 0)),
                ("Salary", best_job.get("salary_score", 50)),
            ]

            for column, (label, value) in zip(breakdown_cols, breakdown):
                with column:
                    st.metric(label, f"{float(value):.0f}%")

            st.caption(
                "Score weights: TF-IDF 50% • Skills 25% • Experience 15% • "
                "Location 5% • Salary 5%. Missing salary information is excluded "
                "and the remaining weights are renormalized."
            )

            matching_skills = list(best_job.get("_matched_skills", []))

            if matching_skills:
                st.write("**Matched skills:**")
                render_skill_chips(matching_skills[:8])
            else:
                matching_terms = get_matching_terms(
                    resume_text,
                    best_job,
                    top_n=6,
                )
                if matching_terms:
                    st.write("**Matching resume terms:**")
                    render_skill_chips(matching_terms)
                else:
                    st.caption("No strong exact skill/keyword overlap was detected.")

            url = best_job.get("url")

            if pd.notna(url) and str(url).strip():
                st.link_button(
                    "🔗 View Best-Match Job",
                    str(url),
                )

        # --------------------------------------------------------
        # ALL JOB CARDS
        # --------------------------------------------------------

        st.subheader("📋 Recommended Jobs")

        for i, (_, row) in enumerate(
            recommendations.iterrows(),
            start=1,
        ):
            title = safe_text(
                row.get("title"),
                "Job Title Not Available",
            )

            company = safe_text(
                row.get("company"),
                "Company Not Specified",
            )

            location = safe_text(
                row.get("location"),
                "Location Not Specified",
            )

            experience = safe_text(
                row.get("experience"),
                "Not Specified",
            )

            score = float(row["match_score"])
            score_label, score_class = get_score_label(score)

            st.markdown(f"### {i}. {title.title()}")

            with st.container(border=True):
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.write(f"🏢 **{company}**")
                    st.write(f"📍 {location}")
                    st.write(f"💼 {experience}")
                    st.write(f"💰 {format_salary(row)}")

                    with st.expander("🧠 Why this match?"):
                        score_items = [
                            ("TF-IDF similarity", row.get("tfidf_score", 0)),
                            ("Skills", row.get("skills_score", 0)),
                            ("Experience", row.get("experience_score", 0)),
                            ("Location", row.get("location_score", 0)),
                            ("Salary", row.get("salary_score", 50)),
                        ]

                        for label, value in score_items:
                            st.progress(
                                min(float(value) / 100.0, 1.0),
                                text=f"{label}: {float(value):.1f}%",
                            )

                        matching_skills = list(
                            row.get("_matched_skills", [])
                        )

                        if matching_skills:
                            st.write("**Matched skills:**")
                            render_skill_chips(matching_skills[:6])
                        else:
                            matching_terms = get_matching_terms(
                                resume_text,
                                row,
                                top_n=5,
                            )
                            if matching_terms:
                                st.write("**Matching resume terms:**")
                                render_skill_chips(matching_terms)
                            else:
                                st.caption(
                                    "No strong exact skill/keyword overlap was detected."
                                )

                with col2:
                    st.metric(
                        "Match",
                        f"{score:.2f}%",
                    )

                    st.markdown(
                        f'<div class="{score_class}">'
                        f'{score_label}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.progress(
                        min(score / 100, 1.0),
                        text=f"{score:.2f}%",
                    )

                    url = row.get("url")

                    if pd.notna(url) and str(url).strip():
                        st.link_button(
                            "🔗 View Job",
                            str(url),
                        )

        # --------------------------------------------------------
        # RESULTS TABLE
        # --------------------------------------------------------

        with st.expander("📊 View Results as a Table"):
            table_columns = [
                "title",
                "company",
                "location",
                "experience",
                "salary_min",
                "salary_max",
                "pay_period",
                "match_score",
                "tfidf_score",
                "skills_score",
                "experience_score",
                "location_score",
                "salary_score",
            ]

            available_table_columns = [
                c for c in table_columns
                if c in recommendations.columns
            ]

            st.dataframe(
                recommendations[available_table_columns],
                use_container_width=True,
                hide_index=True,
            )

        # --------------------------------------------------------
        # EXPORT
        # --------------------------------------------------------

        st.divider()
        st.subheader("📥 Export Recommendations")

        export_columns = [
            "title",
            "company",
            "location",
            "experience",
            "salary_min",
            "salary_max",
            "pay_period",
            "match_score",
            "tfidf_score",
            "skills_score",
            "experience_score",
            "location_score",
            "salary_score",
            "url",
        ]

        available_columns = [
            c for c in export_columns
            if c in recommendations.columns
        ]

        download_data = recommendations[available_columns].copy()

        csv_data = download_data.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Recommendations as CSV",
            data=csv_data,
            file_name="smarthire_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption("SmartHire | AI-Powered Job Recommendation System")
st.caption("Machine Learning: TF-IDF + Cosine Similarity + Explainable Smart Scoring")

with st.expander("ℹ️ How SmartHire works"):
    st.markdown(
        """
        **1. Resume Upload**  
        The system extracts text from the uploaded DOCX resume.

        **2. TF-IDF Representation**  
        The resume is converted into the same 10,000-feature TF-IDF
        space used by the saved job dataset.

        **3. Filtering**  
        Location, experience and minimum yearly salary filters are
        applied before similarity calculation.

        **4. Cosine Similarity**  
        The filtered job vectors are compared with the resume vector.

        **5. Smart Match Score**  
        The system combines TF-IDF similarity (50%), skills (25%),
        experience (15%), location (5%) and salary (5%). If salary
        information is unavailable, that component is excluded and the
        remaining weights are normalized.

        **6. Ranking & Explanation**  
        Jobs are sorted by the overall smart score, while the individual
        components are displayed so the recommendation is explainable.
        """
    )
