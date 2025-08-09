import json
import os
import streamlit as st

import parser as parser 
import ranker as ranker         

st.set_page_config(page_title="Sprouts AI Matcher", layout="wide")
st.title("Sprouts AI: Candidate Recommender")

# Paths to sample files based on your folder structure
SAMPLE_JOB_PATH = os.path.join("File", "Job.txt")
SAMPLE_RESUMES_PATH = os.path.join("File", "Resume.txt")  

def read_file_safe(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[ERROR] Could not read sample file at {path}: {e}"


# Session state (buttons populate the textareas)
if "job_text" not in st.session_state:
    st.session_state.job_text = ""
if "resumes_text" not in st.session_state:
    st.session_state.resumes_text = ""


# Layout
col1, col2 = st.columns(2)

with col1:
    st.header("1. Enter Job")

    # Load sample job button
    if st.button("Load Sample Job"):
        sample_job_text = read_file_safe(SAMPLE_JOB_PATH)
        st.session_state.job_text = sample_job_text

    # Job textarea bound to session state
    st.text_area(
        label="Paste the full Job Description here",
        height=440,
        key="job_text",
        placeholder=(
            "Job Title: ...\n"
            "Company: ...\n"
            "Location: ...\n\n"
            "Job Description: ...\n"
            "Role Overview: ...\n\n"
            "Responsibilities:\n"
            "- ...\n"
            "- ...\n\n"
            "Qualifications: ...\n\n"
            "Skills: ...\n"
        ),
    )

with col2:
    st.header("2. Enter Resume(s)")

    # Load sample resumes button
    if st.button("Load Sample Resumes"):
        sample_resumes_text = read_file_safe(SAMPLE_RESUMES_PATH)
        st.session_state.resumes_text = sample_resumes_text

    # Resumes textarea bound to session state
    st.text_area(
        label="Paste one or more resumes below (use the provided format for each):",
        height=440,
        key="resumes_text",
        placeholder=(
            "ResumeID: resume-1\n"
            "Candidate Name: [Enter Candidate Name]\n\n"
            "Education 1:\n"
            "  University/College: [Enter University/College Name]\n"
            "  Degree: [Enter Degree]\n"
            "  Field: [Enter Field of Study]\n"
            "  Duration: [Enter Duration]\n"
            "  GPA: [Optional]\n\n"
            "Experience 1:\n"
            "  Company: [Enter Company Name]\n"
            "  Position: [Enter Position/Title]\n"
            "  Duration: [Enter Duration]\n"
            "  Work Description:\n"
            "    - [Bullet...]\n"
            "  Skills: [List, comma separated]\n\n"
            "Project 1:\n"
            "  Project Name: [Enter Project Name]\n"
            "  Duration: [Optional]\n"
            "  Project Description:\n"
            "    - [Bullet...]\n"
            "  Skills: [List]\n\n"
            "---\n"
            "ResumeID: resume-2\n"
            "..."
        ),
    )

# Recommend button + results
job_desc = (st.session_state.job_text or "").strip()
resumes_text = (st.session_state.resumes_text or "").strip()

if job_desc and resumes_text:
    st.markdown("---")
    button_center = st.columns([0.2, 0.6, 0.2])
    with button_center[1]:
        if st.button("Recommend Candidates", use_container_width=True):
            # Parse
            job_dict = parser.parse_job_description(job_desc)
            resumes_list = parser.parse_resumes_text(resumes_text)

            # Rank
            ranked_candidates = ranker.rank_candidates(job_dict, resumes_list)

            # Show ranked results
            st.subheader("Ranked Candidates")
            rows = [
                {
                    "Rank": r.rank,
                    "Candidate": r.candidate_name,
                    "Resume ID": r.resume_id,
                    "Score": round(r.score, 3),
                }
                for r in ranked_candidates
            ]
            st.table(rows)

            with st.expander("Show Parsed Job Description"):
                st.code(json.dumps(job_dict, indent=2))
            with st.expander("Show Parsed Resumes"):
                st.code(json.dumps(resumes_list, indent=2))
else:
    st.info("Click 'Load Sample Job' and 'Load Sample Resumes' (or paste your own) to enable recommendations.")
