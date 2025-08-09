# Scoring & ranking pipeline for job-resume matching

import re
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "ranker.py requires sentence-transformers. Install with: pip install sentence-transformers"
    ) from e


# Utilities & Shared Helpers
MONTHS = {
    'jan':1,'january':1,
    'feb':2,'february':2,
    'mar':3,'march':3,
    'apr':4,'april':4,
    'may':5,
    'jun':6,'june':6,
    'jul':7,'july':7,
    'aug':8,'august':8,
    'sep':9,'sept':9,'september':9,
    'oct':10,'october':10,
    'nov':11,'november':11,
    'dec':12,'december':12
}

def normalize_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip())

def concat_sentences(parts: List[str]) -> str:
    parts = [normalize_text(p) for p in (parts or []) if normalize_text(p)]
    return " ".join(parts)

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def months_between(a: datetime, b: datetime) -> int:
    # both must be tz-aware; if naive, assume UTC
    if a.tzinfo is None:
        a = a.replace(tzinfo=timezone.utc)
    if b.tzinfo is None:
        b = b.replace(tzinfo=timezone.utc)
    if a < b:
        a, b = b, a
    return (a.year - b.year) * 12 + (a.month - b.month)

def parse_end_date(duration_text: str) -> datetime:
    """
    Parse end date from strings like:
      "June 2021 – September 2021"
      "May 2022 – Aug 2022 (3 mos)"
      "July 2022 – Present"
    Return aware datetime (month-end approx day=28).
    If missing/unknown, assume Present (now).
    """
    if not duration_text:
        return utcnow()
    t = duration_text.lower()
    if "present" in t:
        return utcnow()
    m = re.search(r"[–\-]\s*([a-zA-Z]{3,9})\s+(\d{4})", duration_text)
    if m:
        mon = MONTHS.get(m.group(1).lower(), None)
        year = int(m.group(2))
        if mon:
            return datetime(year, mon, 28, tzinfo=timezone.utc)
    candidates = re.findall(r"([a-zA-Z]{3,9})\s+(\d{4})", duration_text)
    if candidates:
        mon = MONTHS.get(candidates[-1][0].lower(), 12)
        year = int(candidates[-1][1])
        return datetime(year, mon, 28, tzinfo=timezone.utc)
    return utcnow()

def recency_factor(end_date: datetime) -> float:
    """
    No penalty if end date ≤ 6 months old.
    Else penalty fraction = years_old / 10 (subtracted via factor).
    factor = 1 - years_old/10 (clamped to [0,1])
    """
    m_old = months_between(utcnow(), end_date)
    years_old = m_old / 12.0
    if years_old <= 0.5:
        return 1.0
    factor = 1.0 - (years_old / 10.0)
    return max(0.0, min(1.0, factor))

def clamp01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return max(0.0, min(1.0, float(x)))


# Embedding Model (singleton)
class E5Model:
    _instance: Optional["E5Model"] = None

    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @classmethod
    def instance(cls) -> "E5Model":
        if cls._instance is None:
            cls._instance = E5Model()
        return cls._instance

    def encode(self, texts: List[str]) -> np.ndarray:
        texts = [f"query: {normalize_text(t)}" if t else "query: " for t in texts]
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


# Education Scorer
DEGREE_TOKEN = r"(?:Bachelor(?:'s)?|Master(?:'s)?|Ph\.?\s?D\.?|M\.?\s?S\.?|B\.?\s?S\.?|BSc|MSc|BE|ME|BTech|MTech|B\.?\s?Eng|M\.?\s?Eng|MBA)"
DEGREE_TOKEN_RE = re.compile(DEGREE_TOKEN, re.IGNORECASE)

def split_fields(fields_text: str) -> List[str]:
    if not fields_text:
        return []
    t = re.sub(r"\s*(?:\band\b|\bor\b|/)\s*", ",", fields_text, flags=re.IGNORECASE)
    parts = [normalize_text(p) for p in t.split(",")]
    return [re.sub(r"[.;]+$", "", p).strip() for p in parts if p.strip()]

def extract_job_degrees_fields(qual_text: str) -> Tuple[List[str], List[str]]:
    if not qual_text:
        return [], []
    text = qual_text.strip()
    m_in = re.search(r"\bin\b\s+([^.;\n]+)", text, flags=re.IGNORECASE)
    fields_text = m_in.group(1) if m_in else ""
    job_fields = split_fields(fields_text)
    degrees_region = text[:m_in.start()] if m_in else text
    job_degrees = DEGREE_TOKEN_RE.findall(degrees_region)
    seen = set()
    degs = []
    for d in job_degrees:
        dd = normalize_text(d)
        if dd.lower() not in seen:
            seen.add(dd.lower())
            degs.append(dd)
    return degs, job_fields

class EducationScorer:
    def __init__(self, job_dict: Dict[str, Any]):
        self.job_degrees, self.job_fields = extract_job_degrees_fields(job_dict.get("qualifications", ""))
        self.model = E5Model.instance()
        self.job_deg_embs = self.model.encode(self.job_degrees or [""])
        self.job_fld_embs = self.model.encode(self.job_fields or [""])

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _score_one(self, degree_text: str, field_text: str) -> float:
        if not (degree_text or field_text):
            return 0.0
        d_sim = 0.0
        f_sim = 0.0
        if degree_text and self.job_deg_embs.size:
            cand_deg = self.model.encode([degree_text])[0]
            d_sim = max(self._cos(cand_deg, self.job_deg_embs[i]) for i in range(self.job_deg_embs.shape[0]))
        if field_text and self.job_fld_embs.size:
            cand_fld = self.model.encode([field_text])[0]
            f_sim = max(self._cos(cand_fld, self.job_fld_embs[i]) for i in range(self.job_fld_embs.shape[0]))
        if degree_text and field_text:
            return (d_sim + f_sim) / 2.0
        return d_sim if degree_text else f_sim

    def score_resume(self, resume: Dict[str, Any]) -> float:
        edus = resume.get("education", []) or []
        if not edus:
            return 0.0
        scores = []
        for edu in edus:
            deg = normalize_text(edu.get("degree", ""))
            fld = normalize_text(edu.get("field", ""))
            scores.append(self._score_one(deg, fld))
        return float(np.mean(scores)) if scores else 0.0


# Experience Scorer (with recency)
class ExperienceScorer:
    def __init__(self, job_dict: Dict[str, Any]):
        self.model = E5Model.instance()
        resp = job_dict.get("responsibilities", []) or []
        self.job_text = concat_sentences(resp)
        self.job_emb = self.model.encode([self.job_text])[0]

    def score_resume(self, resume: Dict[str, Any]) -> float:
        exps = resume.get("experience", []) or []
        if not exps:
            return 0.0
        texts = []
        metas = []
        for i, exp in enumerate(exps, 1):
            bullets = exp.get("work_description", []) or []
            t = concat_sentences(bullets)
            texts.append(t)
            metas.append({"idx": i, "duration": exp.get("duration", "")})
        if not any(texts):
            return 0.0
        embs = self.model.encode(texts)
        sims = np.dot(embs, self.job_emb)  # normalized cosine

        # Apply recency factor + select top ≤3
        adjusted = []
        for k, m in enumerate(metas):
            end_dt = parse_end_date(m["duration"])
            factor = recency_factor(end_dt)
            adjusted.append(max(0.0, float(sims[k]) * factor))
        if not adjusted:
            return 0.0
        top = sorted(adjusted, reverse=True)[:3]
        return float(np.mean(top)) if top else 0.0

# Project Scorer (no recency)
class ProjectScorer:
    def __init__(self, job_dict: Dict[str, Any]):
        self.model = E5Model.instance()
        resp = job_dict.get("responsibilities", []) or []
        self.job_text = concat_sentences(resp)
        self.job_emb = self.model.encode([self.job_text])[0]

    def _project_text(self, p: Dict[str, Any]) -> str:
        name = normalize_text(p.get("project_name", ""))
        desc = concat_sentences(p.get("project_description", []) or [])
        skills = [normalize_text(s) for s in (p.get("skills") or []) if normalize_text(s)]
        comp = []
        if name:
            comp.append(name)
        if desc:
            comp.append(desc)
        if skills:
            comp.append("Skills: " + ", ".join(skills))
        return ". ".join(comp)

    def score_resume(self, resume: Dict[str, Any]) -> float:
        projects = resume.get("projects", []) or []
        texts = [self._project_text(p) for p in projects]
        texts = [t for t in texts if t]
        if not texts:
            return 0.0
        embs = self.model.encode(texts)
        sims = np.dot(embs, self.job_emb)  # normalized cosine
        top = sorted([float(s) for s in sims], reverse=True)[:3]
        return float(np.mean(top)) if top else 0.0


# Skills Scorer (semantic %)
def norm_skill(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[\(\)\[\]\{\}:/\\|]", " ", s)
    s = re.sub(r"[^a-z0-9\.\+\#\-\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

class SkillsScorer:
    def __init__(self, job_dict: Dict[str, Any], threshold: float = 0.70):
        self.model = E5Model.instance()
        job_skills_raw = job_dict.get("skills", []) or []
        self.job_skills = [norm_skill(s) for s in job_skills_raw if norm_skill(s)]
        self.job_skills = list(dict.fromkeys(self.job_skills))
        self.threshold = threshold
        self.job_embs = self.model.encode(self.job_skills or [""])

    def score_resume(self, resume: Dict[str, Any]) -> float:
        # Collect skills only from experiences and projects
        cand_set = set()
        for exp in resume.get("experience", []) or []:
            for s in exp.get("skills", []) or []:
                ns = norm_skill(s)
                if ns:
                    cand_set.add(ns)
        for proj in resume.get("projects", []) or []:
            for s in proj.get("skills", []) or []:
                ns = norm_skill(s)
                if ns:
                    cand_set.add(ns)

        cand_skills = sorted(cand_set)
        if not self.job_skills:
            return 0.0
        if not cand_skills:
            return 0.0

        cand_embs = self.model.encode(cand_skills)
        matched = 0
        for j in range(len(self.job_skills)):
            sims = np.dot(cand_embs, self.job_embs[j])  # cosine
            if float(np.max(sims)) >= self.threshold:
                matched += 1
        return matched / len(self.job_skills)



# Ranker Orchestrator

@dataclass
class RankResult:
    resume_id: str
    candidate_name: str
    score: float
    rank: int

class Ranker:
    # Base weights (renormalized automatically if some components missing)
    BASE_WEIGHTS = {
        "experience_score": 0.60,
        "project_score":    0.35,
        "skills_score":     0.025,
        "education_score":  0.025,
    }

    def __init__(self, job_dict: Dict[str, Any]):
        self.job_dict = job_dict
        self.education = EducationScorer(job_dict)
        self.experience = ExperienceScorer(job_dict)
        self.projects = ProjectScorer(job_dict)
        self.skills = SkillsScorer(job_dict)

    @staticmethod
    def _safe_float(x) -> Optional[float]:
        if x is None:
            return None
        try:
            return clamp01(float(x))
        except Exception:
            return None

    def _composite(self, rec: Dict[str, Any]) -> float:
        parts = []
        total_w = 0.0
        for key, w in self.BASE_WEIGHTS.items():
            v = self._safe_float(rec.get(key))
            if v is not None:
                parts.append((v, w))
                total_w += w
        if not parts or total_w <= 0.0:
            return 0.0
        score = sum(v * (w / total_w) for v, w in parts)
        return clamp01(score) or 0.0

    def build_initial_results_dict(self, resumes_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        results = {}
        for r in resumes_list:
            rid = r.get("resume_id") or r.get("candidate_name") or f"rid-{len(results)+1}"
            results[rid] = {
                "job_id": self.job_dict.get("job_title") or "job",
                "resume_id": rid,
                "candidate_name": r.get("candidate_name", "Unknown"),
                "education_score": None,
                "qualification_score": None,   # reserved if you add a separate qualification scorer later
                "project_score": None,
                "experience_score": None,
                "skills_score": None,
                "composite": None,
                "rank": None
            }
        return results

    def score_all(self, resumes_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        results = self.build_initial_results_dict(resumes_list)

        # Compute each component and update results
        for r in resumes_list:
            rid = r.get("resume_id")
            # Education
            results[rid]["education_score"] = self.education.score_resume(r)
            # Experience (with recency, top ≤3)
            results[rid]["experience_score"] = self.experience.score_resume(r)
            # Projects (no recency, top ≤3)
            results[rid]["project_score"] = self.projects.score_resume(r)
            # Skills (semantic %)
            results[rid]["skills_score"] = self.skills.score_resume(r)

        # Composite
        for rid, rec in results.items():
            rec["composite"] = self._composite(rec)

        # Rank (dense)
        ranked_ids = sorted(
            results.keys(),
            key=lambda k: (-(self._safe_float(results[k]["composite"]) or 0.0),
                           results[k]["candidate_name"].lower(),
                           k)
        )
        last_score = None
        current_rank = 0
        for idx, rid in enumerate(ranked_ids, start=1):
            score = self._safe_float(results[rid]["composite"]) or 0.0
            if last_score is None or not math.isclose(score, last_score, rel_tol=1e-9, abs_tol=1e-9):
                current_rank = idx
                last_score = score
            results[rid]["rank"] = current_rank

        return results


# API
def rank_candidates(job_dict: Dict[str, Any], resumes_list: List[Dict[str, Any]]) -> List[RankResult]:
    """
    Main entrypoint: compute scores & return ranked list for UI.
    Returns: List of RankResult (resume_id, candidate_name, composite score, rank)
    """
    ranker = Ranker(job_dict)
    results_dict = ranker.score_all(resumes_list)
    # Build a compact list for Streamlit display
    out: List[RankResult] = []
    for rid, rec in results_dict.items():
        out.append(RankResult(
            resume_id=rid,
            candidate_name=rec.get("candidate_name", "Unknown"),
            score=rec.get("composite") or 0.0,
            rank=rec.get("rank") or 0
        ))
    # Sort by rank
    out.sort(key=lambda r: (r.rank, r.candidate_name.lower()))
    return out
