import re


def parse_job_description(job_text: str):
    job = {}
    job["job_title"] = re.search(r"Job Title:\s*(.*)", job_text).group(1).strip() if re.search(r"Job Title:\s*(.*)", job_text) else ""
    job["company"] = re.search(r"Company:\s*(.*)", job_text).group(1).strip() if re.search(r"Company:\s*(.*)", job_text) else ""
    job["location"] = re.search(r"Location:\s*(.*)", job_text).group(1).strip() if re.search(r"Location:\s*(.*)", job_text) else ""
    job["job_description"] = re.search(r"Job Description:\s*([\s\S]*?)\nRole Overview:", job_text).group(1).strip() if re.search(r"Job Description:\s*([\s\S]*?)\nRole Overview:", job_text) else ""
    job["role_overview"] = re.search(r"Role Overview:\s*([\s\S]*?)\nResponsibilities:", job_text).group(1).strip() if re.search(r"Role Overview:\s*([\s\S]*?)\nResponsibilities:", job_text) else ""
    job["responsibilities"] = []
    match_resp = re.search(r"Responsibilities:\s*([\s\S]*?)\nQualifications:", job_text)
    if match_resp:
        job["responsibilities"] = [line.strip('- ').strip() for line in match_resp.group(1).split('\n') if line.strip()]
    job["qualifications"] = re.search(r"Qualifications:\s*(.*)", job_text).group(1).strip() if re.search(r"Qualifications:\s*(.*)", job_text) else ""
    skills = re.search(r"Skills:\s*(.*)", job_text)
    job["skills"] = [s.strip() for s in skills.group(1).split(',')] if skills else []
    return job


def _non_empty_list(lst):
    """Return a list with empty/blank strings removed."""
    return [x for x in (lst or []) if isinstance(x, str) and x.strip()]


def _has_project_content(proj: dict) -> bool:
    """True if project has any non-empty content."""
    if not proj:
        return False
    if proj.get("project_name", "").strip():
        return True
    if _non_empty_list(proj.get("project_description")):
        return True
    if _non_empty_list(proj.get("skills")):
        return True
    return False


def parse_resumes_text(resumes_text: str):
    resumes = []
    current_resume = {}
    section = None

    in_work_description = False
    in_project_description = False

    pending_project = None
    lines = resumes_text.splitlines()

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        s = line.strip()

        # New resume
        if s.startswith("ResumeID:"):
            if pending_project and _has_project_content(pending_project):
                current_resume.setdefault("projects", []).append(pending_project)
            pending_project = None
            if current_resume:
                resumes.append(current_resume)
            current_resume = {
                "resume_id": s.split(":", 1)[1].strip(),
                "candidate_name": "",
                "education": [],
                "experience": [],
                "projects": []
            }
            section = None
            in_work_description = False
            in_project_description = False
            continue

        # Candidate name
        if s.startswith("Candidate Name:"):
            current_resume["candidate_name"] = s.split(":", 1)[1].strip()
            section = None
            continue

        # Education section
        if re.match(r"^Education\s*\d*\s*:", s):
            if pending_project and _has_project_content(pending_project):
                current_resume["projects"].append(pending_project)
            pending_project = None
            section = "education"
            current_resume["education"].append({})
            in_work_description = False
            in_project_description = False
            continue

        # Experience section
        if re.match(r"^Experience\s*\d*\s*:", s):
            if pending_project and _has_project_content(pending_project):
                current_resume["projects"].append(pending_project)
            pending_project = None
            section = "experience"
            current_resume["experience"].append({"work_description": [], "skills": []})
            in_work_description = False
            in_project_description = False
            continue

        # Projects: None listed
        if s.lower().startswith("projects:"):
            if pending_project and _has_project_content(pending_project):
                current_resume["projects"].append(pending_project)
            pending_project = None
            section = None
            continue

        # Project header
        if re.match(r"^Project\s*\d+\s*:", s):
            if pending_project and _has_project_content(pending_project):
                current_resume.setdefault("projects", []).append(pending_project)
            project_id = s.split(":", 1)[0].strip()
            pending_project = {
                "project_id": project_id,
                "project_name": "",
                "duration": "Not listed",
                "project_description": [],
                "skills": []
            }
            section = "project"
            in_project_description = False
            continue

        # Education fields
        if section == "education" and current_resume.get("education"):
            edu = current_resume["education"][-1]
            if s.startswith("University/College:"):
                edu["university"] = s.split(":", 1)[1].strip()
            elif s.startswith("Degree:"):
                edu["degree"] = s.split(":", 1)[1].strip()
            elif s.startswith("Field:"):
                edu["field"] = s.split(":", 1)[1].strip()
            elif s.startswith("Duration:"):
                edu["duration"] = s.split(":", 1)[1].strip()
            elif s.startswith("GPA:"):
                edu["gpa"] = s.split(":", 1)[1].strip()
            continue

        # Experience fields
        if section == "experience" and current_resume.get("experience"):
            exp = current_resume["experience"][-1]
            if s.startswith("Company:"):
                exp["company"] = s.split(":", 1)[1].strip()
                in_work_description = False
            elif s.startswith("Position:"):
                exp["position"] = s.split(":", 1)[1].strip()
                in_work_description = False
            elif s.startswith("Duration:"):
                exp["duration"] = s.split(":", 1)[1].strip()
                in_work_description = False
            elif s.startswith("Work Description:"):
                in_work_description = True
            elif s.startswith("Skills:"):
                exp["skills"] = [t.strip() for t in re.split(r'[;,]', s.split(":", 1)[1]) if t.strip()]
                in_work_description = False
            elif in_work_description:
                m = re.match(r'^-\s*(.*)$', s)
                if m:
                    bullet = m.group(1).strip()
                    if bullet:
                        exp["work_description"].append(bullet)
            continue

        # Project fields
        if section == "project" and pending_project is not None:
            if s.startswith("Project Name:"):
                pending_project["project_name"] = s.split(":", 1)[1].strip()
                in_project_description = False
                continue
            if s.startswith("Duration:"):
                pending_project["duration"] = s.split(":", 1)[1].strip() or "Not listed"
                in_project_description = False
                continue
            if s.startswith("Project Description:"):
                in_project_description = True
                continue
            if s.startswith("Skills:"):
                skills_part = s.split(":", 1)[1]
                pending_project["skills"] = [t.strip() for t in re.split(r'[;,]', skills_part) if t.strip()]
                in_project_description = False
                continue
            if in_project_description:
                m = re.match(r'^-\s*(.*)$', s)
                if m:
                    bullet = m.group(1).strip()
                    if bullet:
                        pending_project["project_description"].append(bullet)
                continue
            continue

    # Eof
    if pending_project and _has_project_content(pending_project):
        current_resume.setdefault("projects", []).append(pending_project)
    pending_project = None

    if current_resume:
        for proj in current_resume.get("projects", []):
            proj["project_description"] = _non_empty_list(proj.get("project_description"))
            proj["skills"] = _non_empty_list(proj.get("skills"))
        resumes.append(current_resume)

    # Cleanup + ensure projects always present
    cleaned_resumes = []
    for r in resumes:
        r = dict(r)
        good_projects = []
        for proj in r.get("projects", []):
            proj = dict(proj)
            proj["project_description"] = _non_empty_list(proj.get("project_description"))
            proj["skills"] = _non_empty_list(proj.get("skills"))
            if _has_project_content(proj):
                good_projects.append(proj)
        r["projects"] = good_projects if good_projects else []
        cleaned_resumes.append(r)

    return cleaned_resumes
