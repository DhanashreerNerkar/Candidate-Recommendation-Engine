Candidate–Job Semantic Matching & Ranking System
---------------------------------------------------------------------------------------------------------------
Overview
This project implements an end-to-end candidate evaluation pipeline that takes a job description and a set of candidate resumes (parsed into structured data) and calculates semantic similarity scores across multiple dimensions:
Education
Experience
Projects
Skills

It then applies custom weightage rules to generate a composite score for each candidate and ranks them accordingly.
The solution uses Sentence-BERT embeddings (intfloat/e5-base-v2) for semantic comparison, ensuring that contextual meaning is captured rather than relying on keyword overlap.

Thought Process & Approach
The core principle of the system is: "A great candidate match is not just about having the right words, but the right meaning and recency."

This Streamlit app ranks resumes against a given job description using semantic scoring.
Functional Flow:
1. User pastes or loads a sample Job Description and Resume list.
2. On "Recommend Candidates", the app parses inputs into structured JSON using parser.py.
3. ranker.py computes semantic similarity for Education, Experience, Projects, and Skills, applies weights, and produces a composite score.
4. Candidates are ranked in descending order of their composite scores.

Technical Flow:
1. Uses SentenceTransformer for embeddings and cosine similarity scoring.
2. Applies recency adjustment for experiences.
3. Maintains a results_dict to store and update all feature scores before ranking.

---------------------------------------------------------------------------------------------------------------
I broke down the problem into four scoring modules:
1. Education Score
Extract degrees & fields from the job description’s qualifications section.
Extract degrees & fields from candidate resumes.
Compute semantic similarity between each candidate's education and job requirements.
Average scores per candidate.

Idea:
Used regex + semantic embeddings to handle variations like "M.S." vs "Master of Science" and still match accurately.
---------------------------------------------------------------------------------------------------------------
2. Experience Score
Concatenate job responsibilities into one semantic representation.

For each candidate:
Extract work descriptions for each experience.
Compare each experience with job responsibilities.
Keep top 3 relevant experiences (highest semantic match).
Apply a recency decay factor — more recent experiences score higher.

Formula:
Recency decay formula prevents outdated experience from boosting rankings unfairly:
Adjusted Score = Semantic Score × (1 - Years Old / 10)
(No penalty for experiences within the last 6 months.)
---------------------------------------------------------------------------------------------------------------
3. Project Score
Similar to experience scoring, but:
Include project skills in the text before computing similarity.
No recency penalty — old but relevant projects can still score high.
Top 3 projects considered per candidate.

Idea:
Merging project description + skills boosts semantic richness.
---------------------------------------------------------------------------------------------------------------
4. Skills Score
Extract skills from the job description.
Match against skills used in candidate’s projects and experiences (not from the resume's plain skills section to avoid self-reported, unverifiable claims).
Score = % of job-required skills found in candidate’s applied work.

Idea:
By ignoring the self-listed skills section, we reduce the risk of "resume keyword stuffing."
---------------------------------------------------------------------------------------------------------------
5. Composite Score & Ranking
Apply custom weightages based on my hiring philosophy:
Experience  = 60%
Projects    = 35%
Skills      = 2.5%
Education   = 2.5%
Composite scores are computed only from available metrics (weights renormalized if any metric is missing).

Dense ranking assigns the same rank to tied candidates.
---------------------------------------------------------------------------------------------------------------
Advantages of Approach
1. Semantic Understanding - Uses transformer embeddings to match meaning, not just words.
2. Recency-Aware Experience Scoring - Rewards recent, relevant work.
3. Practical Skills Matching - Only considers applied skills.
4. Modular Design - Each scoring module is independent and reusable.
5. Custom Weightages - Reflects real-world recruiter priorities.
---------------------------------------------------------------------------------------------------------------
Disadvantages / Limitations of Approach
1. Model Bias - Embedding model accuracy depends on its training data; niche domains may have lower accuracy. So train a model of your own works best!
2. Experience Overlap — Current scoring treats experiences independently; doesn’t consider career progression.
3. Current mechanism doesn’t consider publication, volunteering exp.
4. Refining Job Description Quality before use as poorly written job descriptions can reduce scoring accuracy.
---------------------------------------------------------------------------------------------------------------
Possible Improvements
1. Weighted Recency Decay — Different decay rates for technical vs. soft skills in experiences.
2. Contextual Project Relevance — Weight projects higher if they match specific high-priority job responsibilities.
3. 1-1 job Responsibility search in exp/project/publication is can improve scoring over entire job_desc <-> full exp/proj/pub desc match.
---------------------------------------------------------------------------------------------------------------
Logics/Approaches:
Recency adjustment for experience relevance.
Combining project descriptions with skills to improve semantic richness.
Excluding self-reported skills from scoring to avoid keyword stuffing bias.
Renormalizing weights if some metrics are missing, avoiding unfair penalties.
Breaking down the pipeline into multiple analysis checkpoints (charts, graphs, heatmaps after each stage); used them to improve code logic and recalculate scores