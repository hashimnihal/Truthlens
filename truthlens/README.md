# TruthLens

> **Online News Credibility & Clickbait Classifier**  
> BAI507P Mini Project · CSE (AI & ML) · Sahyadri College of Engineering and Management · AY 2026-27

**Team:** Hashim Nihal (leader) · Pawan P Bangera · Samuel Joy Fernandes · Mohammed Ashqar Sameer  
**Guide:** Mrs. Shruthi Vishwajeeth

---

## What It Does

TruthLens accepts a news headline, article body, URL and/or an image, then returns:

- A **credibility score** (0–100) and a verdict (`Likely Authentic`, `Mixed`, `Likely False`, `Insufficient Evidence`)
- A **clickbait score** (scikit-learn TF-IDF baseline blended with Gemini's rating)
- **Source citations** from live DuckDuckGo News retrieval, each labelled as *supports*, *contradicts* or *unrelated*
- An **image check** (text–image fit and manipulation risk indicators)
- A **stress test** (Deep mode): counterfactual variants that expose confirmation bias

---

## Setup — under 10 steps

### 1. Prerequisites
- Python 3.11 or later
- A Gemini API key ([get one free](https://aistudio.google.com/))

### 2. Clone and enter the project

```bash
git clone <repo-url>
cd truthlens
```

### 3. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment

```bash
cp .env.example .env
# Open .env and set GEMINI_API_KEY=<your key>
```

### 6. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

### 7. (Optional) Run in MOCK mode — no API quota needed

```bash
# Windows PowerShell
$env:MOCK="1"; streamlit run app.py

# macOS / Linux
MOCK=1 streamlit run app.py
```

### 8. Run the tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
truthlens/
├── app.py                    # Analyse page (home)
├── pages/
│   ├── 1_History.py
│   ├── 2_Evaluation.py
│   └── 3_About.py
├── core/                     # All engine logic — no UI here
│   ├── models.py             # Pydantic v2 schemas
│   ├── pipeline.py           # run_check(request, mode) → AnalysisResult
│   ├── gemini_client.py      # Gemini 2.5 Flash: retry, JSON schema, mock
│   ├── extract.py            # URL scraping & image handling (trafilatura)
│   ├── claims.py             # Claim extraction & text analysis
│   ├── retrieval.py          # DuckDuckGo News + source trust tiers
│   ├── evidence.py           # Stance judgement per source
│   ├── clickbait.py          # sklearn TF-IDF baseline + Gemini blend
│   ├── image_check.py        # Gemini Vision multimodal check
│   ├── stress.py             # Counterfactual stress test (Deep mode)
│   ├── scoring.py            # Credibility formula from PRD §7.2
│   └── storage.py            # SQLite: analyses, cache, feedback
├── ui/
│   ├── components.py         # Reusable Streamlit components
│   └── styles.py             # Custom CSS injected via st.markdown
├── data/
│   ├── trusted_sources.csv   # Curated source tier list
│   └── eval/                 # Evaluation scripts and result tables
├── models/
│   └── clickbait_lr.joblib   # Trained sklearn model
├── extension/                # Manifest V3 browser extension
├── tests/                    # pytest unit tests
├── .streamlit/config.toml    # Theme colours from PRD §9.5
├── .env.example              # Template — copy to .env
└── requirements.txt
```

---

## Architecture Rule

> **Pages call only `pipeline.run_check`.** No page imports from `gemini_client`, `retrieval` or any other engine module directly. This keeps the engine testable and lets evaluation scripts reuse it.

---

## Scoring Formula (PRD §7.2)

```
credibility = 100 × (0.45·E  +  0.20·(1-H)  +  0.15·(1-V)  +  0.10·(1-C)  +  0.10·R)

E  = evidence support score (tier-weighted, 0–1)
H  = headline–body incongruity (Gemini, 0–1)
V  = image risk (Gemini Vision, 0–1)   ← dropped & rescaled when no image
C  = clickbait score (0–1)
R  = stress-test robustness (0–1)      ← dropped & rescaled in Fast mode
```

---

## Gemini Call Budget

| Mode | Max calls | Notes |
|------|-----------|-------|
| Fast | 3 | claim extraction, evidence judgement, image check |
| Deep | 5 | + altered claim generation, judgement of altered claims |
| Cached | 0 | Full cache hit returns stored result |

---

## Ethics & Limits

- Results are **evidence-based assessments**, never final rulings.
- Image checks return **risk indicators only** — not proof of manipulation.
- Sources and reasoning are always shown so users can disagree with the verdict.
- English only; no video or audio; no user accounts in v1.

---

## Licence

MIT — see `LICENSE` for details.

