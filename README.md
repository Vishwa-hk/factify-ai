# Factify AI

A self-correcting, multi-agent research assistant built with Streamlit and Groq. Two LLM agents
work against each other — a **Researcher** drafts an answer and a strict **Fact-checker** reviews it
— looping with corrective feedback until the draft passes, so the final report is verified rather
than taken on the model's first word.

## How it works

```
Query → Researcher (draft) → Fact-checker (VALID / REJECTED: reason)
                 ↑__________________ corrective feedback __________________|
          (loops up to 3 times, then outputs the best draft)
```

- **Researcher** — writes a clear, factual, well-structured answer; on a rejection it must directly
  address the fact-checker's feedback.
- **Fact-checker** — a strict editorial reviewer. The first pass is extra strict (rejects drafts
  lacking specific statistics, named examples or verifiable claims); later passes approve once the
  draft has meaningfully improved. Returns exactly `VALID` or `REJECTED: <reason>`.
- **Self-correction loop** — rejected drafts route back to the Researcher with the reason, up to
  `MAX_ITER` (3) iterations.
- **Live UI** — the Streamlit interface streams each agent's step in real time (a colour-coded
  pipeline log, intermediate drafts, verdicts) and saves past sessions so they can be revisited.

Model: `llama-3.3-70b-versatile` (via Groq).

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the app, enter your name and a **Groq API key** (free at
[console.groq.com](https://console.groq.com) — no credit card). The key is entered in the UI and is
never hardcoded or committed; optionally set a `GROQ_API_KEY` environment variable to pre-fill it.

## Notes

- Session history is stored locally in `history.json` (git-ignored — not committed).
- Built as a course project exploring agentic, human-in-the-loop-style LLM workflows.
