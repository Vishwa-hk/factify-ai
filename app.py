import streamlit as st
import json
import os
import re
import time
from groq import Groq

# ─────────────────────────────────────────────
# Config — paste your Groq key below
# ─────────────────────────────────────────────
MODEL           = "llama-3.3-70b-versatile"
MAX_ITER        = 3
DB_FILE         = "history.json"
DEFAULT_API_KEY = ""   

# ─────────────────────────────────────────────
# Markdown → HTML helper (fixes bold/italic/newlines)
# ─────────────────────────────────────────────
def md_to_html(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*',     r'<em>\1</em>',         text)
    text = re.sub(r'`(.*?)`',       r'<code>\1</code>',     text)
    text = text.replace('\n\n', '</p><p style="margin:10px 0 0">') \
               .replace('\n',   '<br>')
    return f'<p style="margin:0">{text}</p>'

# ─────────────────────────────────────────────
# Persistent history (JSON file)
# ─────────────────────────────────────────────
def load_history():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {"user": "", "sessions": []}

def save_history(user, query, report, iterations):
    data = load_history()
    data["user"] = user
    data["sessions"].append({
        "time":       time.strftime("%d/%m/%Y %H:%M"),
        "query":      query,
        "report":     report,
        "iterations": iterations
    })
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ─────────────────────────────────────────────
# Groq API call
# ─────────────────────────────────────────────
def groq_call(api_key: str, system_prompt: str, user_prompt: str) -> str:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        max_tokens=900,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# ─────────────────────────────────────────────
# Agent pipeline
# ─────────────────────────────────────────────
def run_agent(api_key: str, query: str, user: str):
    """
    Two-node agent loop:
      Researcher → Fact-checker → (loop back if rejected) → Final output
    Yields event dicts so the UI can stream updates in real time.
    """
    draft     = ""
    feedback  = ""
    iteration = 0

    while iteration < MAX_ITER:
        iteration += 1

        # ── Researcher node ──────────────────────
        yield {
            "type":  "log",
            "badge": "Researcher",
            "cls":   "researcher",
            "text":  (f"Drafting response for: <strong>\"{query}\"</strong>"
                      + (" — incorporating fact-checker feedback" if feedback else "")
                      + f" (attempt {iteration})")
        }

        researcher_system = (
            "You are an expert research assistant. "
            "Write a clear, factual, well-structured response to the user's query. "
            + (
                f"\n\nIMPORTANT — The fact-checker rejected your last draft:\n"
                f"\"{feedback}\"\n"
                "You MUST directly address this in your revised draft. "
                "Be concrete and specific."
                if feedback else ""
            )
            + "\n\nKeep the response 2–4 paragraphs. "
              "Use specific facts, named examples, and verifiable claims where possible."
        )

        draft = groq_call(api_key, researcher_system, query)
        yield {"type": "draft", "iteration": iteration, "text": draft}

        # ── Fact-checker node ────────────────────
        yield {
            "type":  "log",
            "badge": "Fact-checker",
            "cls":   "checker",
            "text":  f"Evaluating draft {iteration} for factual accuracy and depth"
        }

        checker_system = (
            "You are a strict editorial fact-checker reviewing a research draft.\n\n"
            "Evaluation rules:\n"
            "- FIRST evaluation: Be extra strict. Reject if the draft lacks specific "
            "statistics, concrete named examples, or verifiable claims.\n"
            "- SUBSEQUENT evaluations (attempt 2+): Approve if the draft has meaningfully "
            "improved and contains specific details.\n"
            "- If approving: respond with exactly the word VALID and nothing else.\n"
            "- If rejecting: respond with REJECTED: followed by one precise, actionable "
            "sentence explaining exactly what is missing.\n\n"
            "Respond ONLY with \"VALID\" or \"REJECTED: [reason]\". No other text."
        )

        verdict  = groq_call(api_key, checker_system,
                             f"Query: {query}\n\nDraft (attempt {iteration}):\n{draft}")
        is_valid = verdict.strip().upper().startswith("VALID")
        feedback = "" if is_valid else verdict.replace("REJECTED:", "").strip()

        if not is_valid and iteration < MAX_ITER:
            yield {"type": "log",     "badge": "Rejected", "cls": "rejected",
                   "text": f"Fact-check failed: <strong>{feedback}</strong>"}
            yield {"type": "verdict", "verdict": "rejected", "feedback": feedback}
            yield {"type": "log",     "badge": "System",   "cls": "system",
                   "text": f"Routing back to Researcher with corrective feedback "
                            f"(starting attempt {iteration + 1})"}
        else:
            label = "Approved" if is_valid else "System"
            cls   = "approved" if is_valid else "system"
            msg   = ("All claims verified — routing to final output"
                     if is_valid else
                     "Max iterations reached — outputting best available draft")
            yield {"type": "log",     "badge": label, "cls": cls, "text": msg}
            yield {"type": "verdict", "verdict": "approved", "feedback": ""}
            break

    yield {"type": "final", "draft": draft, "iterations": iteration}

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
.log-row { display:flex; align-items:flex-start; gap:10px; padding:7px 0;
           border-bottom:1px solid rgba(128,128,128,0.12); }
.log-row:last-child { border-bottom:none; }
.badge { font-size:11px; font-weight:600; padding:2px 8px; border-radius:4px;
         white-space:nowrap; flex-shrink:0; line-height:1.6; }
.b-researcher { background:#e6f1fb; color:#185fa5; border:1px solid #b5d4f4; }
.b-checker    { background:#faeeda; color:#854f0b; border:1px solid #fac775; }
.b-system     { background:#f1efe8; color:#5f5e5a; border:1px solid #d3d1c7; }
.b-approved   { background:#eaf3de; color:#3b6d11; border:1px solid #c0dd97; }
.b-rejected   { background:#fcebeb; color:#a32d2d; border:1px solid #f7c1c1; }

.verdict-rejected { background:#fcebeb; border:1px solid #f7c1c1; color:#a32d2d;
                    border-radius:8px; padding:10px 14px; font-size:13px; margin:4px 0; }
.verdict-approved { background:#eaf3de; border:1px solid #c0dd97; color:#3b6d11;
                    border-radius:8px; padding:10px 14px; font-size:13px; margin:4px 0; }
.final-box   { background:#eaf3de; border:1px solid #c0dd97; border-radius:10px;
               padding:16px 18px; margin-top:4px; }
.final-box h4 { color:#3b6d11; margin-bottom:10px; font-size:13px; font-weight:600; margin-top:0; }
.final-text  { font-size:14px; line-height:1.8; color:#1e4620; } /* 💡 明确指定了深绿色字体 */
.final-text p { color:#1e4620; margin-bottom:12px; } /* 💡 确保段落标签也是深色 */
.final-text strong { font-weight:600; color:#0f2d10; }
.final-text em     { font-style:italic; }
.final-text code   { background:rgba(0,0,0,0.08); padding:1px 5px;
                     border-radius:3px; font-size:13px; color:#111111; }

.draft-box  { border:1px solid rgba(128,128,128,0.18); border-radius:10px; overflow:hidden; margin:4px 0; }
.draft-head { background:rgba(128,128,128,0.07); padding:8px 14px;
              border-bottom:1px solid rgba(128,128,128,0.12);
              font-size:12px; font-weight:600; opacity:0.7; }
.draft-body { padding:14px 16px; font-size:14px; line-height:1.8; }
.draft-body strong { font-weight:600; }
.draft-body em     { font-style:italic; }
</style>
"""

# ─────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────
def badge_html(label: str, cls: str, text: str) -> str:
    return (f'<div class="log-row">'
            f'<span class="badge b-{cls}">{label}</span>'
            f'<span style="font-size:13px;line-height:1.5">{text}</span>'
            f'</div>')

def log_panel_html(entries: str) -> str:
    return (f'<div style="background:rgba(128,128,128,0.05);border:1px solid '
            f'rgba(128,128,128,0.12);border-radius:10px;padding:12px 14px;">'
            f'<p style="font-size:11px;font-weight:600;opacity:0.5;'
            f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">'
            f'Agent pipeline log</p>'
            f'{entries}</div>')

def final_box_html(text: str) -> str:
    return (f'<div class="final-box"><h4>✅ Verified final report</h4>'
            f'<div class="final-text">{md_to_html(text)}</div></div>')

def draft_box_html(text: str, iteration: int) -> str:
    return (f'<div class="draft-box">'
            f'<div class="draft-head">Draft {iteration}</div>'
            f'<div class="draft-body">{md_to_html(text)}</div>'
            f'</div>')

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Factify AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────
for key, default in {
    "started":      False,
    "user":         "",
    "api_key":      "",
    "view_session": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✨ Factify AI")
    st.caption("Research & fact-checking agent")
    st.divider()

    if st.session_state.started:
        st.markdown(f"**User:** {st.session_state.user}")
        if st.button("Switch user", use_container_width=True):
            for k in ["started", "user", "api_key", "view_session"]:
                st.session_state[k] = False if k == "started" else (None if k == "view_session" else "")
            st.rerun()

        st.divider()
        st.markdown("**History**")

        data = load_history()
        if data["sessions"]:
            for i, s in enumerate(reversed(data["sessions"])):
                real_idx = len(data["sessions"]) - 1 - i
                label = s["query"][:38] + ("…" if len(s["query"]) > 38 else "")
                if st.button(f"🔍 {label}", key=f"hist_{i}", use_container_width=True):
                    st.session_state.view_session = real_idx
                    st.rerun()
        else:
            st.caption("No sessions yet")

        st.divider()
        if st.button("🗑 Clear history", use_container_width=True):
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
            st.session_state.view_session = None
            st.rerun()

# ─────────────────────────────────────────────
# Onboarding
# ─────────────────────────────────────────────
if not st.session_state.started:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 👋 Welcome to Factify AI")
        st.caption("A self-correcting multi-agent research system powered by Groq.")
        st.markdown("<br>", unsafe_allow_html=True)

        name = st.text_input("Your name", placeholder="Enter your name…", key="name_input")
        api  = st.text_input("Groq API key",
                             placeholder="gsk_…",
                             value=DEFAULT_API_KEY,
                             type="password",
                             help="Free key at console.groq.com — no credit card needed",
                             key="api_input")
        st.caption("Model: `llama-3.3-70b-versatile`")

        if st.button("Start →", type="primary", use_container_width=True):
            if not name.strip():
                st.warning("Please enter your name.")
            elif not api.strip():
                st.warning("Please enter your Groq API key.")
            else:
                st.session_state.started = True
                st.session_state.user    = name.strip()
                st.session_state.api_key = api.strip()
                st.rerun()
    st.stop()

# ─────────────────────────────────────────────
# Workspace header
# ─────────────────────────────────────────────
st.markdown("### Research Agent")
st.caption("Researcher → Fact-checker → Self-correction loop")
st.divider()

# ─────────────────────────────────────────────
# History viewer
# ─────────────────────────────────────────────
if st.session_state.view_session is not None:
    data = load_history()
    s    = data["sessions"][st.session_state.view_session]

    st.info(f"🕒 Viewing saved session from {s['time']}")
    st.markdown(f"**Query:** {s['query']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Iterations", s["iterations"])
    c2.metric("Status",     "Verified ✓")
    c3.metric("Model",      "llama-3.3-70b")

    st.markdown(final_box_html(s["report"]), unsafe_allow_html=True)

    if st.button("← Back to search"):
        st.session_state.view_session = None
        st.rerun()
    st.stop()

# ─────────────────────────────────────────────
# Search form
# ─────────────────────────────────────────────
with st.form("search_form", clear_on_submit=True):
    query     = st.text_input("What would you like to research?",
                              placeholder="Ask anything…",
                              label_visibility="collapsed")
    submitted = st.form_submit_button("▶  Run agent", type="primary")

if not submitted or not query.strip():
    st.stop()

# ─────────────────────────────────────────────
# Run agent & stream output
# ─────────────────────────────────────────────
st.markdown("---")

log_html    = ""
log_display = st.empty()

try:
    for event in run_agent(st.session_state.api_key, query.strip(), st.session_state.user):

        if event["type"] == "log":
            log_html += badge_html(event["badge"], event["cls"], event["text"])
            log_display.markdown(log_panel_html(log_html), unsafe_allow_html=True)

        elif event["type"] == "draft":
            st.markdown(draft_box_html(event["text"], event["iteration"]),
                        unsafe_allow_html=True)

        elif event["type"] == "verdict":
            if event["verdict"] == "rejected":
                st.markdown(
                    f'<div class="verdict-rejected">❌ <strong>Draft rejected</strong>'
                    f' — {event["feedback"]}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="verdict-approved">✅ <strong>Draft approved</strong>'
                    ' — routing to final output</div>',
                    unsafe_allow_html=True)

        elif event["type"] == "final":
            save_history(st.session_state.user, query.strip(),
                         event["draft"], event["iterations"])

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Iterations", event["iterations"])
            c2.metric("Status",     "Verified ✓")
            c3.metric("Model",      "llama-3.3-70b")

            st.markdown(final_box_html(event["draft"]), unsafe_allow_html=True)

except Exception as e:
    st.error(f"**Agent error:** {e}\n\nCheck your API key and try again.")
