"""
app.py - Streamlit interactive demo for the Scholarship Eligibility Checker.
Run with: streamlit run demo/app.py

UI redesign: presentation only. The RAG pipeline, retrieval, prompt,
profile building and verdict logic are unchanged.
"""

import html
import os
import re
import sys

# Add parent directory to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
from chunking import load_and_chunk_all
from rag_pipeline import ScholarshipRAG, extract_verdict

# Page config
st.set_page_config(
    page_title="Scholarship Eligibility Checker",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --navy: #0B2545;
    --navy-2: #13315C;
    --blue: #1D4E89;
    --blue-soft: #EAF0F8;
    --bg: #F6F7F9;
    --card: #FFFFFF;
    --border: #E3E7ED;
    --text: #1B2430;
    --muted: #5E6B7A;
    --muted-2: #8A96A3;
    --green: #1E7B4F;   --green-bg: #EDF7F1;  --green-border: #CBE7D6;
    --red: #A63D40;     --red-bg: #FBF0F0;    --red-border: #F0D3D4;
    --amber: #8A5A00;   --amber-bg: #FDF6E7;  --amber-border: #F1DFB5;
    --radius: 14px;
    --shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 2px 8px rgba(16, 24, 40, 0.04);
}

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.stApp { background: var(--bg); color: var(--text); }

/* Streamlit top toolbar (Deploy / menu): keep but make unobtrusive */
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { opacity: 0.55; transition: opacity .2s ease; }
[data-testid="stToolbar"]:hover { opacity: 1; }

/* Centered content container */
.block-container { max-width: 1200px; padding-top: 3rem; padding-bottom: 4rem; }

/* Headings */
h1, h2, h3, h4 { color: var(--navy); letter-spacing: -0.01em; }

/* ---------- Header ---------- */
.app-header { display: flex; align-items: center; gap: 18px; margin: 0 0 40px 0; }
.app-header .logo {
    width: 52px; height: 52px; border-radius: 14px; flex-shrink: 0;
    background: var(--navy); display: flex; align-items: center; justify-content: center;
    box-shadow: var(--shadow);
}
.app-header h1 {
    font-size: 1.65rem; font-weight: 700; margin: 0; padding: 0; line-height: 1.25; color: var(--navy);
}
.app-header p { margin: 4px 0 0 0; color: var(--muted); font-size: 0.97rem; }

/* ---------- Cards (bordered containers) ----------
   Newer Streamlit puts the border on stVerticalBlock, older versions on
   stVerticalBlockBorderWrapper; both are matched by "contains a .card-title". */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-title),
[data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] .card-title) {
    background: var(--card);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow);
    padding: 26px 28px !important;
    gap: 1.1rem;
    box-sizing: border-box !important;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] .card-title) > [data-testid="element-container"],
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-title) > [data-testid="stElementContainer"] {
    max-width: 100%;
}
/* Older Streamlit fixes markdown width inline; let it follow the card's padding */
[data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] .card-title) [data-testid="stMarkdown"],
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-title) [data-testid="stMarkdown"],
[data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] .card-title) > [data-testid="element-container"] > div {
    width: 100% !important;
}
/* Older Streamlit: its own bordered wrapper is neutralised so the card is drawn once */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .card-title) {
    border: none !important; padding: 0 !important; background: transparent !important; box-shadow: none !important;
}
.card-title { display: flex; align-items: center; justify-content: space-between; margin: 2px 0 2px 0; }
.card-title h3 { font-size: 1.08rem; font-weight: 600; margin: 0; padding: 0; color: var(--navy); }
.card-sub { color: var(--muted); font-size: 0.88rem; margin: 6px 0 16px 0; line-height: 1.6; }
.pill {
    font-size: 0.72rem; font-weight: 600; letter-spacing: .03em; text-transform: uppercase;
    padding: 3px 9px; border-radius: 999px; border: 1px solid var(--border); color: var(--muted);
    background: #FAFBFC;
}
.pill.req { color: var(--blue); border-color: #C9D7EA; background: var(--blue-soft); }

/* ---------- Inputs ---------- */
[data-testid="stWidgetLabel"] p { font-size: 0.86rem !important; font-weight: 500; color: var(--text); }
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, .stTextArea textarea {
    border-radius: 10px !important;
    border-color: var(--border) !important;
    background: #FFFFFF !important;
    transition: border-color .15s ease, box-shadow .15s ease;
}
div[data-baseweb="select"] > div:focus-within, .stTextInput input:focus,
.stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(29, 78, 137, 0.12) !important;
}
.stTextArea textarea { font-size: 0.98rem; line-height: 1.6; padding: 14px 16px; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted-2) !important; }

/* Expanders -> clean accordions */
[data-testid="stExpander"] details {
    border: 1px solid var(--border) !important; border-radius: 12px !important; background: #FCFCFD;
}
[data-testid="stExpander"] summary { padding: 14px 18px; font-weight: 500; color: var(--navy); }
[data-testid="stExpander"] summary:hover { color: var(--blue); }
[data-testid="stExpander"] summary p { font-size: 0.93rem; }

/* Primary button */
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: var(--navy) !important; border: 1px solid var(--navy) !important; color: #fff !important;
    border-radius: 10px !important; padding: 0.62rem 1.2rem !important; font-weight: 600 !important;
    letter-spacing: .01em; box-shadow: 0 1px 2px rgba(11, 37, 69, 0.2);
    transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background: var(--navy-2) !important; transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(11, 37, 69, 0.18);
}
.stButton > button[kind="primary"]:active, button[data-testid="stBaseButton-primary"]:active {
    transform: translateY(0);
}

/* Slider accent */
[data-testid="stSlider"] [role="slider"] { background: var(--navy) !important; box-shadow: none !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarContent"] { padding-top: 1.2rem; }
.side-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
    color: var(--muted-2); margin: 6px 0 10px 0; display: flex; justify-content: space-between; align-items: center;
}
.src-count {
    text-transform: none; letter-spacing: 0; font-weight: 500; color: var(--blue);
    background: var(--blue-soft); border: 1px solid #C9D7EA;
    padding: 2px 8px; border-radius: 999px; font-size: 0.72rem;
}
.kb-item { display: flex; gap: 12px; padding: 14px 4px; border-bottom: 1px solid #F0F2F5; }
.kb-item:last-child { border-bottom: none; }
.kb-icon {
    width: 32px; height: 32px; border-radius: 8px; background: var(--blue-soft); flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
}
.kb-tag { font-size: 0.7rem; font-weight: 600; color: var(--blue); letter-spacing: .03em; }
.kb-name { font-size: 0.86rem; font-weight: 600; color: var(--text); line-height: 1.35; }
.kb-desc { font-size: 0.78rem; color: var(--muted); line-height: 1.45; margin-top: 2px; }
.about-card {
    border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; background: #FAFBFC;
    font-size: 0.8rem; color: var(--muted); line-height: 1.7;
}
.about-card b { color: var(--text); font-weight: 600; }
.note { font-size: 0.76rem; color: var(--muted-2); line-height: 1.55; margin-top: 12px; }

/* ---------- Result ---------- */
.status {
    display: flex; gap: 16px; align-items: center; padding: 22px 24px; border-radius: 12px; border: 1px solid;
    margin: 8px 0 26px 0;
}
.status .ico { width: 42px; height: 42px; border-radius: 50%; display: flex; align-items: center;
    justify-content: center; flex-shrink: 0; }
.status .t { font-size: 1.3rem; font-weight: 700; line-height: 1.2; }
.status .s { font-size: 0.85rem; margin-top: 3px; opacity: .9; }
.status.ok { background: var(--green-bg); border-color: var(--green-border); color: var(--green); }
.status.ok .ico { background: #D6EFE0; }
.status.no { background: var(--red-bg); border-color: var(--red-border); color: var(--red); }
.status.no .ico { background: #F5DEDF; }
.status.maybe { background: var(--amber-bg); border-color: var(--amber-border); color: var(--amber); }
.status.maybe .ico { background: #F8EACB; }

.sec-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
    color: var(--muted-2); margin: 28px 0 10px 0;
}
.prose { font-size: 0.95rem; line-height: 1.72; color: var(--text); }
.prose p { margin: 0 0 14px 0; }
.prose p.callout { margin: 14px 0; padding: 12px 16px; }
.prose ul { margin: 0 0 6px 0; padding-left: 20px; }
.prose li { margin-bottom: 4px; }
.callout {
    border-left: 3px solid var(--blue); background: var(--blue-soft); border-radius: 0 10px 10px 0;
    padding: 10px 14px; margin: 8px 0; font-size: 0.9rem; line-height: 1.6; color: var(--navy);
}
.citation {
    border: 1px solid var(--border); border-radius: 12px; background: #FAFBFC;
    padding: 14px 16px; font-size: 0.92rem; line-height: 1.65; color: var(--text);
    display: flex; gap: 12px;
}
.citation .q { font-size: 1.6rem; line-height: 1; color: var(--blue); font-family: Georgia, serif; }
.profile-used { font-size: 0.8rem; color: var(--muted); margin: -8px 0 6px 0; }
.profile-used b { font-weight: 600; color: var(--text); }

/* Source cards */
.src-card {
    border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; margin: 0 0 14px 0;
    background: #FFFFFF; transition: border-color .15s ease;
}
.src-card:hover { border-color: #C9D7EA; }
.src-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 6px; }
.src-doc { font-size: 0.84rem; font-weight: 600; color: var(--navy); display: flex; gap: 8px; align-items: center; }
.src-rank { font-size: 0.7rem; font-weight: 600; color: var(--blue); background: var(--blue-soft);
    padding: 2px 8px; border-radius: 999px; white-space: nowrap; }
.src-sec { font-size: 0.76rem; color: var(--muted); margin-bottom: 6px; }
.src-text { font-size: 0.84rem; line-height: 1.6; color: #3A4552; }
.src-ref { font-size: 0.72rem; color: var(--muted-2); margin-top: 8px; font-family: ui-monospace, Menlo, monospace; }

/* Empty state */
.empty { text-align: center; padding: 36px 10px 14px 10px; }
.empty .ico { width: 56px; height: 56px; border-radius: 16px; background: var(--blue-soft);
    display: inline-flex; align-items: center; justify-content: center; margin-bottom: 12px; }
.empty h4 { font-size: 1.02rem; font-weight: 600; margin: 0 0 4px 0; color: var(--navy); }
.empty p { font-size: 0.88rem; color: var(--muted); margin: 0; }
.ex { border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; margin: 10px 0;
    font-size: 0.87rem; color: #3A4552; background: #FCFCFD; line-height: 1.5; }

.footer { text-align: center; color: var(--muted-2); font-size: 0.8rem; margin-top: 56px; }
.footer a { color: var(--muted); }

/* ---------- Extra whitespace ---------- */
[data-testid="stMain"] [data-testid="stColumn"] > div > [data-testid="stVerticalBlock"],
[data-testid="stMain"] [data-testid="column"] > div > [data-testid="stVerticalBlock"],
[data-testid="stMain"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { gap: 1.75rem; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 6px 20px 20px 20px; }
[data-testid="stWidgetLabel"] { margin-bottom: 4px; }
[data-testid="stSidebarContent"] { padding-left: 8px; padding-right: 8px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Icons (inline SVG, stroke style)
# ----------------------------------------------------------------------
def icon(name, color="#1D4E89", size=18):
    paths = {
        "cap": '<path d="M22 10 12 5 2 10l10 5 10-5z"/><path d="M6 12v5c3 2.5 9 2.5 12 0v-5"/>',
        "doc": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h4"/>',
        "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
        "x": '<path d="M7 7l10 10M17 7 7 17"/>',
        "alert": '<path d="M12 8v5"/><path d="M12 16.5v.5"/><circle cx="12" cy="12" r="9"/>',
        "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    }
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{paths[name]}</svg>')


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(f"""
<div class="app-header">
  <div class="logo">{icon("cap", "#FFFFFF", 26)}</div>
  <div>
    <h1>Scholarship &amp; Grant Eligibility Checker</h1>
    <p>AI-powered eligibility guidance based on official scholarship and grant documents</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div class="side-label">Knowledge Base <span class="src-count">3 official sources</span></div>
    <div class="kb-item"><div class="kb-icon">{icon("doc", size=16)}</div><div>
        <div class="kb-tag">KB-1</div>
        <div class="kb-name">RMIT Research Scholarship Terms &amp; Conditions</div>
        <div class="kb-desc">Student-facing rules: stipends, part-time, leave, duration</div></div></div>
    <div class="kb-item"><div class="kb-icon">{icon("doc", size=16)}</div><div>
        <div class="kb-tag">KB-2</div>
        <div class="kb-name">RMIT HDR and RTP Scholarships Policy</div>
        <div class="kb-desc">Administrative rules: concurrent income, transfers, termination</div></div></div>
    <div class="kb-item"><div class="kb-icon">{icon("doc", size=16)}</div><div>
        <div class="kb-tag">KB-3</div>
        <div class="kb-name">ARC Discovery Program Grant Guidelines</div>
        <div class="kb-desc">Grant eligibility: DECRA, Future Fellowships, project limits</div></div></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-label" style="margin-top:22px">Settings</div>', unsafe_allow_html=True)
    with st.expander("Advanced settings", expanded=False):
        retrieval_mode = st.radio(
            "Retrieval method", ["dense", "hybrid"], horizontal=True,
            format_func=lambda m: "Dense (embeddings)" if m == "dense" else "Hybrid (dense + BM25)",
            help="Dense finds text with the same meaning. Hybrid also matches exact keywords.")
        llm_model = st.selectbox(
            "Language model",
            ["qwen2.5:7b", "llama3.1", "llama3"],
            help="Must be downloaded first, e.g. `ollama pull qwen2.5:7b`. qwen2.5:7b handles conditions and exceptions best.")
        n_results = st.slider("Retrieved chunks", min_value=3, max_value=10, value=6,
                              help="Number of document chunks to retrieve for each query")
        kb_choice = st.selectbox(
            "Search in",
            ["All documents", "KB-1: RMIT Scholarship Terms", "KB-2: RMIT RTP Policy", "KB-3: ARC Discovery (DECRA / Future Fellowship)"],
            help="Restricting the search to the right document improves accuracy")

    st.markdown('<div class="side-label" style="margin-top:22px">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card">
        <b>Built for:</b> RMIT WIL Project<br/>
        <b>Methodology:</b> Test-Driven RAG<br/>
        <b>Evaluation:</b> 30 gold-standard test cases<br/>
        <b>LLM:</b> Local via Ollama (no API costs)
    </div>
    <div class="note">This is a prototype for educational purposes. Always verify with official university advice.</div>
    """, unsafe_allow_html=True)

# Initialize RAG (cached)
@st.cache_resource(show_spinner="Loading knowledge base and embedding documents...")
def get_rag_pipeline(model_name, mode):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks = load_and_chunk_all(
        os.path.join(base_dir, "data/raw/kb1_rmit_scholarship_terms.txt"),
        os.path.join(base_dir, "data/raw/kb2_rmit_rtp_policy.txt"),
        os.path.join(base_dir, "data/raw/kb3_arc_discovery_guidelines.txt")
    )
    return ScholarshipRAG(chunks, llm_model=model_name, retrieval_mode=mode,
                          collection_name=f"scholarship_kb_{mode}")

# Load RAG
try:
    rag = get_rag_pipeline(llm_model, retrieval_mode)
    rag_ready = True
except Exception as e:
    st.error(f"Failed to initialize RAG pipeline: {e}")
    st.error("Make sure Ollama is running and the model is pulled. Run: `ollama pull qwen2.5:7b`")
    rag_ready = False


# ----------------------------------------------------------------------
# Display helpers (presentation only - they never change the verdict)
# ----------------------------------------------------------------------
SECTION_KEYS = ["FACTS", "REASONING", "ANALYSIS", "EVIDENCE", "CITATION", "VERDICT"]
CONDITION_WORDS = re.compile(r"\b(unless|except|provided that|only if|does not apply|not eligible|must not|exceed|limit)\b", re.I)


def split_sections(text):
    """Split the model's response into labelled sections for display."""
    pattern = re.compile(r"^\s*[#*\s]*(" + "|".join(SECTION_KEYS) + r")\s*[*]*\s*:\s*[*]*", re.I | re.M)
    matches = list(pattern.finditer(text))
    if not matches:
        return {}
    sections = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        key = m.group(1).upper()
        body = text[m.end():end].strip()
        sections[key] = (sections.get(key, "") + "\n" + body).strip()
    return sections


def prose_html(text, highlight=False):
    """Escape model text and turn lines / bullets into readable paragraphs."""
    lines = [l.strip() for l in text.strip().splitlines()]
    out, bullets = [], []
    for line in lines + [""]:
        is_bullet = bool(re.match(r"^([-*•]|\d+[.)])\s+", line))
        if is_bullet:
            bullets.append(html.escape(re.sub(r"^([-*•]|\d+[.)])\s+", "", line)))
            continue
        if bullets:
            out.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
            bullets = []
        if line:
            # Highlight sentences that state a condition or exception
            tag = 'p class="callout"' if highlight and CONDITION_WORDS.search(line) else "p"
            out.append(f"<{tag}>{html.escape(line)}</p>")
    body = "".join(out)
    body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)
    return f'<div class="prose">{body}</div>'


STATUS = {
    "ELIGIBLE": ("ok", "check", "#1E7B4F", "Eligible",
                 "Based on the provided profile and official documents."),
    "INELIGIBLE": ("no", "x", "#A63D40", "Not eligible",
                   "Based on the provided profile and official documents."),
    "CANNOT_DETERMINE": ("maybe", "alert", "#8A5A00", "Needs review",
                         "The documents don't give a definitive answer for this profile."),
}


# ----------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------
col1, col2 = st.columns([1, 1.25], gap="large")

with col1:
    with st.container(border=True):
        st.markdown("""
        <div class="card-title"><h3>Your Profile</h3><span class="pill">Optional</span></div>
        <div class="card-sub">Adding your details helps the checker apply the right conditions and exceptions.</div>
        """, unsafe_allow_html=True)

        with st.expander("Add profile details", expanded=False):
            citizenship = st.selectbox(
                "Citizenship",
                ["Select...", "Domestic (Australian/NZ)", "International"],
                index=0
            )
            c_a, c_b = st.columns(2)
            with c_a:
                degree = st.selectbox(
                    "Degree Type",
                    ["Select...", "PhD", "Masters by Research", "Not yet enrolled"],
                    index=0
                )
            with c_b:
                enrollment = st.selectbox(
                    "Enrollment Status",
                    ["Select...", "Full-time", "Part-time", "Not sure"],
                    index=0
                )
            prior_scholarship = st.checkbox("Previously held a Commonwealth research scholarship?")
            c_c, c_d = st.columns(2)
            with c_c:
                prior_decra_apps = st.number_input("Prior DECRA/FF nominations", min_value=0, max_value=5, value=0)
            with c_d:
                phd_date = st.text_input("PhD award date (if applicable)", placeholder="e.g., March 2021")
            st.caption("Nominations and PhD date are only needed for ARC grants (DECRA / Future Fellowship).")
            extra_profile = st.text_input("Anything else relevant?", placeholder="e.g., took 2 years parental leave")

    with st.container(border=True):
        st.markdown("""
        <div class="card-title"><h3>Your Question</h3><span class="pill req">Required</span></div>
        <div class="card-sub">Describe your situation and what you'd like to know, in plain language.</div>
        """, unsafe_allow_html=True)

        query = st.text_area(
            "Your Question",
            height=140,
            placeholder="e.g., I am a single parent with a 3-year-old. Can I get a part-time RTP stipend?",
            label_visibility="collapsed",
        )
        st.caption("You can ask about stipends, part-time study, income limits, transfers, DECRA or Future Fellowships.")

        check_button = st.button("Check Eligibility", type="primary", use_container_width=True)

with col2:
    with st.container(border=True):
        st.markdown('<div class="card-title"><h3>Eligibility Result</h3></div>', unsafe_allow_html=True)

        if check_button and rag_ready:
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving relevant policy clauses and analysing..."):
                    try:
                        # Build the student profile from the form so the LLM can use it
                        parts = []
                        if citizenship != "Select...":
                            parts.append(f"Citizenship: {citizenship}")
                        if degree != "Select...":
                            parts.append(f"Degree: {degree}")
                        if enrollment != "Select...":
                            parts.append(f"Enrolment: {enrollment}")
                        if prior_scholarship:
                            parts.append("Has previously held a Commonwealth research scholarship")
                        if prior_decra_apps:
                            parts.append(f"Previous DECRA/Future Fellowship nominations: {prior_decra_apps}")
                        if phd_date.strip():
                            parts.append(f"PhD awarded: {phd_date.strip()}")
                        if extra_profile.strip():
                            parts.append(extra_profile.strip())
                        profile = "; ".join(parts)
                        source_filter = None if kb_choice == "All documents" else kb_choice.split(":")[0]

                        result = rag.answer_with_sources(query, n_results=n_results, profile=profile,
                                                         source_filter=source_filter)
                        response = result['response']
                        sources = result['sources']

                        # Verdict (same parser as the evaluation)
                        verdict = result['verdict']
                        cls, ico, col, title, sub = STATUS.get(
                            verdict, ("maybe", "alert", "#8A5A00", "Unclear verdict",
                                      "The response could not be classified. Please read the details below."))

                        # Verdict banner
                        st.markdown(f"""
                        <div class="status {cls}">
                          <div class="ico">{icon(ico, col, 22)}</div>
                          <div><div class="t">{title}</div><div class="s">{sub}</div></div>
                        </div>""", unsafe_allow_html=True)

                        if profile:
                            st.markdown(f'<div class="profile-used"><b>Profile used:</b> {html.escape(profile)}</div>',
                                        unsafe_allow_html=True)

                        # Detailed response, split into readable sections
                        sections = split_sections(response)
                        if sections:
                            reasoning = sections.get("ANALYSIS") or sections.get("REASONING", "")
                            if sections.get("FACTS"):
                                st.markdown('<div class="sec-label">Your situation</div>', unsafe_allow_html=True)
                                st.markdown(prose_html(sections["FACTS"]), unsafe_allow_html=True)
                            if reasoning:
                                st.markdown('<div class="sec-label">Reasoning</div>', unsafe_allow_html=True)
                                st.markdown(prose_html(reasoning, highlight=True), unsafe_allow_html=True)
                            if sections.get("EVIDENCE"):
                                st.markdown('<div class="sec-label">Evidence</div>', unsafe_allow_html=True)
                                st.markdown(prose_html(sections["EVIDENCE"]), unsafe_allow_html=True)
                            citation = sections.get("CITATION", "").strip()
                            if citation and citation.strip('"\' .').lower() in ("none", "n/a", "not applicable", ""):
                                st.markdown('<div class="sec-label">Citation</div>'
                                            '<div class="card-sub">The model did not cite a specific clause for this answer. '
                                            'Check the retrieved sources below, or rephrase the question.</div>',
                                            unsafe_allow_html=True)
                            elif citation:
                                st.markdown('<div class="sec-label">Citation</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="citation"><span class="q">&ldquo;</span>'
                                            f'<div>{html.escape(sections["CITATION"].strip())}</div></div>',
                                            unsafe_allow_html=True)
                            with st.expander("View full model response", expanded=False):
                                st.markdown(response)
                        else:
                            # Response not in the expected format: show it as-is
                            st.markdown('<div class="sec-label">Detailed response</div>', unsafe_allow_html=True)
                            st.markdown(response)

                        # Retrieved sources
                        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
                        with st.expander(f"View retrieved sources ({len(sources)})", expanded=False):
                            st.markdown('<div class="card-sub">The exact document chunks used to generate this answer, '
                                        'in retrieval order.</div>', unsafe_allow_html=True)
                            cards = []
                            for i, src in enumerate(sources, 1):
                                section = src['metadata'].get('section', 'Unknown Section')
                                source_name = src['metadata'].get('source', 'Unknown')
                                excerpt = src['text'][:600] + ("..." if len(src['text']) > 600 else "")
                                sim = src.get('similarity')
                                rank = ("Top match" if i == 1 else f"Rank {i}") + \
                                       (f" &middot; {sim:.2f} similarity" if isinstance(sim, (int, float)) else "")
                                cards.append(f"""
                                <div class="src-card">
                                  <div class="src-head">
                                    <div class="src-doc">{icon("doc", size=15)} {html.escape(source_name.replace("_", " "))}</div>
                                    <span class="src-rank">{rank}</span>
                                  </div>
                                  <div class="src-sec">Section: {html.escape(str(section))}</div>
                                  <div class="src-text">{html.escape(excerpt).replace(chr(10), "<br/>")}</div>
                                  <div class="src-ref">[Source {i}] {html.escape(source_name)} &middot; {html.escape(str(section))}</div>
                                </div>""")
                            # Strip indentation so markdown never treats the HTML as a code block
                            st.markdown("".join(" ".join(l.strip() for l in c.splitlines()) for c in cards),
                                        unsafe_allow_html=True)

                        # Confidence indicator
                        if verdict == "CANNOT_DETERMINE":
                            st.info("The system could not find a definitive answer in the documents. "
                                    "Consider consulting the School of Graduate Research directly.")

                    except Exception as e:
                        st.error(f"Error processing query: {e}")

        elif check_button and not rag_ready:
            st.error("RAG pipeline is not ready. Please check the error message above.")

        else:
            st.markdown(f"""
            <div class="empty">
              <div class="ico">{icon("search", size=26)}</div>
              <h4>Your result will appear here</h4>
              <p>Enter your question and click <b>Check Eligibility</b> to get started.</p>
            </div>""", unsafe_allow_html=True)

            # Example queries
            examples = [
                "I am a single parent with a preschool child. Can I get a part-time RTP stipend?",
                "I got my PhD in February 2021. Am I eligible for DECRA?",
                "I earn $40k as an RA in my supervisor's lab. Can I still get RTP stipend?",
                "I have applied for DECRA twice before. Can I apply again?",
            ]
            st.markdown('<div class="sec-label">Example questions</div>' +
                        "".join(f'<div class="ex">{html.escape(ex)}</div>' for ex in examples),
                        unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Built for RMIT WIL Project &middot; Test-Driven RAG &middot;
    <a href="https://github.com/your-repo" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)
