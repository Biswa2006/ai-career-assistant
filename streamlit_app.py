"""
streamlit_app.py
Agentic AI Career Decision Assistant — Streamlit Chat Interface
"""

import streamlit as st
import uuid
import os
import json
from datetime import datetime

# Import the agent runner
from agent import run_agent, skill_gap_analyzer, ROLE_SKILL_MAP

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Career AI Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS — Modern dark theme with sharp accents
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0d0f1a 0%, #111827 50%, #0d1117 100%);
    color: #e2e8f0;
}

/* Header */
.main-header {
    background: linear-gradient(90deg, #1e3a5f 0%, #1a1f36 100%);
    border-left: 4px solid #38bdf8;
    padding: 1.2rem 1.5rem;
    border-radius: 0 12px 12px 0;
    margin-bottom: 1.5rem;
}
.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #38bdf8;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94a3b8;
    font-size: 0.85rem;
    margin: 0.3rem 0 0 0;
}

/* Chat messages */
.chat-message-user {
    background: linear-gradient(135deg, #1e3a5f, #1a2744);
    border: 1px solid #2563eb40;
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0 0.5rem 15%;
    color: #bfdbfe;
    font-size: 0.93rem;
    line-height: 1.6;
}
.chat-message-assistant {
    background: linear-gradient(135deg, #0f2027, #131d2e);
    border: 1px solid #0ea5e920;
    border-left: 3px solid #0ea5e9;
    border-radius: 4px 16px 16px 16px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 15% 0.5rem 0;
    color: #e2e8f0;
    font-size: 0.93rem;
    line-height: 1.6;
}
.user-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.assistant-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #22d3ee;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}

/* Meta badges */
.meta-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.6rem;
}
.badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}
.badge-route { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb; }
.badge-faith-high { background: #052e16; color: #4ade80; border: 1px solid #16a34a; }
.badge-faith-mid  { background: #431407; color: #fb923c; border: 1px solid #c2410c; }
.badge-faith-low  { background: #3b0764; color: #c084fc; border: 1px solid #9333ea; }
.badge-sources { background: #1c1917; color: #a8a29e; border: 1px solid #44403c; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stTextArea > div > div > textarea,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Input area */
.stChatInputContainer {
    background: #111827 !important;
    border-top: 1px solid #1e3a5f !important;
}

/* Section titles in sidebar */
.sidebar-section {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e3a5f;
}

/* Info boxes */
.info-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.83rem;
    color: #94a3b8;
    line-height: 1.6;
    margin-bottom: 0.6rem;
}
.info-box strong { color: #38bdf8; }

/* Stat cards */
.stat-row { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
.stat-card {
    flex: 1;
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
    text-align: center;
}
.stat-card .stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #38bdf8;
    font-weight: 700;
}
.stat-card .stat-label {
    font-size: 0.65rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #475569;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.8rem; }
.empty-state h3 { color: #64748b; font-family: 'Space Mono', monospace; font-size: 1rem; }

/* Scrollable chat area */
.chat-container { max-height: 65vh; overflow-y: auto; padding-right: 0.5rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #0f2d4a) !important;
    border: 1px solid #2563eb60 !important;
    color: #60a5fa !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: #38bdf8 !important;
    color: #38bdf8 !important;
    transform: translateY(-1px) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #111827 !important;
    color: #94a3b8 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())[:8]
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_skills" not in st.session_state:
        st.session_state.user_skills = []
    if "target_role" not in st.session_state:
        st.session_state.target_role = ""
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "avg_faithfulness" not in st.session_state:
        st.session_state.avg_faithfulness = []
    if "api_key_set" not in st.session_state:
        st.session_state.api_key_set = bool(os.getenv("OPENAI_API_KEY"))


init_session()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <div style="font-size:2.5rem">🎯</div>
        <div style="font-family:'Space Mono',monospace; font-size:1rem; color:#38bdf8; font-weight:700;">
            Career AI
        </div>
        <div style="font-size:0.72rem; color:#475569; margin-top:2px;">
            Agentic Decision Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">⚙️ Configuration</div>', unsafe_allow_html=True)

    # API Key
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Required for AI-generated answers. Get yours at platform.openai.com"
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        st.session_state.api_key_set = True
    
    if st.session_state.api_key_set:
        st.markdown('<div style="color:#4ade80;font-size:0.75rem;">✅ API Key configured</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#f59e0b;font-size:0.75rem;">⚠️ API Key not set — skill gap tool still works</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">👤 User Profile</div>', unsafe_allow_html=True)

    st.session_state.user_name = st.text_input(
        "Your Name",
        value=st.session_state.user_name,
        placeholder="e.g., Arjun Sharma"
    )

    st.session_state.target_role = st.selectbox(
        "Target Career Role",
        options=["", "Data Analyst", "Data Scientist", "Software Engineer", "AI Engineer"],
        index=0 if not st.session_state.target_role else
              ["", "Data Analyst", "Data Scientist", "Software Engineer", "AI Engineer"].index(
                  st.session_state.target_role) if st.session_state.target_role in
              ["", "Data Analyst", "Data Scientist", "Software Engineer", "AI Engineer"] else 0
    )

    skills_input = st.text_area(
        "Your Current Skills (comma-separated)",
        value=", ".join(st.session_state.user_skills),
        placeholder="e.g., Python, SQL, Excel, pandas",
        height=80
    )
    if skills_input:
        st.session_state.user_skills = [
            s.strip() for s in skills_input.split(",") if s.strip()
        ]

    # Quick Skill Gap Button
    if st.button("🔍 Run Skill Gap Analysis Now", use_container_width=True):
        if st.session_state.user_skills and st.session_state.target_role:
            with st.spinner("Analyzing..."):
                result = skill_gap_analyzer(
                    st.session_state.user_skills,
                    st.session_state.target_role
                )
            st.session_state.messages.append({
                "role": "user",
                "content": f"Skill gap analysis for {st.session_state.target_role}",
                "meta": {}
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": result,
                "meta": {"route": "tool", "faithfulness": 0.95, "sources": []}
            })
            st.session_state.total_queries += 1
            st.rerun()
        else:
            st.warning("Please enter your skills and select a target role first.")

    st.markdown('<div class="sidebar-section">📊 Session Stats</div>', unsafe_allow_html=True)

    avg_faith = (
        sum(st.session_state.avg_faithfulness) / len(st.session_state.avg_faithfulness)
        if st.session_state.avg_faithfulness else 0.0
    )

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-value">{st.session_state.total_queries}</div>
            <div class="stat-label">Queries</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_faith:.0%}</div>
            <div class="stat-label">Avg Faith</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{st.session_state.thread_id}</div>
            <div class="stat-label">Thread ID</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">🔄 Controls</div>', unsafe_allow_html=True)

    if st.button("🆕 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.total_queries = 0
        st.session_state.avg_faithfulness = []
        st.rerun()

    # Sample questions
    st.markdown('<div class="sidebar-section">💡 Sample Questions</div>', unsafe_allow_html=True)

    sample_questions = [
        "How do I become a Data Scientist?",
        "What salary can I expect as an AI Engineer?",
        "What skills am I missing for my target role?",
        "Compare Data Analyst vs Data Scientist",
        "What tools should I learn for software engineering?",
    ]

    for sq in sample_questions:
        if st.button(sq, use_container_width=True, key=f"sq_{sq[:20]}"):
            st.session_state["prefill_question"] = sq
            st.rerun()

    st.markdown('<div class="sidebar-section">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Architecture:</strong> LangGraph-style Agentic AI<br>
        <strong>Nodes:</strong> memory → router → retrieve/tool → answer → eval → save<br>
        <strong>Vector DB:</strong> ChromaDB (all-MiniLM-L6-v2)<br>
        <strong>LLM:</strong> GPT-3.5-Turbo<br>
        <strong>Docs:</strong> 10 career knowledge documents<br>
        <strong>Memory:</strong> Thread-based sliding window (6 msgs)
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Agentic AI Career Decision Assistant</h1>
    <p>Multi-step reasoning • Skill gap analysis • Career path guidance • Powered by LangGraph-style architecture</p>
</div>
""", unsafe_allow_html=True)

# Render chat history
if not st.session_state.messages:
    greeting_name = f", {st.session_state.user_name}" if st.session_state.user_name else ""
    st.markdown(f"""
    <div class="empty-state">
        <div class="icon">🎓</div>
        <h3>Hello{greeting_name}! Ready to explore your career path?</h3>
        <p style="font-size:0.85rem; color:#475569; max-width:500px; margin:0 auto;">
            Ask me about careers in data, AI, or software engineering. 
            I can help you understand skill requirements, salaries, roadmaps, 
            interview tips, and analyze your skill gaps.
        </p>
        <div style="margin-top:1.5rem; display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap;">
            <span style="background:#1e3a5f; color:#60a5fa; padding:4px 12px; border-radius:20px; font-size:0.78rem; border:1px solid #2563eb;">Career Paths</span>
            <span style="background:#1e3a5f; color:#60a5fa; padding:4px 12px; border-radius:20px; font-size:0.78rem; border:1px solid #2563eb;">Skill Analysis</span>
            <span style="background:#1e3a5f; color:#60a5fa; padding:4px 12px; border-radius:20px; font-size:0.78rem; border:1px solid #2563eb;">Salary Insights</span>
            <span style="background:#1e3a5f; color:#60a5fa; padding:4px 12px; border-radius:20px; font-size:0.78rem; border:1px solid #2563eb;">Interview Prep</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message-user">
                <div class="user-label">👤 You</div>
                {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            meta = msg.get("meta", {})
            route = meta.get("route", "")
            faithfulness = meta.get("faithfulness", 0.0)
            sources = meta.get("sources", [])
            eval_retries = meta.get("eval_retries", 0)

            # Faithfulness badge color
            if faithfulness >= 0.8:
                faith_class = "badge-faith-high"
            elif faithfulness >= 0.6:
                faith_class = "badge-faith-mid"
            else:
                faith_class = "badge-faith-low"

            source_topics = ", ".join([s.get("topic", "") for s in sources if s.get("topic")]) or "general knowledge"

            retry_badge = f'<span class="badge badge-faith-mid">🔄 retried {eval_retries}x</span>' if eval_retries > 0 else ""

            # Format content (convert **text** to <strong> and \n to <br>)
            content = msg["content"]
            content = content.replace("**", "<strong>", 1)
            while "**" in content:
                content = content.replace("**", "</strong>", 1)
                if "**" in content:
                    content = content.replace("**", "<strong>", 1)
            content = content.replace("\n", "<br>")

            st.markdown(f"""
            <div class="chat-message-assistant">
                <div class="assistant-label">🤖 Career AI</div>
                {content}
                <div class="meta-row">
                    <span class="badge badge-route">📍 {route}</span>
                    <span class="badge {faith_class}">⚡ {faithfulness:.0%} faithful</span>
                    <span class="badge badge-sources">📚 {source_topics[:40]}</span>
                    {retry_badge}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------

# Handle prefilled question from sidebar buttons
prefill = st.session_state.pop("prefill_question", None)

user_input = st.chat_input(
    placeholder="Ask about careers, skills, salaries, roadmaps, or interview tips...",
)

# Use prefill if no direct input
if prefill and not user_input:
    user_input = prefill

if user_input:
    user_input = user_input.strip()
    if not user_input:
        st.stop()

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "meta": {}
    })

    # Build user profile from session state
    user_profile = {
        "name": st.session_state.user_name,
        "skills": st.session_state.user_skills,
        "target_role": st.session_state.target_role,
        "thread_id": st.session_state.thread_id
    }

    # Run agent
    with st.spinner("🧠 Thinking..."):
        result = run_agent(
            question=user_input,
            user_profile=user_profile,
            thread_id=st.session_state.thread_id
        )

    # Update session stats
    st.session_state.total_queries += 1
    if result["faithfulness"] > 0:
        st.session_state.avg_faithfulness.append(result["faithfulness"])

    # Add AI response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "meta": {
            "route": result["route"],
            "faithfulness": result["faithfulness"],
            "sources": result["sources"],
            "eval_retries": result["eval_retries"]
        }
    })

    st.rerun()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; color:#1e293b; font-size:0.72rem; 
            font-family:'Space Mono',monospace; margin-top:2rem; 
            padding-top:1rem; border-top:1px solid #1e293b;">
    Agentic AI Career Decision Assistant • LangGraph Architecture • 
    ChromaDB + all-MiniLM-L6-v2 • Capstone Project 2025
</div>
""", unsafe_allow_html=True)
