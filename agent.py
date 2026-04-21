"""
agent.py
--------
Agentic AI Career Decision Assistant — Capstone Project 2025–26
Academic Year 2025–2026

Full LangGraph-style agentic workflow implementing all 7 processing nodes.

Processing pipeline:
    User → memory_node → router_node → (retrieval_node | tool_node | skip)
         → answer_node → eval_node → save_node → END

Key design principles:
    - Shared CapstoneState TypedDict flows through all nodes
    - Conditional routing after router_node and eval_node
    - Deterministic skill_gap_analyzer tool (no LLM dependency)
    - Faithfulness evaluation with automatic retry (max 2 retries)
    - Thread-based memory with sliding window (last 6 messages)
"""

import os
import json
import re
import uuid
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime

# LangChain / LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Knowledge base module
from knowledge_base import build_knowledge_base, retrieve_documents

# ---------------------------------------------------------------------------
# Knowledge Base Initialisation
# ---------------------------------------------------------------------------

try:
    build_knowledge_base(force_rebuild=False)
except Exception as _kb_err:
    print(f"[agent] Warning: knowledge base initialisation issue: {_kb_err}")


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class CapstoneState(TypedDict):
    """
    Shared state object passed through all nodes of the agentic graph.

    Fields:
        question      -- The current user question.
        messages      -- Full conversation history with LangGraph add_messages reducer.
        route         -- Router decision: 'retrieve', 'tool', or 'skip'.
        retrieved     -- List of document text chunks from ChromaDB.
        sources       -- List of metadata dicts for retrieved documents.
        tool_result   -- String output from the skill_gap_analyzer tool.
        answer        -- Final LLM-generated or fallback answer string.
        faithfulness  -- Faithfulness score assigned by the evaluation node (0.0–1.0).
        eval_retries  -- Number of answer regeneration attempts performed so far.
        user_profile  -- Dict containing name, skills, target_role, and thread_id.
    """
    question:     str
    messages:     Annotated[List[BaseMessage], add_messages]
    route:        str
    retrieved:    List[str]
    sources:      List[Dict[str, Any]]
    tool_result:  str
    answer:       str
    faithfulness: float
    eval_retries: int
    user_profile: Dict[str, Any]


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """
    Returns a ChatOpenAI LLM instance.

    Reads OPENAI_API_KEY from the environment. Uses a placeholder key
    when the variable is not set so the graph does not crash at import time
    (the answer_node handles the resulting API error gracefully).

    Args:
        temperature: Sampling temperature for the model.

    Returns:
        Configured ChatOpenAI instance.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
        api_key=api_key if api_key else "sk-placeholder",
        max_tokens=800,
    )


# ---------------------------------------------------------------------------
# Thread-Based Memory Store
# ---------------------------------------------------------------------------

MEMORY_STORE: Dict[str, List[BaseMessage]] = {}
SLIDING_WINDOW: int = 6  # Maximum messages retained per thread


def get_thread_history(thread_id: str) -> List[BaseMessage]:
    """Returns the stored message history for a given thread ID."""
    return MEMORY_STORE.get(thread_id, [])


def save_thread_history(thread_id: str, messages: List[BaseMessage]) -> None:
    """
    Persists the message history for a thread, applying the sliding window limit.

    Args:
        thread_id: Unique identifier for the conversation thread.
        messages:  Full list of messages to store (most recent SLIDING_WINDOW retained).
    """
    MEMORY_STORE[thread_id] = messages[-SLIDING_WINDOW:]


# ---------------------------------------------------------------------------
# Skill Map: Required Skills per Role
# ---------------------------------------------------------------------------

ROLE_SKILL_MAP: Dict[str, List[str]] = {
    "data analyst": [
        "SQL", "Excel", "Python", "pandas", "matplotlib", "seaborn",
        "Tableau", "Power BI", "Statistics", "Data Cleaning",
        "Business Communication", "Hypothesis Testing",
    ],
    "data scientist": [
        "Python", "SQL", "Machine Learning", "Scikit-learn", "TensorFlow",
        "PyTorch", "Linear Algebra", "Statistics", "Feature Engineering",
        "Deep Learning", "Model Evaluation", "Data Visualization",
        "Kaggle", "NLP Basics", "pandas", "NumPy",
    ],
    "software engineer": [
        "Python", "Data Structures", "Algorithms", "Git", "System Design",
        "Object-Oriented Programming", "REST APIs", "SQL", "Linux",
        "Docker", "Testing", "CI/CD", "JavaScript",
    ],
    "ai engineer": [
        "Python", "Machine Learning", "LangChain", "LangGraph", "LlamaIndex",
        "Vector Databases", "ChromaDB", "Pinecone", "RAG", "Prompt Engineering",
        "Docker", "Kubernetes", "FastAPI", "MLOps", "Hugging Face",
        "TensorFlow", "PyTorch", "SQL", "Git", "Linux", "Fine-tuning",
    ],
}


# ---------------------------------------------------------------------------
# Tool: skill_gap_analyzer
# ---------------------------------------------------------------------------

def skill_gap_analyzer(user_skills: List[str], target_role: str) -> str:
    """
    Computes a structured skill gap report for a given user and target role.

    Compares the user's current skills against the required skill set for
    the target role, then returns a formatted string report containing:
    present skills, missing skills, readiness percentage, top priority skills
    to learn next with learning resource recommendations, and an estimated
    timeline to bridge the gap.

    This function is fully deterministic — it never invokes an LLM and
    always returns a non-empty string regardless of input.

    Args:
        user_skills:  List of skill strings the user currently possesses.
        target_role:  The career role the user is targeting (case-insensitive).

    Returns:
        A formatted skill gap analysis string.
    """
    try:
        if not user_skills:
            return (
                "⚠️ No skills provided. Please share your current skills so I can "
                "perform a gap analysis."
            )

        if not target_role:
            return (
                "⚠️ No target role specified. Please provide a target role such as "
                "'Data Scientist' or 'AI Engineer'."
            )

        role_key = target_role.lower().strip()
        user_skills_lower = [s.lower().strip() for s in user_skills]

        # Match user-supplied role to the closest entry in ROLE_SKILL_MAP
        matched_role = None
        for role in ROLE_SKILL_MAP:
            if role in role_key or role_key in role:
                matched_role = role
                break

        if not matched_role:
            available = ", ".join(r.title() for r in ROLE_SKILL_MAP)
            return (
                f"Role '{target_role}' is not in the skill database. "
                f"Supported roles: {available}."
            )

        required_skills = ROLE_SKILL_MAP[matched_role]

        present_skills = [
            skill for skill in required_skills
            if skill.lower() in user_skills_lower
        ]
        missing_skills = [
            skill for skill in required_skills
            if skill.lower() not in user_skills_lower
        ]

        readiness_pct = len(present_skills) / len(required_skills) * 100
        priority_skills = missing_skills[:5]

        # Generate targeted learning resource suggestions for each priority skill
        suggestions = []
        for skill in priority_skills:
            skill_lower = skill.lower()
            if "python" in skill_lower:
                suggestions.append(
                    "• Python: Start with Python.org tutorials and Automate the Boring Stuff."
                )
            elif "sql" in skill_lower:
                suggestions.append(
                    "• SQL: Practice on SQLZoo, Mode Analytics, or StrataScratch."
                )
            elif "machine learning" in skill_lower or "scikit" in skill_lower:
                suggestions.append(
                    "• Machine Learning: Andrew Ng's ML Specialization on Coursera."
                )
            elif "statistics" in skill_lower or "linear algebra" in skill_lower:
                suggestions.append(
                    f"• {skill}: Khan Academy Statistics + StatQuest YouTube channel."
                )
            elif "docker" in skill_lower or "kubernetes" in skill_lower:
                suggestions.append(
                    f"• {skill}: TechWorld with Nana YouTube channel."
                )
            elif any(kw in skill_lower for kw in ["langchain", "langgraph", "rag", "llamaindex"]):
                suggestions.append(
                    f"• {skill}: LangChain official documentation + DeepLearning.AI short courses."
                )
            elif "tableau" in skill_lower or "power bi" in skill_lower:
                suggestions.append(
                    f"• {skill}: Official Tableau / Power BI learning portal (free tier available)."
                )
            else:
                suggestions.append(
                    f"• {skill}: Search for dedicated courses on Coursera, Udemy, or YouTube."
                )

        timeline = "2–3 months" if len(missing_skills) <= 3 else "6–12 months"

        return (
            f"📊 Skill Gap Analysis for: {target_role.title()}\n"
            f"{'─' * 45}\n"
            f"✅ Skills You Have ({len(present_skills)}/{len(required_skills)}): "
            f"{', '.join(present_skills) if present_skills else 'None matched yet'}\n\n"
            f"❌ Missing Skills ({len(missing_skills)}): "
            f"{', '.join(missing_skills) if missing_skills else 'None — you are ready!'}\n\n"
            f"📈 Role Readiness: {readiness_pct:.0f}%\n\n"
            f"🎯 Top Priority Skills to Learn Next:\n"
            f"{chr(10).join(suggestions) if suggestions else 'Great job! No critical gaps found.'}\n\n"
            f"💡 Recommended Timeline: {timeline} of focused study to bridge these gaps."
        )

    except Exception as exc:
        return f"Skill gap analysis encountered an error: {exc}. Please try again."


# ---------------------------------------------------------------------------
# NODE 1: memory_node
# ---------------------------------------------------------------------------

def memory_node(state: CapstoneState) -> CapstoneState:
    """
    Entry point of the graph. Loads the conversation history for the active
    thread, prepends it to the current state messages, and resets all
    transient fields to ensure a clean processing context for the new turn.

    Args:
        state: Incoming CapstoneState.

    Returns:
        Updated CapstoneState with merged message history and reset fields.
    """
    thread_id = state.get("user_profile", {}).get("thread_id", "default")
    history = get_thread_history(thread_id)

    new_human_msg = HumanMessage(content=state["question"])
    merged = (history + [new_human_msg])[-SLIDING_WINDOW:]

    print(f"[memory_node] Thread '{thread_id}' | "
          f"history={len(history)} msgs | merged={len(merged)} msgs")

    return {
        **state,
        "messages":     merged,
        "retrieved":    [],
        "sources":      [],
        "tool_result":  "",
        "answer":       "",
        "faithfulness": 0.0,
        "eval_retries": state.get("eval_retries", 0),
    }


# ---------------------------------------------------------------------------
# NODE 2: router_node
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = (
    "You are a routing agent for a Career Decision Assistant.\n\n"
    "Classify the user's question into exactly ONE of these three categories:\n"
    "1. \"retrieve\" — career paths, roles, salaries, roadmaps, tools, interview prep, or career advice.\n"
    "2. \"tool\" — skill comparison, skill gap analysis, missing skills, or user skill matching.\n"
    "3. \"skip\" — general conversation (greetings, thanks, off-topic, personal questions).\n\n"
    "Respond with ONLY the single word: retrieve, tool, or skip. No punctuation or explanation."
)

_SKILL_KEYWORDS = [
    "skill gap", "missing skills", "what skills do i need",
    "skills i have", "my skills", "compare my skills",
    "skill analysis", "am i ready", "skill check",
]

_GENERAL_KEYWORDS = [
    "hello", "hi", "hey", "thanks", "thank you",
    "how are you", "who are you", "what can you do", "bye", "okay",
]


def router_node(state: CapstoneState) -> CapstoneState:
    """
    Classifies the user's question into one of three routing categories.

    Uses keyword-based fast paths for obvious cases (no LLM call required).
    Falls back to a zero-temperature LLM classification for ambiguous queries.
    Defaults to 'retrieve' if the LLM call fails.

    Args:
        state: Incoming CapstoneState.

    Returns:
        Updated CapstoneState with the 'route' field populated.
    """
    question_lower = state["question"].lower()

    for kw in _SKILL_KEYWORDS:
        if kw in question_lower:
            print("[router_node] Keyword match → 'tool'")
            return {**state, "route": "tool"}

    for kw in _GENERAL_KEYWORDS:
        if question_lower.strip().startswith(kw):
            print("[router_node] Keyword match → 'skip'")
            return {**state, "route": "skip"}

    # LLM-based routing for ambiguous queries
    try:
        llm = get_llm(temperature=0.0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", _ROUTER_SYSTEM),
            ("human",  "{question}"),
        ])
        response = (prompt | llm).invoke({"question": state["question"]})
        route = response.content.strip().lower().strip("\"'")

        if route not in ("retrieve", "tool", "skip"):
            route = "retrieve"

        print(f"[router_node] LLM route → '{route}'")
        return {**state, "route": route}

    except Exception as exc:
        print(f"[router_node] LLM routing failed: {exc}. Defaulting to 'retrieve'.")
        return {**state, "route": "retrieve"}


# ---------------------------------------------------------------------------
# NODE 3: retrieval_node
# ---------------------------------------------------------------------------

def retrieval_node(state: CapstoneState) -> CapstoneState:
    """
    Queries the ChromaDB vector store for the top-3 semantically relevant
    documents. Only executes when route == 'retrieve'.

    Writes retrieved document texts and their metadata into the state for
    downstream use by answer_node. On retrieval failure, writes a graceful
    fallback message so the graph can continue.

    Args:
        state: Incoming CapstoneState.

    Returns:
        Updated CapstoneState with 'retrieved' and 'sources' fields populated.
    """
    if state["route"] != "retrieve":
        return state

    try:
        results = retrieve_documents(state["question"], n_results=3)
        print(f"[retrieval_node] Retrieved {len(results['documents'])} documents.")
        return {
            **state,
            "retrieved": results["documents"],
            "sources":   results["metadatas"],
        }
    except Exception as exc:
        print(f"[retrieval_node] Retrieval error: {exc}")
        return {
            **state,
            "retrieved": ["No relevant documents found due to a retrieval error."],
            "sources":   [{"topic": "Unknown", "type": "error"}],
        }


# ---------------------------------------------------------------------------
# NODE 4: tool_node
# ---------------------------------------------------------------------------

def tool_node(state: CapstoneState) -> CapstoneState:
    """
    Invokes the skill_gap_analyzer deterministic tool. Only executes when
    route == 'tool'.

    Extracts user skills and target role from the user profile. If the target
    role is missing from the profile, attempts to infer it from the question
    text. Returns a clear prompting message if required inputs are absent.

    Args:
        state: Incoming CapstoneState.

    Returns:
        Updated CapstoneState with 'tool_result' field populated.
    """
    if state["route"] != "tool":
        return state

    profile     = state.get("user_profile", {})
    user_skills = profile.get("skills", [])
    target_role = profile.get("target_role", "")

    # Attempt to infer target role from the question if not set in profile
    if not target_role:
        question_lower = state["question"].lower()
        for role in ROLE_SKILL_MAP:
            if role in question_lower:
                target_role = role.title()
                break

    if not user_skills:
        result = (
            "⚠️ Please share your current skills to perform a gap analysis. "
            "You can enter them in the sidebar, or mention them directly: "
            "'I know Python, SQL, and Excel — what am I missing for a Data Analyst role?'"
        )
    elif not target_role:
        result = (
            "⚠️ Please specify a target role for the skill gap analysis. "
            "For example: 'What skills am I missing for a Data Scientist role?'"
        )
    else:
        result = skill_gap_analyzer(user_skills, target_role)

    print(f"[tool_node] Skill gap analysis complete for role: '{target_role}'")
    return {**state, "tool_result": result}


# ---------------------------------------------------------------------------
# NODE 5: answer_node
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM = (
    "You are an expert Career Decision Assistant specialising in data, AI, and "
    "software engineering careers. Your goal is to help students and professionals "
    "make informed career decisions.\n\n"
    "Guidelines:\n"
    "- Provide accurate, actionable, and encouraging advice.\n"
    "- Use concrete examples, numbers, and timelines where relevant.\n"
    "- If context documents are provided, use them as your primary source.\n"
    "- Keep answers focused and structured (under 350 words).\n"
    "- Always end with one concrete, actionable next step.\n\n"
    "Context Documents (if available):\n{context}\n\n"
    "Tool Result (if available):\n{tool_result}"
)


def answer_node(state: CapstoneState) -> CapstoneState:
    """
    Generates the final answer using the LLM, grounded in retrieved documents
    and/or tool results.

    Assembles a prompt with the system persona, context from retrieved
    documents, tool results (if any), and the user's question. Includes a
    comprehensive fallback strategy if the LLM call fails, ensuring the user
    always receives a useful response.

    Args:
        state: Incoming CapstoneState.

    Returns:
        Updated CapstoneState with the 'answer' field populated.
    """
    # Build context string from retrieved documents
    if state["retrieved"]:
        context_parts = []
        for i, (doc, meta) in enumerate(zip(state["retrieved"], state["sources"])):
            topic = meta.get("topic", "General")
            context_parts.append(f"[Source {i + 1}: {topic}]\n{doc[:500]}")
        context_str = "\n\n".join(context_parts)
    else:
        context_str = "No specific documents retrieved."

    tool_result_str = state.get("tool_result", "") or "No tool analysis performed."

    try:
        llm = get_llm(temperature=0.4)
        prompt = ChatPromptTemplate.from_messages([
            ("system", _ANSWER_SYSTEM.format(
                context=context_str,
                tool_result=tool_result_str,
            )),
            ("human", "{question}"),
        ])
        response = (prompt | llm).invoke({"question": state["question"]})
        answer = response.content.strip()

    except Exception as exc:
        print(f"[answer_node] LLM error: {exc}")
        # Graceful fallback: surface tool result or retrieved content directly
        if state.get("tool_result"):
            answer = state["tool_result"]
        elif state["retrieved"]:
            answer = f"Based on our knowledge base:\n\n{state['retrieved'][0][:600]}"
        else:
            answer = (
                "I am currently unable to connect to the AI service. "
                "Please ensure your OPENAI_API_KEY is set correctly and try again. "
                "Skill gap analysis remains available if you enter your skills in the sidebar."
            )

    print(f"[answer_node] Answer generated ({len(answer)} characters).")
    return {**state, "answer": answer}


# ---------------------------------------------------------------------------
# NODE 6: eval_node
# ---------------------------------------------------------------------------

_EVAL_SYSTEM = (
    "You are a quality evaluation judge for a career advisory AI system.\n\n"
    "Score the faithfulness of the ANSWER relative to the QUESTION and CONTEXT:\n"
    "- 1.0: Completely accurate, directly addresses the question, well-grounded in context.\n"
    "- 0.7–0.9: Mostly accurate with minor omissions or additions.\n"
    "- 0.4–0.6: Partially addresses the question or contains some inaccuracies.\n"
    "- 0.0–0.3: Irrelevant, hallucinated, or contradicts the context.\n\n"
    "Respond with ONLY a decimal number between 0.0 and 1.0. No explanation."
)


def eval_node(state: CapstoneState) -> CapstoneState:
    """
    Evaluates the faithfulness of the generated answer using an LLM judge.

    If the score falls below 0.7 and fewer than 2 retries have been attempted,
    clears the answer and increments eval_retries, causing the conditional edge
    to route back to answer_node for regeneration.

    Tool-route answers receive an automatic score of 0.95 (deterministic output,
    no LLM call required).

    Args:
        state: Incoming CapstoneState.

    Returns:
        Updated CapstoneState with 'faithfulness' and optionally cleared 'answer'.
    """
    answer       = state["answer"]
    eval_retries = state.get("eval_retries", 0)

    # Penalise trivially short answers immediately
    if not answer or len(answer) < 20:
        print("[eval_node] Answer too short → faithfulness = 0.0")
        return {**state, "faithfulness": 0.0}

    # Deterministic tool outputs are inherently faithful
    if state["route"] == "tool" and state.get("tool_result"):
        print("[eval_node] Tool route → faithfulness = 0.95")
        return {**state, "faithfulness": 0.95}

    context = " ".join(state["retrieved"][:2]) if state["retrieved"] else "general knowledge"

    try:
        llm = get_llm(temperature=0.0)
        eval_prompt = (
            f"QUESTION: {state['question']}\n\n"
            f"CONTEXT: {context[:800]}\n\n"
            f"ANSWER: {answer}\n\n"
            "Score the faithfulness of the ANSWER (0.0 to 1.0):"
        )
        response = llm.invoke([
            SystemMessage(content=_EVAL_SYSTEM),
            HumanMessage(content=eval_prompt),
        ])
        match = re.search(r"[0-9]\.[0-9]+|[01]", response.content.strip())
        faithfulness = float(match.group()) if match else 0.75
        faithfulness = max(0.0, min(1.0, faithfulness))

    except Exception as exc:
        print(f"[eval_node] Evaluation LLM error: {exc}. Defaulting to 0.75.")
        faithfulness = 0.75

    print(f"[eval_node] Faithfulness: {faithfulness:.2f} | Retries so far: {eval_retries}")

    # Trigger retry if score is below threshold and retries remain
    if faithfulness < 0.7 and eval_retries < 2:
        print("[eval_node] Score below threshold → retrying answer generation.")
        return {
            **state,
            "faithfulness": faithfulness,
            "eval_retries": eval_retries + 1,
            "answer":       "",
        }

    return {**state, "faithfulness": faithfulness}


# ---------------------------------------------------------------------------
# NODE 7: save_node
# ---------------------------------------------------------------------------

_CONVERSATION_LOG_FILE = "conversation_log.jsonl"


def save_node(state: CapstoneState) -> CapstoneState:
    """
    Persists the completed conversation turn to thread memory and the JSONL
    audit log.

    Appends the AI message to the thread's memory store (subject to sliding
    window) and writes a structured JSON record to conversation_log.jsonl.

    Args:
        state: Incoming CapstoneState.

    Returns:
        Updated CapstoneState with the AI message appended to 'messages'.
    """
    thread_id = state.get("user_profile", {}).get("thread_id", "default")

    ai_msg = AIMessage(content=state["answer"])
    updated_messages = list(state["messages"]) + [ai_msg]
    save_thread_history(thread_id, updated_messages)

    log_record = {
        "timestamp":    datetime.now().isoformat(),
        "thread_id":    thread_id,
        "question":     state["question"],
        "route":        state["route"],
        "answer":       state["answer"],
        "faithfulness": state["faithfulness"],
        "eval_retries": state["eval_retries"],
        "sources":      [m.get("topic", "") for m in state["sources"]],
        "user_profile": state.get("user_profile", {}),
    }

    try:
        with open(_CONVERSATION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record) + "\n")
    except Exception as exc:
        print(f"[save_node] Log write error: {exc}")

    print(
        f"[save_node] Saved. Thread: {thread_id} | "
        f"Faithfulness: {state['faithfulness']:.2f}"
    )
    return {**state, "messages": updated_messages}


# ---------------------------------------------------------------------------
# Conditional Edge Functions
# ---------------------------------------------------------------------------

def route_dispatcher(state: CapstoneState) -> str:
    """
    Conditional edge after router_node.

    Returns the name of the next node based on the routing decision:
    'retrieve', 'tool', or 'answer' (for skip).
    """
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool"
    if route == "skip":
        return "answer"
    return "retrieve"


def should_retry(state: CapstoneState) -> str:
    """
    Conditional edge after eval_node.

    Returns 'retry' if the answer field has been cleared (indicating a
    retry was requested by eval_node), otherwise returns 'save'.
    """
    if not state.get("answer") and state.get("eval_retries", 0) > 0:
        return "retry"
    return "save"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_career_agent_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph-style agentic workflow.

    Graph topology:
        memory → router → [retrieve | tool | answer] → answer
               → eval → [retry (→ answer) | save] → END

    Returns:
        Compiled StateGraph ready for invocation.
    """
    workflow = StateGraph(CapstoneState)

    workflow.add_node("memory",   memory_node)
    workflow.add_node("router",   router_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("tool",     tool_node)
    workflow.add_node("answer",   answer_node)
    workflow.add_node("eval",     eval_node)
    workflow.add_node("save",     save_node)

    workflow.set_entry_point("memory")
    workflow.add_edge("memory", "router")

    workflow.add_conditional_edges(
        "router",
        route_dispatcher,
        {"retrieve": "retrieve", "tool": "tool", "answer": "answer"},
    )

    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("tool",     "answer")
    workflow.add_edge("answer",   "eval")

    workflow.add_conditional_edges(
        "eval",
        should_retry,
        {"retry": "answer", "save": "save"},
    )

    workflow.add_edge("save", END)

    return workflow.compile()


# Compile the graph once at module level for reuse across requests
career_agent = build_career_agent_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(
    question: str,
    user_profile: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the full agentic pipeline for a given career question.

    Args:
        question:     The user's career-related question.
        user_profile: Optional dict with 'name', 'skills', 'target_role'.
        thread_id:    Optional thread ID for conversation memory continuity.
                      A new ID is generated if not provided.

    Returns:
        Dict containing 'answer', 'route', 'faithfulness', 'sources',
        'tool_result', 'eval_retries', and 'thread_id'.
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())[:8]

    profile = dict(user_profile or {})
    profile["thread_id"] = thread_id

    initial_state: CapstoneState = {
        "question":     question,
        "messages":     [],
        "route":        "",
        "retrieved":    [],
        "sources":      [],
        "tool_result":  "",
        "answer":       "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "user_profile": profile,
    }

    try:
        final_state = career_agent.invoke(initial_state)
        return {
            "answer":       final_state.get("answer", "No answer generated."),
            "route":        final_state.get("route", "unknown"),
            "faithfulness": final_state.get("faithfulness", 0.0),
            "sources":      final_state.get("sources", []),
            "tool_result":  final_state.get("tool_result", ""),
            "eval_retries": final_state.get("eval_retries", 0),
            "thread_id":    thread_id,
        }
    except Exception as exc:
        print(f"[run_agent] Graph execution error: {exc}")
        return {
            "answer":       f"An error occurred while processing your question: {exc}",
            "route":        "error",
            "faithfulness": 0.0,
            "sources":      [],
            "tool_result":  "",
            "eval_retries": 0,
            "thread_id":    thread_id,
        }


# ---------------------------------------------------------------------------
# CLI Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Agentic AI Career Decision Assistant — CLI Test")
    print("  Capstone Project 2025–26")
    print("=" * 60)

    test_profile = {
        "name":        "Test User",
        "skills":      ["Python", "SQL", "Excel", "pandas"],
        "target_role": "Data Scientist",
        "thread_id":   "cli_test_001",
    }

    test_questions = [
        "What skills do I need to become a Data Scientist?",
        "What is the salary of an AI Engineer in India?",
        "What am I missing to become a Data Scientist?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n[Test {i}] {question}")
        result = run_agent(question, user_profile=test_profile, thread_id="cli_test_001")
        print(f"  Route:        {result['route']}")
        print(f"  Faithfulness: {result['faithfulness']:.2f}")
        print(f"  Answer:       {result['answer'][:250]}...")
        print("-" * 60)
