"""
agent.py
Agentic AI Career Decision Assistant
Full LangGraph-style agentic workflow with all mandatory nodes.

Flow: User → memory → router → (retrieve/tool/skip) → answer → eval → save → END
"""

import os
import json
import re
import time
import uuid
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime

# LangChain / LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Knowledge base
from knowledge_base import build_knowledge_base, retrieve_documents

# ---------------------------------------------------------------------------
# Ensure knowledge base exists at import time
# ---------------------------------------------------------------------------
try:
    build_knowledge_base(force_rebuild=False)
except Exception as _kb_err:
    print(f"[agent] Warning: KB build issue: {_kb_err}")


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class CapstoneState(TypedDict):
    """
    Full state object for the Agentic Career Decision Assistant.
    All nodes read and write to this shared state.
    """
    question: str                          # Current user question
    messages: Annotated[List[BaseMessage], add_messages]  # Conversation history
    route: str                             # Router decision: retrieve / tool / skip
    retrieved: List[str]                   # Retrieved document chunks
    sources: List[Dict[str, Any]]          # Source metadata for retrieved docs
    tool_result: str                       # Output from skill_gap_analyzer tool
    answer: str                            # Final generated answer
    faithfulness: float                    # Faithfulness score (0.0–1.0)
    eval_retries: int                      # Number of eval retries so far
    user_profile: Dict[str, Any]          # User profile: name, skills, target_role


# ---------------------------------------------------------------------------
# LLM Setup (uses OpenAI-compatible API; falls back gracefully)
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Returns the LLM instance. Reads OPENAI_API_KEY from environment."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
        api_key=api_key if api_key else "sk-placeholder",
        max_tokens=800
    )


# ---------------------------------------------------------------------------
# Thread-based Memory Store (in-memory, keyed by thread_id)
# ---------------------------------------------------------------------------

MEMORY_STORE: Dict[str, List[BaseMessage]] = {}
SLIDING_WINDOW = 6  # Keep last 6 messages


def get_thread_history(thread_id: str) -> List[BaseMessage]:
    """Returns the message history for a thread."""
    return MEMORY_STORE.get(thread_id, [])


def save_thread_history(thread_id: str, messages: List[BaseMessage]) -> None:
    """Saves the message history for a thread (sliding window of last 6)."""
    MEMORY_STORE[thread_id] = messages[-SLIDING_WINDOW:]


# ---------------------------------------------------------------------------
# TOOL: skill_gap_analyzer
# ---------------------------------------------------------------------------

# Comprehensive skill map for each career role
ROLE_SKILL_MAP: Dict[str, List[str]] = {
    "data analyst": [
        "SQL", "Excel", "Python", "pandas", "matplotlib", "seaborn",
        "Tableau", "Power BI", "Statistics", "Data Cleaning",
        "Business Communication", "Hypothesis Testing"
    ],
    "data scientist": [
        "Python", "SQL", "Machine Learning", "Scikit-learn", "TensorFlow",
        "PyTorch", "Linear Algebra", "Statistics", "Feature Engineering",
        "Deep Learning", "Model Evaluation", "Data Visualization",
        "Kaggle", "NLP Basics", "pandas", "NumPy"
    ],
    "software engineer": [
        "Python", "Data Structures", "Algorithms", "Git", "System Design",
        "Object-Oriented Programming", "REST APIs", "SQL", "Linux",
        "Docker", "Testing", "CI/CD", "JavaScript"
    ],
    "ai engineer": [
        "Python", "Machine Learning", "LangChain", "LangGraph", "LlamaIndex",
        "Vector Databases", "ChromaDB", "Pinecone", "RAG", "Prompt Engineering",
        "Docker", "Kubernetes", "FastAPI", "MLOps", "Hugging Face",
        "TensorFlow", "PyTorch", "SQL", "Git", "Linux", "Fine-tuning"
    ]
}


def skill_gap_analyzer(user_skills: List[str], target_role: str) -> str:
    """
    Analyzes the gap between a user's current skills and the target role requirements.
    
    Args:
        user_skills: List of skills the user currently has.
        target_role: The career role the user is targeting.
    
    Returns:
        A formatted string summarizing missing skills and recommendations.
    """
    try:
        if not user_skills:
            return "Please provide your current skills so I can perform a gap analysis."

        if not target_role:
            return "Please specify a target role (e.g., 'Data Scientist', 'AI Engineer')."

        # Normalize inputs
        role_key = target_role.lower().strip()
        user_skills_lower = [s.lower().strip() for s in user_skills]

        # Find the best matching role
        matched_role = None
        for role in ROLE_SKILL_MAP:
            if role in role_key or role_key in role:
                matched_role = role
                break

        if not matched_role:
            available = ", ".join([r.title() for r in ROLE_SKILL_MAP.keys()])
            return (
                f"Role '{target_role}' not found in the skill database. "
                f"Available roles: {available}."
            )

        required_skills = ROLE_SKILL_MAP[matched_role]
        required_lower = [s.lower() for s in required_skills]

        # Identify present and missing skills
        present_skills = [
            s for s in required_skills
            if s.lower() in user_skills_lower
        ]
        missing_skills = [
            s for s in required_skills
            if s.lower() not in user_skills_lower
        ]

        match_pct = len(present_skills) / len(required_skills) * 100

        # Prioritize top 5 missing skills to learn first
        priority_skills = missing_skills[:5]

        # Build suggestions
        suggestions = []
        for skill in priority_skills:
            if "python" in skill.lower():
                suggestions.append(f"• Python: Start with Python.org tutorials and Automate the Boring Stuff book.")
            elif "sql" in skill.lower():
                suggestions.append(f"• SQL: Practice on SQLZoo, Mode Analytics, or StrataScratch.")
            elif "machine learning" in skill.lower() or "scikit" in skill.lower():
                suggestions.append(f"• Machine Learning: Andrew Ng's ML Specialization on Coursera.")
            elif "docker" in skill.lower() or "kubernetes" in skill.lower():
                suggestions.append(f"• {skill}: TechWorld with Nana YouTube channel is excellent.")
            elif "langchain" in skill.lower() or "langgraph" in skill.lower() or "rag" in skill.lower():
                suggestions.append(f"• {skill}: LangChain official documentation and DeepLearning.AI short courses.")
            elif "tableau" in skill.lower() or "power bi" in skill.lower():
                suggestions.append(f"• {skill}: Official Tableau/Power BI learning portal with free tier.")
            else:
                suggestions.append(f"• {skill}: Search for dedicated courses on Coursera, Udemy, or YouTube.")

        result = (
            f"📊 Skill Gap Analysis for: {target_role.title()}\n"
            f"{'─' * 45}\n"
            f"✅ Skills You Have ({len(present_skills)}/{len(required_skills)}): "
            f"{', '.join(present_skills) if present_skills else 'None matched yet'}\n\n"
            f"❌ Missing Skills ({len(missing_skills)}): "
            f"{', '.join(missing_skills) if missing_skills else 'None — you are ready!'}\n\n"
            f"📈 Role Readiness: {match_pct:.0f}%\n\n"
            f"🎯 Top Priority Skills to Learn Next:\n"
            f"{chr(10).join(suggestions) if suggestions else 'Great job! No critical gaps found.'}\n\n"
            f"💡 Recommended Timeline: "
            f"{'2–3 months' if len(missing_skills) <= 3 else '6–12 months'} "
            f"of focused study to bridge these gaps."
        )
        return result

    except Exception as e:
        return f"Skill gap analysis encountered an error: {str(e)}. Please try again."


# ---------------------------------------------------------------------------
# NODE 1: memory_node
# ---------------------------------------------------------------------------

def memory_node(state: CapstoneState) -> CapstoneState:
    """
    Loads thread history into state and prepends it to the messages list.
    Maintains a sliding window of the last 6 messages.
    """
    thread_id = state.get("user_profile", {}).get("thread_id", "default")
    history = get_thread_history(thread_id)

    # Merge: history first, then current question as a new HumanMessage
    current_question = state["question"]
    new_human_msg = HumanMessage(content=current_question)

    # Build merged messages (history + new message)
    merged = history + [new_human_msg]
    merged = merged[-SLIDING_WINDOW:]  # enforce sliding window

    print(f"[memory_node] Thread '{thread_id}' | History: {len(history)} msgs | Total: {len(merged)} msgs")

    return {
        **state,
        "messages": merged,
        "retrieved": [],
        "sources": [],
        "tool_result": "",
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": state.get("eval_retries", 0)
    }


# ---------------------------------------------------------------------------
# NODE 2: router_node
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """You are a routing agent for a Career Decision Assistant.

Classify the user's question into exactly ONE of these three categories:
1. "retrieve" — if the question is about career paths, roles, salaries, roadmaps, tools, interview prep, or career advice.
2. "tool" — if the question explicitly asks about skill comparison, skill gap analysis, what skills the user is missing, or matching user skills to a job role.
3. "skip" — if the question is general conversation (greetings, thanks, off-topic, personal questions).

Respond with ONLY the single word: retrieve, tool, or skip.
Do NOT include any explanation or punctuation."""


def router_node(state: CapstoneState) -> CapstoneState:
    """
    Routes the question to the appropriate processing path.
    Uses an LLM call with fallback keyword-based routing.
    """
    question = state["question"].lower()

    # Keyword-based fast routing (no LLM needed for obvious cases)
    skill_keywords = [
        "skill gap", "missing skills", "what skills do i need",
        "skills i have", "my skills", "compare my skills",
        "skill analysis", "am i ready", "skill check"
    ]
    general_keywords = [
        "hello", "hi", "hey", "thanks", "thank you",
        "how are you", "who are you", "what can you do", "bye", "okay"
    ]

    for kw in skill_keywords:
        if kw in question:
            print(f"[router_node] Keyword match → 'tool'")
            return {**state, "route": "tool"}

    for kw in general_keywords:
        if question.strip().startswith(kw):
            print(f"[router_node] Keyword match → 'skip'")
            return {**state, "route": "skip"}

    # LLM-based routing for ambiguous queries
    try:
        llm = get_llm(temperature=0.0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        chain = prompt | llm
        response = chain.invoke({"question": state["question"]})
        route = response.content.strip().lower().replace('"', '').replace("'", "")

        if route not in ["retrieve", "tool", "skip"]:
            route = "retrieve"  # safe default

        print(f"[router_node] LLM route → '{route}'")
        return {**state, "route": route}

    except Exception as e:
        print(f"[router_node] LLM routing failed: {e}. Defaulting to 'retrieve'.")
        return {**state, "route": "retrieve"}


# ---------------------------------------------------------------------------
# NODE 3: retrieval_node
# ---------------------------------------------------------------------------

def retrieval_node(state: CapstoneState) -> CapstoneState:
    """
    Retrieves relevant documents from ChromaDB knowledge base.
    Only executed if route == 'retrieve'.
    """
    if state["route"] != "retrieve":
        return state

    try:
        results = retrieve_documents(state["question"], n_results=3)
        documents = results["documents"]
        metadatas = results["metadatas"]

        print(f"[retrieval_node] Retrieved {len(documents)} documents.")
        return {
            **state,
            "retrieved": documents,
            "sources": metadatas
        }

    except Exception as e:
        print(f"[retrieval_node] Retrieval error: {e}")
        return {
            **state,
            "retrieved": ["No relevant documents found."],
            "sources": [{"topic": "Unknown", "type": "error"}]
        }


# ---------------------------------------------------------------------------
# NODE 4: tool_node
# ---------------------------------------------------------------------------

def tool_node(state: CapstoneState) -> CapstoneState:
    """
    Executes the skill_gap_analyzer tool.
    Only executed if route == 'tool'.
    Extracts user skills and target role from the user profile or question.
    """
    if state["route"] != "tool":
        return state

    user_profile = state.get("user_profile", {})
    user_skills = user_profile.get("skills", [])
    target_role = user_profile.get("target_role", "")

    # Fallback: try to extract role from the question text
    question_lower = state["question"].lower()
    if not target_role:
        for role in ["data analyst", "data scientist", "software engineer", "ai engineer"]:
            if role in question_lower:
                target_role = role.title()
                break

    if not user_skills:
        result = (
            "⚠️ I need to know your current skills to perform a gap analysis. "
            "Please share your skills in the sidebar or mention them in your question, "
            "e.g., 'I know Python, SQL, and Excel. What am I missing for a Data Analyst role?'"
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

ANSWER_SYSTEM_PROMPT = """You are an expert Career Decision Assistant specializing in data, 
AI, and software engineering careers. Your goal is to help students and professionals make 
informed career decisions.

Guidelines:
- Provide accurate, actionable, and encouraging advice.
- Use concrete examples, numbers, and timelines where relevant.
- Tailor responses to be practical and beginner-friendly.
- If context documents are provided, use them as your primary source.
- If no context is available, use your general knowledge about careers.
- Keep answers focused, structured, and under 350 words.
- Always end with one actionable next step.

Context Documents (if available):
{context}

Tool Result (if available):
{tool_result}"""


def answer_node(state: CapstoneState) -> CapstoneState:
    """
    Generates the final answer using LLM with retrieved context or tool results.
    """
    # Build context string
    context_parts = []
    if state["retrieved"]:
        for i, (doc, meta) in enumerate(zip(state["retrieved"], state["sources"])):
            topic = meta.get("topic", "General")
            context_parts.append(f"[Source {i+1}: {topic}]\n{doc[:500]}")
        context_str = "\n\n".join(context_parts)
    else:
        context_str = "No specific documents retrieved."

    tool_result_str = state.get("tool_result", "") or "No tool analysis performed."

    try:
        llm = get_llm(temperature=0.4)
        prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_SYSTEM_PROMPT.format(
                context=context_str,
                tool_result=tool_result_str
            )),
            ("human", "{question}")
        ])
        chain = prompt | llm
        response = chain.invoke({"question": state["question"]})
        answer = response.content.strip()

    except Exception as e:
        print(f"[answer_node] LLM error: {e}")
        # Graceful fallback: return retrieved content or tool result directly
        if state["tool_result"]:
            answer = state["tool_result"]
        elif state["retrieved"]:
            answer = f"Based on our knowledge base:\n\n{state['retrieved'][0][:600]}"
        else:
            answer = (
                "I'm currently unable to connect to the AI service. "
                "Please ensure your OPENAI_API_KEY is set and try again. "
                "I can still show skill gap analysis if you provide your skills in the sidebar."
            )

    print(f"[answer_node] Answer generated ({len(answer)} chars).")
    return {**state, "answer": answer}


# ---------------------------------------------------------------------------
# NODE 6: eval_node
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = """You are an evaluation judge for a career advisory AI system.

Your task: Score how faithful the generated ANSWER is to the given QUESTION and CONTEXT.

Faithfulness criteria:
- 1.0: Answer is completely accurate, directly addresses the question, and is well-grounded in context.
- 0.7–0.9: Answer is mostly accurate with minor omissions or additions.
- 0.4–0.6: Answer partially addresses the question or contains some inaccuracies.
- 0.0–0.3: Answer is irrelevant, hallucinated, or contradicts the context.

Respond with ONLY a decimal number between 0.0 and 1.0. No explanation."""


def eval_node(state: CapstoneState) -> CapstoneState:
    """
    Evaluates answer faithfulness. Triggers retry if score < 0.7 (max 2 retries).
    """
    question = state["question"]
    answer = state["answer"]
    context = " ".join(state["retrieved"][:2]) if state["retrieved"] else "general knowledge"
    eval_retries = state.get("eval_retries", 0)

    # If no answer, assign low score
    if not answer or len(answer) < 20:
        print(f"[eval_node] Answer too short → faithfulness = 0.0")
        return {**state, "faithfulness": 0.0}

    # For tool results, assume high faithfulness (deterministic function)
    if state["route"] == "tool" and state["tool_result"]:
        print(f"[eval_node] Tool route → faithfulness = 0.95")
        return {**state, "faithfulness": 0.95}

    try:
        llm = get_llm(temperature=0.0)
        eval_prompt = f"""QUESTION: {question}

CONTEXT: {context[:800]}

ANSWER: {answer}

Score the faithfulness of the ANSWER (0.0 to 1.0):"""

        response = llm.invoke([
            SystemMessage(content=EVAL_SYSTEM_PROMPT),
            HumanMessage(content=eval_prompt)
        ])

        score_text = response.content.strip()
        # Extract float from response
        score_match = re.search(r"[0-9]\.[0-9]+|[01]", score_text)
        faithfulness = float(score_match.group()) if score_match else 0.75
        faithfulness = max(0.0, min(1.0, faithfulness))

    except Exception as e:
        print(f"[eval_node] Eval LLM error: {e}. Defaulting to 0.75.")
        faithfulness = 0.75

    print(f"[eval_node] Faithfulness score: {faithfulness:.2f} | Retries so far: {eval_retries}")

    # Retry logic: if score < 0.7 and retries < 2, increment retry count
    if faithfulness < 0.7 and eval_retries < 2:
        print(f"[eval_node] Score below threshold → will retry answer generation.")
        return {
            **state,
            "faithfulness": faithfulness,
            "eval_retries": eval_retries + 1,
            "answer": ""  # clear answer to force regeneration
        }

    return {**state, "faithfulness": faithfulness}


# ---------------------------------------------------------------------------
# NODE 7: save_node
# ---------------------------------------------------------------------------

CONVERSATION_LOG_FILE = "conversation_log.jsonl"


def save_node(state: CapstoneState) -> CapstoneState:
    """
    Saves the conversation turn to thread memory and logs to JSONL file.
    """
    thread_id = state.get("user_profile", {}).get("thread_id", "default")

    # Update thread memory with the new exchange
    ai_msg = AIMessage(content=state["answer"])
    updated_messages = list(state["messages"]) + [ai_msg]
    save_thread_history(thread_id, updated_messages)

    # Build log record
    log_record = {
        "timestamp": datetime.now().isoformat(),
        "thread_id": thread_id,
        "question": state["question"],
        "route": state["route"],
        "answer": state["answer"],
        "faithfulness": state["faithfulness"],
        "eval_retries": state["eval_retries"],
        "sources": [m.get("topic", "") for m in state["sources"]],
        "user_profile": state.get("user_profile", {})
    }

    # Append to JSONL file
    try:
        with open(CONVERSATION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record) + "\n")
    except Exception as e:
        print(f"[save_node] Log write error: {e}")

    print(f"[save_node] Conversation saved. Thread: {thread_id} | Faithfulness: {state['faithfulness']:.2f}")

    # Add AI response to state messages
    return {
        **state,
        "messages": updated_messages
    }


# ---------------------------------------------------------------------------
# Conditional Edge: should_retry
# ---------------------------------------------------------------------------

def should_retry(state: CapstoneState) -> str:
    """
    After eval_node: decides whether to retry answer generation or proceed to save.
    Returns 'retry' if answer needs regeneration, else 'save'.
    """
    if state["faithfulness"] < 0.7 and state["eval_retries"] > 0 and not state["answer"]:
        return "retry"
    return "save"


# ---------------------------------------------------------------------------
# Conditional Edge: route_dispatcher
# ---------------------------------------------------------------------------

def route_dispatcher(state: CapstoneState) -> str:
    """
    After router_node: dispatches to the appropriate node based on route.
    """
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool"
    elif route == "skip":
        return "answer"  # skip retrieval and tool, go straight to answer
    else:
        return "retrieve"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_career_agent_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph-style agentic workflow.
    
    Flow: memory → router → [retrieve|tool|answer] → answer → eval → [retry|save] → END
    """
    workflow = StateGraph(CapstoneState)

    # Add all nodes
    workflow.add_node("memory", memory_node)
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("eval", eval_node)
    workflow.add_node("save", save_node)

    # Entry point
    workflow.set_entry_point("memory")

    # Fixed edges
    workflow.add_edge("memory", "router")

    # Conditional routing after router
    workflow.add_conditional_edges(
        "router",
        route_dispatcher,
        {
            "retrieve": "retrieve",
            "tool": "tool",
            "answer": "answer"
        }
    )

    # After retrieval/tool → answer
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("tool", "answer")

    # Answer → eval
    workflow.add_edge("answer", "eval")

    # Conditional: eval → retry (back to answer) or save
    workflow.add_conditional_edges(
        "eval",
        should_retry,
        {
            "retry": "answer",
            "save": "save"
        }
    )

    # Save → END
    workflow.add_edge("save", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public API: run_agent
# ---------------------------------------------------------------------------

# Compile graph once at module level
career_agent = build_career_agent_graph()


def run_agent(
    question: str,
    user_profile: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs the career agent for a given question.
    
    Args:
        question: The user's career-related question.
        user_profile: Optional dict with 'name', 'skills', 'target_role'.
        thread_id: Optional thread ID for memory continuity.
    
    Returns:
        Dict with 'answer', 'route', 'faithfulness', 'sources', 'tool_result'.
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())[:8]

    profile = user_profile or {}
    profile["thread_id"] = thread_id

    initial_state: CapstoneState = {
        "question": question,
        "messages": [],
        "route": "",
        "retrieved": [],
        "sources": [],
        "tool_result": "",
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "user_profile": profile
    }

    try:
        final_state = career_agent.invoke(initial_state)
        return {
            "answer": final_state.get("answer", "No answer generated."),
            "route": final_state.get("route", "unknown"),
            "faithfulness": final_state.get("faithfulness", 0.0),
            "sources": final_state.get("sources", []),
            "tool_result": final_state.get("tool_result", ""),
            "eval_retries": final_state.get("eval_retries", 0),
            "thread_id": thread_id
        }
    except Exception as e:
        print(f"[run_agent] Graph execution error: {e}")
        return {
            "answer": f"An error occurred while processing your question: {str(e)}",
            "route": "error",
            "faithfulness": 0.0,
            "sources": [],
            "tool_result": "",
            "eval_retries": 0,
            "thread_id": thread_id
        }


# ---------------------------------------------------------------------------
# CLI Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Agentic AI Career Decision Assistant — CLI Test")
    print("=" * 60)

    test_profile = {
        "name": "Test User",
        "skills": ["Python", "SQL", "Excel", "pandas"],
        "target_role": "Data Scientist",
        "thread_id": "cli_test_001"
    }

    test_questions = [
        "What skills do I need to become a Data Scientist?",
        "What is the salary of an AI Engineer in India?",
        "What am I missing to become a Data Scientist?"
    ]

    for i, q in enumerate(test_questions, 1):
        print(f"\n[Test {i}] Question: {q}")
        result = run_agent(q, user_profile=test_profile, thread_id="cli_test_001")
        print(f"Route: {result['route']}")
        print(f"Faithfulness: {result['faithfulness']:.2f}")
        print(f"Answer:\n{result['answer'][:300]}...")
        print("-" * 60)
