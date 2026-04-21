# 🎯 Agentic AI Career Decision Assistant

> A multi-step reasoning system for personalised career path guidance in data, AI, and software engineering.
> **Capstone Project 2025–26 | Academic Year 2025–2026**

---

## 📌 Project Overview

The **Agentic AI Career Decision Assistant** is a production-quality agentic AI system that helps students and early-career professionals make informed decisions about careers in data science, artificial intelligence, and software engineering.

Built on a **LangGraph-style state machine**, it goes beyond conventional chatbots by decomposing every user query into a deliberate sequence of specialised cognitive steps — routing, retrieval, tool execution, answer generation, self-evaluation, and memory management — before delivering a grounded response.

---

## ✨ Key Features

- **Career Q&A via RAG** — Retrieves answers from a curated knowledge base of 10 original career documents using ChromaDB and semantic similarity search
- **Skill Gap Analyser** — A deterministic tool that compares the user's current skills against role requirements and returns a structured gap report with readiness percentage and prioritised learning resources
- **Thread-based Memory** — Maintains contextual multi-turn conversations using a sliding window of the last 6 messages per session thread
- **Self-Evaluation Loop** — An LLM judge scores every answer for faithfulness (0.0–1.0) and triggers automatic regeneration if the score falls below 0.7 (up to 2 retries)
- **Intelligent Router** — Classifies every query into `retrieve`, `tool`, or `skip` using keyword fast-paths and an LLM fallback
- **Professional Streamlit UI** — Dark-themed chat interface with metadata badges (route, faithfulness score, sources) and a fully configured user profile sidebar

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────┐
│ memory_node │  ← Loads thread history (sliding window: 6 messages)
└──────┬──────┘
       ▼
┌─────────────┐
│ router_node │  ← Classifies intent: retrieve | tool | skip
└──────┬──────┘
       │
  ┌────┴──────────────────┐
  ▼                       ▼                  ▼
┌───────────────┐  ┌──────────────┐   (skip → answer)
│ retrieval_node│  │  tool_node   │
│  ChromaDB RAG │  │ skill_gap_   │
│               │  │ analyzer     │
└───────┬───────┘  └──────┬───────┘
        └────────┬─────────┘
                 ▼
        ┌─────────────┐
        │ answer_node │  ← OpenAI GPT-based model generates grounded answer
        └──────┬──────┘
               ▼
        ┌──────────────┐
        │   eval_node  │  ← Faithfulness scoring (0.0–1.0), retry if < 0.7
        └──────┬───────┘
               │
       ┌───────┴──────────┐
       │ (score ≥ 0.7)    │ (score < 0.7, retries < 2)
       ▼                  ▼
  ┌──────────┐       (back to answer_node)
  │ save_node│
  └────┬─────┘
       ▼
      END
```

---

## 📁 Project Structure

```
project/
│
├── agent.py            # Full LangGraph agentic workflow (all 7 nodes)
├── streamlit_app.py    # Streamlit chat interface with dark theme
├── knowledge_base.py   # 10 career documents + ChromaDB vector store setup
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

> **Auto-generated at runtime:**
> - `chroma_career_db/` — ChromaDB persistent vector store
> - `conversation_log.jsonl` — JSONL conversation audit log

---

## 🧩 State Design — `CapstoneState`

| Field | Type | Description |
|-------|------|-------------|
| `question` | `str` | Current user question |
| `messages` | `List[BaseMessage]` | Full conversation history (with LangGraph reducer) |
| `route` | `str` | Router decision: `retrieve` / `tool` / `skip` |
| `retrieved` | `List[str]` | ChromaDB document chunks |
| `sources` | `List[Dict]` | Metadata for retrieved documents |
| `tool_result` | `str` | Output from `skill_gap_analyzer` |
| `answer` | `str` | Final LLM-generated answer |
| `faithfulness` | `float` | Eval score between 0.0 and 1.0 |
| `eval_retries` | `int` | Number of answer retry attempts |
| `user_profile` | `Dict` | Name, skills, target role, thread_id |

---

## 🗂️ Knowledge Base

10 original documents embedded with `all-MiniLM-L6-v2` and stored in ChromaDB:

| # | Document Title | Topic |
|---|---------------|-------|
| 1 | Data Analyst Career Overview | Career overview |
| 2 | Data Scientist Career Overview | Career overview |
| 3 | Software Engineer Career Overview | Career overview |
| 4 | AI Engineer Career Overview | Career overview |
| 5 | Skills Required for Data and AI Roles | Skills guide |
| 6 | Learning Roadmaps for Beginners | Learning path |
| 7 | Salary Insights Across Data and AI Careers | Compensation |
| 8 | Tools and Technologies for Professionals | Tools guide |
| 9 | Interview Preparation for Data and AI Roles | Interview guide |
| 10 | Common Mistakes Students Make | Career advice |

**Embedding Model:** `all-MiniLM-L6-v2` (sentence-transformers)
**Vector Store:** ChromaDB (cosine similarity, persistent)

---

## 🔧 Custom Tool: `skill_gap_analyzer`

```python
skill_gap_analyzer(user_skills: List[str], target_role: str) -> str
```

**Input:** User's current skills list + target job role
**Output:** Structured gap report containing:
- ✅ Skills already acquired
- ❌ Missing skills
- 📈 Role readiness percentage
- 🎯 Top 5 priority skills to learn next (with specific resources)
- 💡 Estimated timeline to bridge the gap

**Supported Roles:** Data Analyst, Data Scientist, Software Engineer, AI Engineer
**Design:** Fully deterministic — never crashes, always returns a meaningful string

---

## 🔁 Router Logic

| Query Type | Route | Example |
|-----------|-------|---------|
| Career questions | `retrieve` | "What salary does a Data Scientist earn?" |
| Skill comparison | `tool` | "What skills am I missing for AI Engineer?" |
| General conversation | `skip` | "Hello!", "Thanks!" |

---

## ⚖️ Evaluation Node

- Scores answer faithfulness from 0.0 to 1.0 using an LLM judge
- Threshold: **0.7**
- If score < 0.7 → retry answer generation (maximum **2 retries**)
- Tool-route answers are automatically scored **0.95** (deterministic output)

---

## 🚀 Quick Start

### 1. Set Up Project Directory

```bash
mkdir career_agent && cd career_agent
# Place all project files here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~85 MB) on first run.

### 3. Configure OpenAI API Key

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-your-key-here"

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-your-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-your-key-here"
```

> **Without an API key:** The Streamlit app still functions for skill gap analysis (deterministic tool). LLM-generated answers will display a graceful fallback message.

### 4. Build the Knowledge Base

```bash
python knowledge_base.py
```

Expected output:
```
Building Career Knowledge Base...
[knowledge_base] Created collection 'career_knowledge_base' with 10 documents.
Knowledge base ready with 10 documents.
```

### 5. Launch the Streamlit Application

```bash
streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501`

### 6. (Optional) Run CLI Test

```bash
python agent.py
```

---

## 🧪 Sample Questions

| # | Question | Expected Route |
|---|----------|---------------|
| 1 | "What does a Data Analyst do?" | `retrieve` |
| 2 | "How much does a Data Scientist earn in India?" | `retrieve` |
| 3 | "What tools should I learn for AI Engineering?" | `retrieve` |
| 4 | "Give me a roadmap to become a Software Engineer" | `retrieve` |
| 5 | "What are common mistakes students make in their career?" | `retrieve` |
| 6 | "What skills am I missing for a Data Scientist role?" | `tool` |
| 7 | "Analyse my skill gap for AI Engineer" | `tool` |
| 8 | "Compare my skills with Data Analyst requirements" | `tool` |
| 9 | "Hello! What can you help me with?" | `skip` |
| 10 | "How do I prepare for a Data Science interview?" | `retrieve` |

---

## 🖥️ Streamlit UI Features

- **Dark theme** with Space Mono + DM Sans typography
- **Chat interface** with styled message bubbles
- **Sidebar controls:** API key input, user profile (name, skills, target role), quick skill gap analysis, session statistics, new conversation reset, 5 sample question shortcuts
- **Message metadata badges:** processing route, faithfulness score, knowledge sources
- **Graceful degradation:** functional without an API key for tool-based queries

---

## 📋 Technical Specifications

| Component | Technology |
|-----------|-----------|
| Agentic Framework | LangGraph (StateGraph) |
| LLM | OpenAI GPT-based models |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| Memory | Thread-based in-memory (sliding window: 6 messages) |
| UI | Streamlit 1.35+ |
| State Management | TypedDict (`CapstoneState`) |
| Logging | JSONL flat file (`conversation_log.jsonl`) |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (for LLM features) | OpenAI API key for answer generation and evaluation |

---

## 🎓 Academic Notes

This project demonstrates the following concepts:

1. **Agentic AI** — Multi-step reasoning with a structured conditional state machine
2. **Retrieval-Augmented Generation (RAG)** — ChromaDB + semantic similarity search for grounded answers
3. **Tool Use** — Custom deterministic tool integrated as a first-class node in the agentic loop
4. **Self-Evaluation** — Faithfulness scoring with automatic retry mechanism (metacognition)
5. **Memory Management** — Thread-based persistent context with sliding window policy
6. **Production Patterns** — Graceful error handling, JSONL audit logging, modular node design

---

## 📸 Screenshots

Screenshots of the running application are included in the project documentation PDF:

1. **Streamlit Chat Interface** — Main chat window with sidebar configuration panel
2. **Skill Gap Analysis Output** — Structured gap report with readiness percentage
3. **Chat Response with Metadata** — AI response annotated with route, faithfulness score, and knowledge sources

---

*Agentic AI Career Decision Assistant — Capstone Project 2025–26*
*Submitted by Biswaranjan Panda | Roll No. 23052961 | Batch 2023–2027 | Academic Year 2025–2026*
