# 🎯 Agentic AI Career Decision Assistant
### Capstone Project — LangGraph-Style Agentic AI System

---

## 📌 Project Overview

The **Agentic AI Career Decision Assistant** is a multi-step reasoning AI system that helps students and early professionals make informed career decisions in the data, AI, and software engineering domain.

Built using a **LangGraph-style agentic workflow**, it goes beyond simple chatbots by decomposing user queries into a structured pipeline of specialized nodes — each responsible for a distinct cognitive task.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────┐
│ memory_node │  ← Loads thread history (sliding window: 6 msgs)
└──────┬──────┘
       ▼
┌─────────────┐
│ router_node │  ← Classifies query: retrieve | tool | skip
└──────┬──────┘
       │
  ┌────┴────────────────────┐
  │                         │                  │
  ▼                         ▼                  ▼
┌───────────────┐  ┌─────────────────┐  (skip → answer)
│ retrieval_node│  │   tool_node     │
│ ChromaDB RAG  │  │ skill_gap_tool  │
└───────┬───────┘  └────────┬────────┘
        └─────────┬──────────┘
                  ▼
         ┌─────────────┐
         │ answer_node │  ← LLM generates answer with context
         └──────┬──────┘
                ▼
         ┌──────────────┐
         │   eval_node  │  ← Faithfulness scoring (0–1)
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
career_agent/
│
├── agent.py            # Full LangGraph agentic workflow (all 7 nodes)
├── streamlit_app.py    # Streamlit chat interface with dark theme
├── knowledge_base.py   # 10 career documents + ChromaDB setup
├── requirements.txt    # Python dependencies
├── README.md           # This file
│
├── chroma_career_db/   # Auto-created: ChromaDB vector store
└── conversation_log.jsonl  # Auto-created: conversation audit log
```

---

## 🧩 State Design — `CapstoneState`

| Field | Type | Description |
|-------|------|-------------|
| `question` | str | Current user question |
| `messages` | List[BaseMessage] | Full conversation history (with reducer) |
| `route` | str | Router decision: `retrieve` / `tool` / `skip` |
| `retrieved` | List[str] | ChromaDB document chunks |
| `sources` | List[Dict] | Metadata for retrieved docs |
| `tool_result` | str | Output from `skill_gap_analyzer` |
| `answer` | str | Final LLM-generated answer |
| `faithfulness` | float | Eval score between 0.0 and 1.0 |
| `eval_retries` | int | Number of answer retry attempts |
| `user_profile` | Dict | Name, skills, target role, thread_id |

---

## 🗂️ Knowledge Base

10 original documents covering:

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

**Embedding Model**: `all-MiniLM-L6-v2` (sentence-transformers)  
**Vector Store**: ChromaDB (cosine similarity)

---

## 🔧 Custom Tool: `skill_gap_analyzer`

```python
skill_gap_analyzer(user_skills: List[str], target_role: str) -> str
```

- **Input**: User's current skills + target job role
- **Output**: Formatted string with:
  - ✅ Skills you already have
  - ❌ Missing skills
  - 📈 Role readiness percentage
  - 🎯 Priority skills to learn next (with resources)
  - 💡 Estimated timeline to bridge the gap
- **Supported Roles**: Data Analyst, Data Scientist, Software Engineer, AI Engineer
- **Guarantees**: Never crashes, always returns a string

---

## 🔁 Router Logic

| Query Type | Route | Example |
|-----------|-------|---------|
| Career questions | `retrieve` | "What salary does a Data Scientist earn?" |
| Skill comparison | `tool` | "What skills am I missing for AI Engineer?" |
| General chat | `skip` | "Hello!", "Thanks!" |

---

## ⚖️ Evaluation Node

- Scores answer faithfulness from 0.0 to 1.0 using an LLM judge
- Threshold: **0.7**
- If score < 0.7 → retry answer generation (max **2 retries**)
- Tool route answers are auto-scored **0.95** (deterministic function)

---

## 🚀 Quick Start

### 1. Clone / Download

```bash
# Place all files in a directory
mkdir career_agent && cd career_agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~85MB) on first run.

### 3. Set OpenAI API Key

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-your-key-here"

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-your-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-your-key-here"
```

> **Without API key**: The Streamlit app still works for skill gap analysis (deterministic tool). LLM-generated answers will show a graceful fallback.

### 4. Build Knowledge Base

```bash
python knowledge_base.py
```

Expected output:
```
Building Career Knowledge Base...
[knowledge_base] Created collection 'career_knowledge_base' with 10 documents.
Knowledge base ready with 10 documents.
```

### 5. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501`

### 6. (Optional) Test Agent CLI

```bash
python agent.py
```

---

## 🧪 Test Questions & Sample Outputs

| # | Question | Expected Route |
|---|----------|---------------|
| 1 | "What does a Data Analyst do?" | `retrieve` |
| 2 | "How much does a Data Scientist earn in India?" | `retrieve` |
| 3 | "What tools should I learn for AI Engineering?" | `retrieve` |
| 4 | "Give me a roadmap to become a Software Engineer" | `retrieve` |
| 5 | "What are common mistakes students make in their career?" | `retrieve` |
| 6 | "What skills am I missing for a Data Scientist role?" | `tool` |
| 7 | "Analyze my skill gap for AI Engineer" | `tool` |
| 8 | "Compare my skills with Data Analyst requirements" | `tool` |
| 9 | "Hello! What can you help me with?" | `skip` |
| 10 | "How do I prepare for a Data Science interview?" | `retrieve` |

### Sample Output 1 — Career Question (retrieve route)

**Question**: "What salary can I expect as an AI Engineer in India?"

**Answer**:
```
AI Engineers are among the highest-paid professionals in India's tech industry.

Entry Level (0–2 years): ₹8–15 LPA
Mid Level (3–5 years): ₹20–40 LPA
Senior Level (6+ years): ₹45–80 LPA

Key factors that increase your compensation:
• Working at product companies vs. service companies (2–5x difference)
• Specializing in high-demand areas: LLM fine-tuning, RAG systems, computer vision
• Strong open-source contributions or research publications
• Advanced degree (MS/PhD adds 20–40% premium)

🎯 Next Step: Build 2–3 production-ready AI projects and deploy them publicly.
This differentiates you from candidates with only course certificates.
```
**Faithfulness**: 0.92 | **Route**: retrieve | **Sources**: AI Engineer, Salary Insights

---

### Sample Output 2 — Skill Gap Analysis (tool route)

**Question**: "What skills am I missing for a Data Scientist role?"  
**User Skills**: Python, SQL, Excel, pandas

**Answer**:
```
📊 Skill Gap Analysis for: Data Scientist
─────────────────────────────────────────────
✅ Skills You Have (3/16): Python, SQL, pandas

❌ Missing Skills (13): Machine Learning, Scikit-learn, TensorFlow,
PyTorch, Linear Algebra, Statistics, Feature Engineering...

📈 Role Readiness: 19%

🎯 Top Priority Skills to Learn Next:
• Machine Learning: Andrew Ng's ML Specialization on Coursera.
• Statistics: Khan Academy Statistics + StatQuest YouTube channel.
• Scikit-learn: Official documentation with hands-on exercises.

💡 Recommended Timeline: 6–12 months of focused study to bridge these gaps.
```
**Faithfulness**: 0.95 | **Route**: tool

---

## 🖥️ Streamlit UI Features

- **Dark theme** with Space Mono + DM Sans typography
- **Chat interface** with styled message bubbles
- **Sidebar** with:
  - OpenAI API key input
  - User profile (name, skills, target role)
  - Quick skill gap analysis button
  - Session statistics (query count, average faithfulness)
  - "New Conversation" button (resets thread)
  - 5 sample question buttons
  - Architecture information
- **Message metadata badges**: route, faithfulness score, sources
- **Prefill support**: clicking sidebar questions auto-fills chat input

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (for LLM) | OpenAI API key for answer generation and evaluation |

---

## 📋 Technical Specifications

| Component | Technology |
|-----------|-----------|
| Agentic Framework | LangGraph (StateGraph) |
| LLM | GPT-3.5-Turbo (OpenAI) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| Memory | Thread-based in-memory (sliding window: 6 msgs) |
| UI | Streamlit 1.35+ |
| State Management | TypedDict (CapstoneState) |
| Logging | JSONL flat file (conversation_log.jsonl) |

---

## 🎓 Academic Notes

This project demonstrates:
1. **Agentic AI**: Multi-step reasoning with a structured state machine
2. **RAG (Retrieval-Augmented Generation)**: ChromaDB + semantic search
3. **Tool Use**: Custom deterministic tool integrated into agentic loop
4. **Self-Evaluation**: Faithfulness scoring with retry mechanism
5. **Memory Management**: Thread-based persistent context
6. **Production Patterns**: Graceful error handling, logging, modular design

---

*Agentic AI Career Decision Assistant — Capstone Project 2025*
