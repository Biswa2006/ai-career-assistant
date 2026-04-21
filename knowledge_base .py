"""
knowledge_base.py
-----------------
Agentic AI Career Decision Assistant — Capstone Project 2025–26
Academic Year 2025–2026

Builds and manages the ChromaDB persistent vector store containing
10 original career guidance documents.

Embedding model : all-MiniLM-L6-v2 (sentence-transformers, 384-dim)
Vector store    : ChromaDB with cosine similarity (persistent, local)
Collection name : career_knowledge_base

Usage:
    # Build from CLI (force-rebuild wipes existing collection)
    python knowledge_base.py

    # Import in other modules
    from knowledge_base import build_knowledge_base, retrieve_documents
"""

import os
import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_PATH     = "./chroma_career_db"
COLLECTION_NAME = "career_knowledge_base"


# ---------------------------------------------------------------------------
# Career Knowledge Documents (10 original documents)
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {
        "id":    "doc_001",
        "title": "Data Analyst Career Overview",
        "content": (
            "A Data Analyst is a professional who collects, processes, and performs statistical "
            "analyses on large datasets to help organisations make informed decisions. Data "
            "Analysts work across industries including finance, healthcare, retail, and "
            "technology. Their primary responsibility is to transform raw data into meaningful "
            "insights using visualisation tools and statistical techniques.\n\n"
            "Daily tasks typically include writing SQL queries, building dashboards in Tableau "
            "or Power BI, preparing stakeholder reports, and collaborating with business teams "
            "to identify trends. Strong communication skills are essential because analysts must "
            "explain complex findings to non-technical audiences.\n\n"
            "Entry-level roles require proficiency in Excel, SQL, and at least one visualisation "
            "tool. Python or R is a strong differentiator. Analysts work primarily with structured "
            "data in relational databases and must have a solid grasp of descriptive statistics.\n\n"
            "Career progression: Junior Analyst → Senior Analyst → Lead Analyst → Analytics "
            "Manager, or a lateral move into Data Science. Starting salaries in India range from "
            "₹3.5–6 LPA for freshers, scaling to ₹12–20 LPA for senior professionals. In the "
            "US, entry-level salaries are $55,000–$75,000. The role offers strong job stability "
            "and clear upward mobility, making it an ideal entry point for data-driven careers."
        ),
        "metadata": {"topic": "Data Analyst", "type": "career_overview"},
    },
    {
        "id":    "doc_002",
        "title": "Data Scientist Career Overview",
        "content": (
            "A Data Scientist is a senior data professional who applies advanced statistical "
            "modelling, machine learning, and deep learning to solve complex business problems. "
            "Unlike Data Analysts who focus on reporting and visualisation, Data Scientists "
            "build predictive models that forecast outcomes, automate decisions, and discover "
            "hidden patterns in data.\n\n"
            "The role demands expertise in Python or R, machine learning frameworks such as "
            "Scikit-learn, TensorFlow, or PyTorch, and a strong foundation in linear algebra, "
            "calculus, and probability theory. Solid SQL skills and familiarity with big data "
            "tools like Spark or Hadoop are also expected.\n\n"
            "A typical Data Scientist workflow involves problem framing, data collection and "
            "cleaning, exploratory analysis, feature engineering, model building, evaluation, "
            "and deployment. Close collaboration with engineers, product managers, and "
            "stakeholders is central to the role.\n\n"
            "Career path: Junior Data Scientist → Data Scientist → Senior → Principal Scientist "
            "→ Director of Data Science. Salaries in India range from ₹6–12 LPA at entry level "
            "to ₹30+ LPA for experienced professionals at product companies. In the US, salaries "
            "typically fall between $95,000–$150,000. A strong project portfolio and Kaggle "
            "competition experience significantly improve hiring prospects."
        ),
        "metadata": {"topic": "Data Scientist", "type": "career_overview"},
    },
    {
        "id":    "doc_003",
        "title": "Software Engineer Career Overview",
        "content": (
            "Software Engineering is one of the broadest and most universally demanded careers "
            "in the technology industry. Software Engineers design, develop, test, and maintain "
            "software systems ranging from mobile applications and web platforms to embedded "
            "systems and operating systems.\n\n"
            "Specialisations include frontend engineering (React, Vue, Angular), backend "
            "engineering (Node.js, Django, Spring Boot), full-stack development, mobile "
            "development (iOS/Android), DevOps and infrastructure, and embedded systems. Each "
            "path requires a distinct but overlapping skill set.\n\n"
            "Core skills include proficiency in at least one programming language (Python, Java, "
            "JavaScript, C++), data structures and algorithms, version control with Git, and "
            "system design principles. Problem-solving ability and the capacity to write clean, "
            "maintainable code are critical traits.\n\n"
            "Career growth: Junior Developer → Software Engineer → Senior Engineer → "
            "Staff/Principal Engineer → Engineering Manager or Architect. In India, freshers "
            "earn ₹4–10 LPA at service companies and ₹12–30 LPA at product companies. In the "
            "US, entry-level engineers earn $90,000–$130,000, and senior engineers at top firms "
            "exceed $200,000 including equity. Global demand continues to grow."
        ),
        "metadata": {"topic": "Software Engineer", "type": "career_overview"},
    },
    {
        "id":    "doc_004",
        "title": "AI Engineer Career Overview",
        "content": (
            "An AI Engineer sits at the intersection of software engineering and data science, "
            "specialising in building, deploying, and maintaining AI-powered systems in "
            "production environments. Unlike Data Scientists who focus primarily on research, "
            "AI Engineers make AI solutions scalable, reliable, and integrated into real-world "
            "applications.\n\n"
            "AI Engineers work with large language models (LLMs), computer vision pipelines, "
            "recommendation systems, and conversational AI. They use LangChain, LlamaIndex, "
            "Hugging Face Transformers, TensorFlow Serving, and ONNX to productionise models. "
            "Cloud platform knowledge (AWS SageMaker, Azure ML, GCP Vertex AI) is essential.\n\n"
            "Required skills: Python, RESTful API design, containerisation (Docker, Kubernetes), "
            "MLOps, vector databases (Pinecone, ChromaDB, Weaviate), and prompt engineering. "
            "Retrieval-Augmented Generation (RAG), agentic AI frameworks (LangGraph, AutoGen), "
            "and fine-tuning are increasingly important.\n\n"
            "Career path: Junior AI Engineer → AI Engineer → Senior AI Engineer → AI Architect "
            "→ Head of AI. Salaries in India range from ₹8–15 LPA at entry level and reach "
            "₹40–60 LPA at MAANG or AI-native companies. In the US, salaries range from "
            "$120,000–$200,000+, making this one of the highest-paying roles in tech."
        ),
        "metadata": {"topic": "AI Engineer", "type": "career_overview"},
    },
    {
        "id":    "doc_005",
        "title": "Skills Required for Data and AI Roles",
        "content": (
            "Choosing the right career in data and AI begins with understanding role-specific "
            "skill requirements. While there is significant overlap, each position demands a "
            "unique combination of technical and soft skills.\n\n"
            "Data Analysts need: SQL (advanced querying, window functions), Excel, Python "
            "(pandas, matplotlib), Tableau or Power BI, statistical concepts (mean, variance, "
            "hypothesis testing), and business communication.\n\n"
            "Data Scientists need: Python (Scikit-learn, TensorFlow, PyTorch), R, linear "
            "algebra, calculus, probability theory, machine learning algorithms, feature "
            "engineering, model evaluation, and data storytelling.\n\n"
            "Software Engineers need: Data structures and algorithms, object-oriented "
            "programming, system design, Git, testing (unit/integration), CI/CD pipelines, "
            "and cloud fundamentals.\n\n"
            "AI Engineers need all of the above, plus LLM orchestration (LangChain, LangGraph), "
            "vector databases, RAG architectures, prompt engineering, model fine-tuning, MLOps, "
            "Docker, Kubernetes, and production AI monitoring.\n\n"
            "Regardless of role, soft skills are critical: problem-solving mindset, translating "
            "technical findings for stakeholders, collaboration, and continuous learning. A "
            "practical approach is to build Python and SQL as your universal foundation, then "
            "branch into your chosen specialisation."
        ),
        "metadata": {"topic": "Skills", "type": "skills_guide"},
    },
    {
        "id":    "doc_006",
        "title": "Learning Roadmaps for Beginners",
        "content": (
            "Starting a career in data or AI can feel overwhelming without a structured plan. "
            "Here are proven roadmaps for each field.\n\n"
            "Data Analyst (6–9 months): Months 1–2: Excel and SQL basics. Months 3–4: Python "
            "fundamentals and pandas. Months 5–6: Tableau or Power BI. Months 7–9: Statistics, "
            "real-world projects, and job applications. Build a portfolio of 3–5 dashboards "
            "and analysis projects on GitHub.\n\n"
            "Data Scientist (12–18 months): Python and SQL (3 months) → mathematics: linear "
            "algebra, statistics, calculus (2 months) → machine learning with Scikit-learn "
            "(3 months) → deep learning with TensorFlow or PyTorch (3 months) → end-to-end "
            "projects and Kaggle competitions (3 months).\n\n"
            "Software Engineer (9–12 months): Master one language (Python or JavaScript), "
            "data structures and algorithms on LeetCode, system design concepts, 3–5 full-stack "
            "or backend projects, and open-source contributions on GitHub.\n\n"
            "AI Engineer (18–24 months): Complete the Data Scientist path first. Then learn "
            "LangChain, LlamaIndex, and LangGraph → vector databases and RAG systems → Docker "
            "and Kubernetes → MLOps with MLflow or Weights & Biases → deploy LLM-powered "
            "applications to production.\n\n"
            "Recommended resources for all paths: Python.org, fast.ai, Coursera (Andrew Ng), "
            "StatQuest (YouTube), roadmap.sh, and Towards Data Science on Medium. Consistency "
            "matters more than speed: 2 hours of focused daily study outperforms weekend cramming."
        ),
        "metadata": {"topic": "Roadmaps", "type": "learning_path"},
    },
    {
        "id":    "doc_007",
        "title": "Salary Insights Across Data and AI Careers",
        "content": (
            "Compensation in data and AI varies significantly by role, experience, company "
            "size, and location. Understanding benchmarks helps set realistic expectations.\n\n"
            "India (₹ LPA — Lakhs Per Annum):\n"
            "- Data Analyst:      Fresher ₹3.5–6 | Mid ₹8–14 | Senior ₹15–25\n"
            "- Data Scientist:    Fresher ₹6–10  | Mid ₹15–25 | Senior ₹28–45\n"
            "- Software Engineer: Fresher ₹4–12  | Mid ₹18–35 | Senior ₹40–80\n"
            "- AI Engineer:       Fresher ₹8–15  | Mid ₹20–40 | Senior ₹45–80\n\n"
            "United States (USD Annual):\n"
            "- Data Analyst:      Entry $55K–$75K    | Mid $80K–$105K  | Senior $110K–$140K\n"
            "- Data Scientist:    Entry $95K–$115K   | Mid $120K–$155K | Senior $160K–$210K\n"
            "- Software Engineer: Entry $95K–$130K   | Mid $140K–$180K | Senior $190K–$280K\n"
            "- AI Engineer:       Entry $115K–$145K  | Mid $155K–$200K | Senior $210K–$350K+\n\n"
            "Factors that increase compensation: advanced degree (MS/PhD adds 20–40%), "
            "employment at MAANG or unicorn companies (2–5x higher), niche specialisations "
            "(LLM fine-tuning, computer vision), and open-source or research contributions.\n\n"
            "Freshers should also evaluate learning opportunities, mentorship quality, and "
            "long-term growth potential — not base salary alone — when choosing their first role."
        ),
        "metadata": {"topic": "Salary", "type": "compensation_guide"},
    },
    {
        "id":    "doc_008",
        "title": "Tools and Technologies for Data and AI Professionals",
        "content": (
            "Mastering the right tools is crucial for productivity and employability in data "
            "and AI careers.\n\n"
            "Data Analyst Tools: SQL (PostgreSQL, MySQL, BigQuery), Excel/Google Sheets, "
            "Python (pandas, matplotlib, seaborn), Tableau, Power BI, Looker, Jupyter Notebooks.\n\n"
            "Data Scientist Tools: Python ecosystem (NumPy, pandas, Scikit-learn, XGBoost, "
            "LightGBM), TensorFlow, PyTorch, Keras, Jupyter Lab, MLflow, DVC, Weights & Biases, "
            "Google Colab.\n\n"
            "Software Engineer Tools: Git and GitHub/GitLab, VS Code or IntelliJ, Docker, "
            "Kubernetes, CI/CD tools (GitHub Actions, Jenkins, CircleCI), Postman, Linux/Bash, "
            "AWS/Azure/GCP cloud platforms.\n\n"
            "AI Engineer Tools: LangChain, LangGraph, LlamaIndex, OpenAI API, Hugging Face "
            "Transformers, ChromaDB, Pinecone, Weaviate, Ollama, FastAPI, Ray, Triton Inference "
            "Server, ONNX.\n\n"
            "Cross-role essentials: Git, Jupyter Notebooks, Linux CLI, REST APIs, and SQL are "
            "universal requirements across all data and AI roles.\n\n"
            "Students should prioritise depth over breadth: genuine proficiency in 3–4 core "
            "tools is more valuable to employers than a long list of superficial familiarity. "
            "Portfolio projects that demonstrate real tool usage are far more impactful than "
            "simply listing technologies on a resume."
        ),
        "metadata": {"topic": "Tools", "type": "tools_guide"},
    },
    {
        "id":    "doc_009",
        "title": "Interview Preparation for Data and AI Roles",
        "content": (
            "Interview preparation strategy varies by role, but certain fundamentals apply "
            "across all data and AI positions.\n\n"
            "Data Analyst Interviews: SQL challenges (complex JOINs, window functions, CTEs), "
            "case studies (business problem → data-driven solution), Excel/Python tasks, and "
            "behavioural questions. Practice on StrataScratch and Mode Analytics.\n\n"
            "Data Scientist Interviews: Three pillars — ML Theory (bias-variance tradeoff, "
            "regularisation, cross-validation), Coding (Python for data manipulation and ML), "
            "and Case Studies (end-to-end problem solving). Study Andrew Ng's ML course and "
            "practice on Kaggle and LeetCode.\n\n"
            "Software Engineer Interviews: Focus heavily on Data Structures and Algorithms "
            "(DSA). Practice 200+ LeetCode problems. System design interviews test scalable "
            "architecture — study 'System Design Interview' by Alex Xu. Behavioural questions "
            "use the STAR format.\n\n"
            "AI Engineer Interviews: LLM knowledge (transformer architecture, attention "
            "mechanism, fine-tuning vs RAG), practical coding (building chains, agents, RAG "
            "pipelines), and MLOps concepts including latency, hallucination mitigation, and "
            "evaluation frameworks.\n\n"
            "Universal tips: Review resume projects in detail (interviewers ask follow-up "
            "questions), research the company's tech stack, practise explaining technical "
            "concepts to non-specialists, and always ask thoughtful questions at the end. "
            "Mock interviews on Pramp or Interviewing.io significantly improve performance."
        ),
        "metadata": {"topic": "Interview Prep", "type": "interview_guide"},
    },
    {
        "id":    "doc_010",
        "title": "Common Mistakes Students Make When Choosing Data and AI Careers",
        "content": (
            "Students entering the data and AI field often make avoidable mistakes that slow "
            "their progress. Awareness of these pitfalls can save months of wasted effort.\n\n"
            "Mistake 1 — Tutorial Hell: Watching dozens of courses without building real "
            "projects. Consuming content feels productive but does not substitute for hands-on "
            "experience. For every tutorial completed, build an original project from scratch.\n\n"
            "Mistake 2 — Skipping Mathematics: Jumping into neural network libraries without "
            "understanding the underlying math creates fragile knowledge. Invest time in linear "
            "algebra, statistics, and calculus basics before advanced ML frameworks.\n\n"
            "Mistake 3 — Chasing Trends Blindly: Switching focus to every new model or "
            "framework without mastering fundamentals. Core principles are stable; adopt new "
            "tools only when you have a concrete use case.\n\n"
            "Mistake 4 — Neglecting Soft Skills: Assuming technical excellence alone guarantees "
            "success. Data insights have no value if they cannot be communicated effectively. "
            "Practise presenting projects to non-technical audiences regularly.\n\n"
            "Mistake 5 — Weak Portfolios: Applying for jobs with only course certificates and "
            "no real projects. Build 3–5 projects that solve genuine problems, document them "
            "professionally on GitHub, and deploy them publicly.\n\n"
            "Mistake 6 — Ignoring Networking: Learning in isolation without engaging with "
            "communities on LinkedIn, Kaggle, GitHub, or local meetups. Professional networks "
            "accelerate job searches significantly. Share your learning journey and connect "
            "with industry professionals consistently."
        ),
        "metadata": {"topic": "Common Mistakes", "type": "career_advice"},
    },
]


# ---------------------------------------------------------------------------
# Embedding Function
# ---------------------------------------------------------------------------

def get_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """
    Returns the sentence-transformer embedding function for ChromaDB.

    Uses the all-MiniLM-L6-v2 model which produces 384-dimensional vectors
    and offers an excellent balance of semantic quality and inference speed.
    The model (~85 MB) is downloaded automatically on first use.

    Returns:
        SentenceTransformerEmbeddingFunction configured for all-MiniLM-L6-v2.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


# ---------------------------------------------------------------------------
# Build / Load Knowledge Base
# ---------------------------------------------------------------------------

def build_knowledge_base(force_rebuild: bool = False) -> chromadb.Collection:
    """
    Builds or loads the ChromaDB collection containing all career documents.

    On first run, creates a persistent collection at CHROMA_PATH, embeds all
    documents using all-MiniLM-L6-v2, and stores them with cosine similarity.
    On subsequent runs, loads the existing collection unless force_rebuild=True.

    Args:
        force_rebuild: If True, deletes the existing collection and rebuilds
                       from scratch. Useful for updating document content.

    Returns:
        ChromaDB Collection object ready for querying.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = get_embedding_function()

    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing:
        if force_rebuild:
            client.delete_collection(COLLECTION_NAME)
            print(f"[knowledge_base] Deleted existing collection '{COLLECTION_NAME}'.")
        else:
            collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
            )
            print(
                f"[knowledge_base] Loaded existing collection "
                f"with {collection.count()} documents."
            )
            return collection

    # Create new collection with cosine similarity metric
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS],
        documents=[doc["content"] for doc in DOCUMENTS],
        metadatas=[doc["metadata"] for doc in DOCUMENTS],
    )

    print(
        f"[knowledge_base] Created collection '{COLLECTION_NAME}' "
        f"with {len(DOCUMENTS)} documents."
    )
    return collection


# ---------------------------------------------------------------------------
# Document Retrieval
# ---------------------------------------------------------------------------

def retrieve_documents(query: str, n_results: int = 3) -> dict:
    """
    Retrieves the top-n most semantically relevant documents for a query.

    Uses cosine similarity between the query embedding and stored document
    embeddings to rank results. The ChromaDB collection must exist before
    calling this function (run build_knowledge_base first).

    Args:
        query:    The natural language search query.
        n_results: Number of documents to retrieve (capped at collection size).

    Returns:
        Dict with keys:
            'documents' -- List of document text strings.
            'metadatas' -- List of metadata dicts for each document.
            'distances' -- List of cosine distance scores (lower = more similar).
    """
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    ef         = get_embedding_function()
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_all_topics() -> list:
    """Returns the list of topic strings for all documents in the knowledge base."""
    return [doc["metadata"]["topic"] for doc in DOCUMENTS]


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building Career Knowledge Base...")
    print(f"Embedding model : all-MiniLM-L6-v2")
    print(f"Storage path    : {os.path.abspath(CHROMA_PATH)}\n")

    collection = build_knowledge_base(force_rebuild=True)
    print(f"\nKnowledge base ready with {collection.count()} documents.")

    print("\nAvailable topics:")
    for topic in get_all_topics():
        print(f"  - {topic}")

    # Quick retrieval test
    print("\n--- Test Retrieval: 'How to become a Data Scientist' ---")
    results = retrieve_documents("How to become a Data Scientist", n_results=2)
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        print(f"\n[Result {i + 1}] Topic: {meta['topic']}")
        print(doc[:200] + "...")
