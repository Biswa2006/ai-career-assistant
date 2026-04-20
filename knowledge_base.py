"""
knowledge_base.py
Agentic AI Career Decision Assistant
Builds and manages the ChromaDB vector store with 10 unique career documents.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# 10 Unique Career Domain Documents (100-300 words each)
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Data Analyst Career Overview",
        "content": """
A Data Analyst is a professional who collects, processes, and performs statistical analyses 
on large datasets to help organizations make informed decisions. Data Analysts work across 
industries including finance, healthcare, retail, and technology. Their primary responsibility 
is to transform raw data into meaningful insights using visualization tools and statistical 
techniques.

Daily tasks typically include writing SQL queries, building dashboards in tools like Tableau 
or Power BI, preparing reports for stakeholders, and collaborating with business teams to 
identify trends. Strong communication skills are essential because analysts must explain 
complex findings to non-technical audiences.

Entry-level Data Analyst roles typically require proficiency in Excel, SQL, and at least one 
visualization tool. Python or R knowledge is a strong plus. Analysts often work with structured 
data in relational databases and must have a solid understanding of statistics.

The career path for a Data Analyst usually progresses from Junior Analyst → Senior Analyst → 
Lead Analyst → Analytics Manager or Analyst transitions into Data Science. Companies in every 
sector hire Data Analysts, making this one of the most in-demand roles in the data domain.

Starting salaries in India range from ₹3.5–6 LPA for freshers, scaling to ₹12–20 LPA for 
senior professionals. In the US, entry-level salaries start at $55,000–$75,000. The role 
offers excellent job stability and clear upward mobility, making it an ideal starting point 
for students interested in data-driven careers.
        """.strip(),
        "metadata": {"topic": "Data Analyst", "type": "career_overview"}
    },
    {
        "id": "doc_002",
        "title": "Data Scientist Career Overview",
        "content": """
A Data Scientist is a senior data professional who applies advanced statistical modeling, 
machine learning, and deep learning to solve complex business problems. Unlike Data Analysts 
who focus on reporting and visualization, Data Scientists build predictive models that can 
forecast outcomes, automate decisions, and discover hidden patterns in data.

The role demands expertise in Python or R, machine learning frameworks like Scikit-learn, 
TensorFlow, or PyTorch, and a strong foundation in mathematics — particularly linear algebra, 
calculus, and probability theory. Data Scientists also need solid SQL skills and experience 
with big data tools like Spark or Hadoop.

A typical Data Scientist workflow involves problem framing, data collection and cleaning, 
exploratory data analysis, feature engineering, model building, evaluation, and deployment. 
They collaborate closely with engineers, product managers, and business stakeholders.

Career progression often follows the path: Junior Data Scientist → Data Scientist → Senior 
Data Scientist → Principal Scientist → Director of Data Science. Some Data Scientists 
specialize in NLP, computer vision, or reinforcement learning.

Salaries for Data Scientists in India range from ₹6–12 LPA at entry level and can exceed 
₹30 LPA for experienced professionals at product companies. In the US, salaries typically 
range from $95,000–$150,000. A strong portfolio of real-world projects and Kaggle competition 
experience significantly boosts hiring prospects for aspiring Data Scientists.
        """.strip(),
        "metadata": {"topic": "Data Scientist", "type": "career_overview"}
    },
    {
        "id": "doc_003",
        "title": "Software Engineer Career Overview",
        "content": """
Software Engineering is one of the broadest and most universally demanded careers in the 
technology industry. Software Engineers design, develop, test, and maintain software systems 
ranging from mobile applications and web platforms to embedded systems and operating systems.

The field is divided into numerous specializations: frontend engineering (React, Vue, Angular), 
backend engineering (Node.js, Django, Spring Boot), full-stack development, mobile development 
(iOS/Android), DevOps and infrastructure, and embedded systems. Each specialization requires 
a distinct but overlapping skill set.

Core skills every Software Engineer needs include proficiency in at least one programming 
language (Python, Java, JavaScript, C++), understanding of data structures and algorithms, 
version control with Git, and knowledge of system design principles. Problem-solving ability 
and the capacity to write clean, maintainable code are critical traits.

Career growth in software engineering is rapid: Junior Developer → Software Engineer → Senior 
Engineer → Staff/Principal Engineer → Engineering Manager or Architect. Many engineers opt for 
the technical track (individual contributor) while others move into people management.

In India, fresher Software Engineers earn ₹4–10 LPA at service companies and ₹12–30 LPA at 
product companies. In the US, entry-level engineers earn $90,000–$130,000, and senior engineers 
at top firms can earn well above $200,000 including equity. The global demand for Software 
Engineers continues to grow, driven by digital transformation across all industries.
        """.strip(),
        "metadata": {"topic": "Software Engineer", "type": "career_overview"}
    },
    {
        "id": "doc_004",
        "title": "AI Engineer Career Overview",
        "content": """
An AI Engineer sits at the intersection of software engineering and data science, specializing 
in building, deploying, and maintaining AI-powered systems in production environments. Unlike 
Data Scientists who primarily focus on research and model development, AI Engineers are 
responsible for making AI solutions scalable, reliable, and integrated into real-world applications.

AI Engineers work with large language models (LLMs), computer vision pipelines, recommendation 
systems, and conversational AI. They use frameworks like LangChain, LlamaIndex, Hugging Face 
Transformers, TensorFlow Serving, and ONNX to productionize machine learning models. Strong 
knowledge of cloud platforms (AWS SageMaker, Azure ML, GCP Vertex AI) is essential.

The role requires programming expertise in Python, understanding of RESTful API design, 
containerization (Docker, Kubernetes), MLOps practices, vector databases (Pinecone, ChromaDB, 
Weaviate), and prompt engineering. Familiarity with Retrieval-Augmented Generation (RAG), 
agentic AI frameworks (LangGraph, AutoGen), and fine-tuning techniques is increasingly important.

Career path: Junior AI Engineer → AI Engineer → Senior AI Engineer → AI Architect → Head of AI. 
Many AI Engineers also transition into product-focused roles like AI Product Manager.

Salaries for AI Engineers in India range from ₹8–15 LPA at the entry level and can reach 
₹40–60 LPA for experienced professionals at MAANG or AI-native companies. In the US, AI 
Engineers earn between $120,000–$200,000+, making it one of the highest-paying roles in tech. 
The rapid adoption of generative AI has made this one of the fastest-growing career paths in 2024–2025.
        """.strip(),
        "metadata": {"topic": "AI Engineer", "type": "career_overview"}
    },
    {
        "id": "doc_005",
        "title": "Skills Required for Data and AI Roles",
        "content": """
Choosing the right career in data and AI begins with understanding the skill requirements for 
each role. While there is significant overlap, each position demands a unique combination of 
technical and soft skills.

For Data Analysts: SQL (advanced querying, joins, window functions), Excel, Python (pandas, 
matplotlib), Tableau or Power BI, statistical concepts (mean, variance, hypothesis testing), 
and business communication skills.

For Data Scientists: Python (scikit-learn, TensorFlow, PyTorch), R, linear algebra, calculus, 
probability theory, machine learning algorithms (regression, classification, clustering), 
feature engineering, model evaluation, and storytelling with data.

For Software Engineers: Data structures and algorithms, object-oriented programming, system 
design (scalability, microservices), Git, testing (unit/integration), CI/CD pipelines, and 
cloud fundamentals.

For AI Engineers: All of the above plus LLM orchestration (LangChain, LangGraph), vector 
databases, RAG architectures, prompt engineering, model fine-tuning, MLOps, Docker, 
Kubernetes, and monitoring AI systems in production.

Regardless of role, soft skills matter enormously: problem-solving mindset, the ability to 
translate technical findings to stakeholders, team collaboration, and continuous learning. 
Students often underestimate the importance of communication and project management in 
data-driven careers.

A practical approach is to build skills progressively — start with Python and SQL as your 
universal foundation, then branch into your chosen specialization. Certifications from 
Google, AWS, Microsoft, or Coursera can supplement self-learning and signal credibility 
to recruiters.
        """.strip(),
        "metadata": {"topic": "Skills", "type": "skills_guide"}
    },
    {
        "id": "doc_006",
        "title": "Learning Roadmaps for Beginners",
        "content": """
Starting a career in data or AI can feel overwhelming without a structured plan. Here are 
proven roadmaps for beginners entering each field.

Data Analyst Roadmap (6–9 months): Month 1–2: Learn Excel and SQL basics. Month 3–4: 
Python fundamentals and pandas. Month 5–6: Tableau or Power BI for visualization. Month 7–9: 
Statistics fundamentals, real-world projects, and job applications. Build a portfolio with 
3–5 dashboards and analysis projects on GitHub.

Data Scientist Roadmap (12–18 months): Build Python and SQL first (3 months). Study 
mathematics: linear algebra, statistics, calculus (2 months). Learn machine learning with 
Scikit-learn (3 months). Deep learning with TensorFlow or PyTorch (3 months). Work on 
end-to-end projects and Kaggle competitions (3 months). 

Software Engineer Roadmap (9–12 months): Learn one language deeply — Python or JavaScript 
recommended. Master data structures and algorithms using LeetCode. Study system design 
concepts. Build 3–5 full-stack or backend projects. Contribute to open source on GitHub.

AI Engineer Roadmap (18–24 months): Complete the Data Scientist path first. Then learn 
LangChain, LlamaIndex, and LangGraph for agentic applications. Study vector databases and 
RAG systems. Learn Docker and Kubernetes for deployment. Study MLOps with MLflow or Weights 
& Biases. Build LLM-powered applications and deploy to production.

Resources recommended for all paths: Python.org, fast.ai, Coursera (Andrew Ng's courses), 
StatQuest (YouTube), roadmap.sh, and Towards Data Science on Medium. Consistency matters 
more than speed — 2 hours of focused daily study outperforms weekend cramming sessions.
        """.strip(),
        "metadata": {"topic": "Roadmaps", "type": "learning_path"}
    },
    {
        "id": "doc_007",
        "title": "Salary Insights Across Data and AI Careers",
        "content": """
Understanding salary benchmarks helps students set realistic expectations and make informed 
career choices. Compensation in data and AI careers varies significantly based on role, 
experience, company size, and location.

In India (₹ LPA — Lakhs Per Annum):
- Data Analyst: Fresher ₹3.5–6 LPA | Mid-level ₹8–14 LPA | Senior ₹15–25 LPA
- Data Scientist: Fresher ₹6–10 LPA | Mid-level ₹15–25 LPA | Senior ₹28–45 LPA
- Software Engineer: Fresher ₹4–12 LPA | Mid-level ₹18–35 LPA | Senior ₹40–80 LPA
- AI Engineer: Fresher ₹8–15 LPA | Mid-level ₹20–40 LPA | Senior ₹45–80 LPA

In the United States (USD Annual):
- Data Analyst: Entry $55K–$75K | Mid $80K–$105K | Senior $110K–$140K
- Data Scientist: Entry $95K–$115K | Mid $120K–$155K | Senior $160K–$210K
- Software Engineer: Entry $95K–$130K | Mid $140K–$180K | Senior $190K–$280K
- AI Engineer: Entry $115K–$145K | Mid $155K–$200K | Senior $210K–$350K+

Factors that boost compensation: advanced degree (MS/PhD adds 20–40%), MAANG/Unicorn 
company employment (2–5x higher than market), niche specializations (LLM fine-tuning, 
computer vision), and strong open-source or research contributions.

Stock options and equity compensation (ESOPs/RSUs) can dramatically increase total 
compensation at startups and product companies. Freshers should not solely compare base 
salaries but also evaluate learning opportunities, mentorship, and long-term growth potential 
when choosing their first job.
        """.strip(),
        "metadata": {"topic": "Salary", "type": "compensation_guide"}
    },
    {
        "id": "doc_008",
        "title": "Tools and Technologies for Data and AI Professionals",
        "content": """
Mastering the right tools is crucial for productivity and employability in data and AI careers. 
Here's a comprehensive overview of must-know tools organized by role.

Data Analyst Tools: SQL (PostgreSQL, MySQL, BigQuery), Microsoft Excel/Google Sheets, 
Python (pandas, matplotlib, seaborn), Tableau, Power BI, Looker, Jupyter Notebooks.

Data Scientist Tools: Python ecosystem (NumPy, pandas, Scikit-learn, XGBoost, LightGBM), 
TensorFlow, PyTorch, Keras, Jupyter Lab, MLflow for experiment tracking, DVC for data 
versioning, Weights & Biases, Google Colab.

Software Engineer Tools: Git and GitHub/GitLab, VS Code or IntelliJ, Docker, Kubernetes, 
CI/CD tools (GitHub Actions, Jenkins, CircleCI), Postman for API testing, Linux/Bash, 
AWS/Azure/GCP cloud platforms.

AI Engineer Tools: LangChain, LangGraph, LlamaIndex, OpenAI API, Hugging Face Transformers, 
ChromaDB, Pinecone, Weaviate (vector stores), Ollama (local LLMs), FastAPI for model serving, 
Ray for distributed computing, Triton Inference Server, ONNX for model optimization.

Cross-Role Tools: Git (version control), Jupyter Notebooks, Linux CLI, REST APIs, 
Slack/Notion for collaboration, and SQL remain universal.

Students should focus on depth over breadth: become genuinely proficient in 3–4 core tools 
before expanding your toolkit. Employers value demonstrated expertise over a long list of 
superficial skills. Building portfolio projects that showcase real tool usage is far more 
impactful than simply listing technologies on a resume.
        """.strip(),
        "metadata": {"topic": "Tools", "type": "tools_guide"}
    },
    {
        "id": "doc_009",
        "title": "Interview Preparation for Data and AI Roles",
        "content": """
Interview preparation strategy varies by role, but certain fundamentals apply across all 
data and AI positions. Understanding what interviewers look for helps you prepare systematically.

Data Analyst Interviews typically include: SQL challenges (complex JOINs, window functions, 
CTEs), case studies (business problem → data-driven approach), Excel/Python data manipulation 
tasks, and behavioral questions. Practice on platforms like StrataScratch and Mode Analytics.

Data Scientist Interviews have three pillars: Machine Learning Theory (bias-variance tradeoff, 
regularization, cross-validation), Coding (Python for data manipulation and ML), and Case 
Studies (end-to-end problem solving). Study Andrew Ng's ML course and review Scikit-learn 
documentation. Practice on Kaggle and LeetCode (ML section).

Software Engineer Interviews focus heavily on Data Structures and Algorithms (DSA). Practice 
200+ LeetCode problems across arrays, trees, graphs, and dynamic programming. System design 
interviews test your ability to design scalable systems — study resources like "System Design 
Interview" by Alex Xu. Behavioral questions use the STAR format.

AI Engineer Interviews test LLM knowledge (transformer architecture, attention mechanism, 
fine-tuning vs. RAG), practical coding (building chains, agents, RAG pipelines), and MLOps 
concepts. Be prepared to discuss production challenges: latency, hallucination mitigation, 
and evaluation frameworks.

Universal preparation tips: Review your resume projects in depth (interviewers ask detailed 
follow-up questions), research the company's tech stack before the interview, practice 
explaining technical concepts simply, and always ask thoughtful questions at the end of 
interviews. Mock interviews with peers or on Pramp/Interviewing.io significantly improve 
performance under pressure.
        """.strip(),
        "metadata": {"topic": "Interview Prep", "type": "interview_guide"}
    },
    {
        "id": "doc_010",
        "title": "Common Mistakes Students Make When Choosing Data and AI Careers",
        "content": """
Students entering the data and AI field often make avoidable mistakes that slow their career 
progress. Awareness of these pitfalls can save months of wasted effort.

Mistake 1 — Tutorial Hell: Watching dozens of YouTube tutorials and online courses without 
building real projects. Consuming content feels productive but is no substitute for hands-on 
experience. Solution: For every tutorial completed, build an original project from scratch.

Mistake 2 — Skipping Mathematics: Students jump into neural network libraries without 
understanding the underlying math. This creates fragile knowledge that breaks during deep 
technical interviews. Solution: Invest time in linear algebra, statistics, and calculus basics 
before diving into advanced ML frameworks.

Mistake 3 — Chasing Trends Blindly: Switching focus to every new AI model or framework 
without mastering fundamentals. The AI landscape evolves rapidly, but core principles remain 
stable. Solution: Build depth in fundamentals; adopt new tools only when you have a concrete 
use case.

Mistake 4 — Neglecting Soft Skills: Assuming technical excellence alone guarantees success. 
Data insights have zero value if they cannot be communicated effectively. Solution: Practice 
presenting your projects to non-technical audiences regularly.

Mistake 5 — Weak Portfolios: Applying for jobs with a resume full of course certificates but 
no real projects. Recruiters want evidence of practical ability. Solution: Build 3–5 projects 
that solve genuine problems, document them professionally on GitHub, and deploy them publicly.

Mistake 6 — Ignoring Networking: Learning in isolation and failing to engage with communities 
on LinkedIn, Kaggle, GitHub, or local meetups. Professional networks accelerate job searches 
significantly. Engage authentically, share your learning journey, and connect with industry 
professionals consistently.
        """.strip(),
        "metadata": {"topic": "Common Mistakes", "type": "career_advice"}
    }
]


# ---------------------------------------------------------------------------
# ChromaDB Setup
# ---------------------------------------------------------------------------

CHROMA_PATH = "./chroma_career_db"
COLLECTION_NAME = "career_knowledge_base"


def get_embedding_function():
    """Returns the sentence-transformer embedding function for ChromaDB."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def build_knowledge_base(force_rebuild: bool = False) -> chromadb.Collection:
    """
    Builds or loads the ChromaDB collection with career documents.
    
    Args:
        force_rebuild: If True, deletes existing collection and rebuilds.
    
    Returns:
        ChromaDB collection object.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = get_embedding_function()

    existing_collections = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing_collections:
        if force_rebuild:
            client.delete_collection(COLLECTION_NAME)
            print(f"[knowledge_base] Deleted existing collection '{COLLECTION_NAME}'.")
        else:
            collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=ef
            )
            print(f"[knowledge_base] Loaded existing collection with {collection.count()} documents.")
            return collection

    # Create new collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Insert all documents
    ids = [doc["id"] for doc in DOCUMENTS]
    texts = [doc["content"] for doc in DOCUMENTS]
    metadatas = [doc["metadata"] for doc in DOCUMENTS]

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )

    print(f"[knowledge_base] Created collection '{COLLECTION_NAME}' with {len(DOCUMENTS)} documents.")
    return collection


def retrieve_documents(query: str, n_results: int = 3) -> dict:
    """
    Retrieves the top-n relevant documents from the knowledge base.
    
    Args:
        query: The search query string.
        n_results: Number of documents to retrieve.
    
    Returns:
        Dictionary with 'documents', 'metadatas', and 'distances'.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = get_embedding_function()

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count())
    )

    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0]
    }


def get_all_topics() -> list:
    """Returns a list of all document topics in the knowledge base."""
    return [doc["metadata"]["topic"] for doc in DOCUMENTS]


# ---------------------------------------------------------------------------
# Main: Build knowledge base when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building Career Knowledge Base...")
    collection = build_knowledge_base(force_rebuild=True)
    print(f"\nKnowledge base ready with {collection.count()} documents.")
    print("\nAvailable topics:")
    for topic in get_all_topics():
        print(f"  - {topic}")

    # Quick test retrieval
    print("\n--- Test Retrieval: 'How to become a Data Scientist' ---")
    results = retrieve_documents("How to become a Data Scientist", n_results=2)
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        print(f"\n[Result {i+1}] Topic: {meta['topic']}")
        print(doc[:200] + "...")
