# Building Production AI Systems

A comprehensive 10-week course on taking AI systems from demos to production.

---

## Course Overview

This course addresses one of the hardest problems in modern engineering: **taking AI systems from impressive demos to reliable, secure, production-ready systems used by real users.**

While building a ChatGPT wrapper that works in a Jupyter notebook takes hours, building an AI system that can handle thousands of concurrent users, survive adversarial attacks, stay within budget, and maintain quality over time takes months of careful engineering.

**This course bridges that gap.**

---

## Learning Outcomes

After completing this course, students will be able to:

1. **Design and implement production-grade AI architectures** using FastAPI, Pydantic v2, and Celery/Redis
2. **Build comprehensive evaluation systems** for LLMs, RAG pipelines, and AI agents
3. **Identify and mitigate AI-specific security threats** including prompt injection and model DoS
4. **Deploy optimized AI systems** using Docker, Kubernetes, and serving engines like vLLM
5. **Implement observability solutions** with distributed tracing and quality monitoring
6. **Make informed trade-offs** between latency, cost, reliability, and development velocity

---

## Course Structure

```
Week 1  │ Foundations: GenAI Review & Production Mindset
Week 2  │ Production Architecture: FastAPI + Pydantic + Async Patterns
Week 3  │ AI Evaluation: Metrics, Datasets, and Scoring Systems
Week 4  │ AI Security: Threat Modeling and Guardrails
Week 5  │ ★ MIDTERM EXAM + RAG Systems Deep Dive
Week 6  │ Deployment: Containers, Kubernetes, and Serving Engines
Week 7  │ Optimization: Quantization, Caching, and Batching
Week 8  │ Observability: Tracing, Monitoring, and Quality Assurance
Week 9  │ Project Development Week (Office Hours)
Week 10 │ ★ FINAL PROJECT DEMONSTRATIONS
```

---

## Repository Structure

```
BuildingProductionAISystems/
│
├── syllabus/
│   └── course-syllabus.md          # Complete syllabus with schedule
│
├── course-notes/
│   ├── 00-preface.md               # Course introduction
│   ├── 01-week1-foundations.md     # Week 1: Foundations
│   ├── 02-week2-production-architecture.md
│   ├── 03-week3-ai-evaluation.md
│   ├── 04-week4-ai-security.md
│   ├── 05-week5-rag-systems.md
│   ├── 06-week6-deployment.md
│   ├── 07-week7-optimization.md
│   └── 08-week8-observability.md
│
├── assignments/
│   ├── assignment1-production-setup.md      # Weeks 1-3
│   ├── assignment2-evaluation-suite.md      # Weeks 3-5
│   ├── assignment3-secure-rag.md            # Weeks 5-7
│   ├── assignment4-deployment-optimization.md # Weeks 7-8
│   └── assignment5-final-project.md         # Weeks 8-10
│
├── resources/
│   └── architecture-diagrams.md    # Visual references
│
└── README.md                       # This file
```

---

## Technology Stack

### Core Technologies
- **Python 3.11+** (Expert level expected)
- **FastAPI + Pydantic v2** (Backend framework)
- **Docker + Kubernetes** (Containerization & orchestration)
- **Redis** (Caching, task queue)
- **Celery** (Background tasks)

### AI/ML Stack
- **OpenAI API** (GPT-4, GPT-4-mini)
- **Anthropic API** (Claude)
- **LangGraph / LangChain** (Agent frameworks)
- **vLLM / TensorRT-LLM** (Model serving)
- **Qdrant / Weaviate** (Vector databases)

### Observability
- **OpenTelemetry** (Distributed tracing)
- **LangSmith / Arize Phoenix** (AI observability)
- **Prometheus + Grafana** (Metrics & dashboards)

---

## Assignments Overview

| # | Assignment | Topics | Weight | Duration |
|---|------------|--------|--------|----------|
| 1 | Production Environment Setup | FastAPI, Pydantic, LLM Service | 10% | Weeks 1-3 |
| 2 | AI Evaluation Suite | Metrics, LLM-as-Judge, Datasets | 15% | Weeks 3-5 |
| 3 | Secure RAG Pipeline | Vector DB, Security, Guardrails | 15% | Weeks 5-7 |
| 4 | Optimized Deployment | Docker, K8s, Caching, Routing | 15% | Weeks 7-8 |
| 5 | **Final Project** | End-to-End Production System | 25% | Weeks 8-10 |

---

## Prerequisites

Students should have:
- **Strong Python proficiency** (classes, async/await, type hints)
- **Basic ML/AI understanding** (what transformers are, how LLMs work)
- **REST API experience** (building or consuming APIs)
- **Git proficiency** (commits, branches, PRs)
- **Command line comfort** (bash/zsh basics)

---

## How to Use This Repository

### For Self-Study
1. Start with `course-notes/00-preface.md` for setup
2. Work through weekly notes in order
3. Complete assignments to cement learning
4. Reference `resources/` for diagrams and examples

### For Instructors
1. Use `syllabus/course-syllabus.md` for planning
2. Course notes are designed to be lecture companions
3. Assignments include detailed rubrics
4. Architecture diagrams can be used in slides

---

## Key Concepts Covered

### Production Foundations
- The demo-to-production gap
- Trade-off frameworks (cost, latency, quality, reliability)
- FastAPI + Pydantic for AI services
- Async patterns for LLM workloads
- Celery + Redis for background tasks

### AI Evaluation
- Evaluation dataset types (golden, adversarial, regression)
- Metrics (BLEU, ROUGE, semantic similarity)
- LLM-as-judge with guardrails
- CI/CD integration for evaluations
- Quality monitoring in production

### AI Security
- OWASP Top 10 for LLMs
- Prompt injection (direct, indirect, jailbreaks)
- MITRE ATLAS threat taxonomy
- Defense in depth architecture
- Input sanitization and output validation

### RAG Systems
- Vector database selection and indexing
- Chunking strategies
- Hybrid search (dense + sparse)
- Reranking and context assembly
- Secure context handling

### Deployment
- Docker best practices for AI
- Kubernetes for AI workloads
- GPU scheduling and management
- vLLM and model serving
- Canary deployments

### Optimization
- Quantization (INT8, INT4, GPTQ)
- Knowledge distillation
- Prompt caching strategies
- Batching for throughput
- Intelligent model routing

### Observability
- OpenTelemetry for AI
- Cost tracking and attribution
- Quality monitoring dashboards
- Incident response playbooks
- Post-mortem practices

---

## Assessment

| Component | Weight |
|-----------|--------|
| Assignments (1-4) | 55% |
| Midterm Exam | 10% |
| Final Project | 25% |
| Participation | 10% |

---

## Contact & Support

- **Office Hours:** See syllabus for schedule
- **Discussion:** Course forum (TBD)
- **Issues:** For curriculum improvements, open an issue

---

## Contributing

This curriculum is open for improvements. Please submit:
- Bug fixes (typos, code errors)
- Clarity improvements
- Updated tool references
- New examples or exercises

---

## License

This curriculum is provided for educational purposes. See individual dependencies for their respective licenses.

---

*"The gap between 'it works on my laptop' and 'it serves 10,000 users reliably' is where most AI projects go to die. This course exists to bridge that gap."*
