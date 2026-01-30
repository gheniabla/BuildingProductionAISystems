# Building Production AI Systems

## Course Syllabus

**Course Duration:** 10 Weeks
**Lectures:** 16 Sessions (2 per week, Weeks 1-8)
**Assessments:** Midterm (Week 5), Final Project Demo (Week 10)
**Prerequisites:** Python proficiency, basic ML/AI understanding, familiarity with REST APIs

---

## Course Overview

This course addresses one of the hardest problems in modern engineering: **taking AI systems from impressive demos to reliable, secure, production-ready systems used by real users.**

While building a ChatGPT wrapper that works in a Jupyter notebook takes hours, building an AI system that can handle thousands of concurrent users, survive adversarial attacks, stay within budget, and maintain quality over time takes months of careful engineering.

This course bridges that gap.

---

## Learning Objectives

By the end of this course, students will be able to:

1. **Design and implement production-grade AI architectures** using FastAPI, Pydantic v2, and Celery/Redis
2. **Build comprehensive evaluation systems** for LLMs, RAG pipelines, and AI agents
3. **Identify and mitigate AI-specific security threats** including prompt injection and model DoS
4. **Deploy optimized AI systems** using Docker, Kubernetes, and serving engines like vLLM
5. **Implement observability solutions** with distributed tracing and quality monitoring
6. **Make informed trade-offs** between latency, cost, reliability, and development velocity

---

## Weekly Schedule Overview

```
┌─────────┬────────────────────────────────────────────────────────────────────┐
│  Week   │                           Topics                                   │
├─────────┼────────────────────────────────────────────────────────────────────┤
│    1    │ Foundations: GenAI Review & Production Mindset                     │
│    2    │ Production Architecture: FastAPI + Pydantic + Async Patterns       │
│    3    │ AI Evaluation: Metrics, Datasets, and Scoring Systems              │
│    4    │ AI Security: Threat Modeling and Guardrails                        │
│    5    │ ★ MIDTERM EXAM + RAG Systems Deep Dive                             │
│    6    │ Deployment: Containers, Kubernetes, and Serving Engines            │
│    7    │ Optimization: Quantization, Caching, and Batching                  │
│    8    │ Observability: Tracing, Monitoring, and Quality Assurance          │
│    9    │ Project Development Week (No Lectures - Office Hours)              │
│   10    │ ★ FINAL PROJECT DEMONSTRATIONS                                     │
└─────────┴────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Weekly Schedule

### Week 1: Foundations of Production AI

**Lecture 1: The Production AI Landscape**
- The demo-to-production gap: Why 90% of AI projects fail in production
- Anatomy of production AI systems: Components and data flows
- War story: The $100,000 weekend (runaway API costs)
- Trade-off frameworks: Latency vs. Cost, Reliability vs. Velocity
- Modern AI architectures: LLMs, RAG, Agents, and Multi-modal systems

**Lecture 2: Generative AI Technical Review**
- Transformer architecture refresher
- Tokenization, context windows, and attention mechanisms
- API-based vs. self-hosted models: Decision framework
- Model selection criteria for production
- Hands-on: Setting up the development environment

**Assignment 1 Released:** Production Environment Setup & Basic LLM Integration

---

### Week 2: Production Architecture Patterns

**Lecture 3: The FastAPI + Pydantic Stack**
- Why FastAPI for AI systems: Performance benchmarks
- Pydantic v2: Data validation at the speed of Rust
- Request/response modeling for AI endpoints
- Dependency injection patterns
- Error handling strategies for AI failures

**Lecture 4: Asynchronous AI Workloads**
- Understanding async/await in Python
- Celery + Redis for background AI tasks
- Task queues, priorities, and dead letter handling
- Implementing retry strategies with exponential backoff
- Hands-on: Building an async AI processing pipeline

---

### Week 3: AI Evaluation Systems

**Lecture 5: Evaluation Fundamentals**
- Why evaluation is the hardest problem in AI
- Evaluation dataset types: Golden, adversarial, regression, and edge cases
- Standard metrics: BLEU, ROUGE, BERTScore, and their limitations
- Building evaluation datasets that actually matter
- Case study: How evaluation failures led to production incidents

**Lecture 6: Advanced Evaluation Techniques**
- LLM-as-judge: Principles and pitfalls
- Designing rubrics that correlate with human judgment
- Guardrails for automated evaluation
- A/B testing for AI systems
- Frameworks: OpenAI Evals, RAGAS, DeepEval
- Hands-on: Building a custom evaluation pipeline

**Assignment 1 Due / Assignment 2 Released:** Building an AI Evaluation Suite

---

### Week 4: AI Security and Guardrails

**Lecture 7: AI Threat Landscape**
- OWASP Top 10 for LLM Applications
- Prompt injection: Direct, indirect, and jailbreaks
- Insecure output handling and data exfiltration
- Model denial of service attacks
- Supply chain risks in AI systems
- MITRE ATLAS threat taxonomy

**Lecture 8: Implementing Guardrails**
- Input sanitization strategies
- Output validation and content filtering
- Rate limiting and abuse prevention
- Secrets management in AI pipelines
- Building defense-in-depth architectures
- Hands-on: Implementing a security layer for LLM applications

---

### Week 5: Midterm & RAG Systems

**Lecture 9: ★ MIDTERM EXAMINATION**
- Covers Weeks 1-4 material
- Written exam + practical component

**Lecture 10: RAG Systems Deep Dive**
- Vector database selection: Pinecone vs. Weaviate vs. Qdrant
- Indexing strategies: HNSW, IVFFlat, and trade-offs
- Chunking strategies that actually work
- Hybrid search: Combining dense and sparse retrieval
- Reranking and contextual compression

**Assignment 2 Due / Assignment 3 Released:** Secure RAG Pipeline with Guardrails

---

### Week 6: Deployment Strategies

**Lecture 11: Containerization and Orchestration**
- Docker best practices for AI workloads
- Kubernetes fundamentals for ML engineers
- Resource management: GPUs, memory, and CPU allocation
- Horizontal Pod Autoscaling for AI services
- Hybrid routing: Self-hosted + managed API patterns

**Lecture 12: Model Serving Engines**
- vLLM: Architecture and optimization techniques
- NVIDIA TensorRT-LLM: When and how to use it
- Serving framework comparison: Triton, Ray Serve, BentoML
- Staged rollouts and canary deployments
- Hands-on: Deploying a model with vLLM on Kubernetes

---

### Week 7: Optimization Techniques

**Lecture 13: Model Optimization**
- Quantization fundamentals: FP16, FP8, INT8, INT4
- Post-training quantization vs. quantization-aware training
- Pruning strategies for production models
- Knowledge distillation: Building smaller, faster models
- When optimization hurts: Quality degradation patterns

**Lecture 14: Inference Optimization**
- Prompt caching strategies
- KV cache optimization
- Efficient attention variants: Flash Attention, Paged Attention
- Batching strategies: Static, dynamic, and continuous
- Speculative decoding
- Hands-on: Optimizing an inference pipeline

**Assignment 3 Due / Assignment 4 Released:** Optimized Model Deployment Pipeline

---

### Week 8: Observability and Operations

**Lecture 15: Distributed Tracing for AI**
- Why AI systems need specialized observability
- OpenTelemetry for AI workloads
- Tracing LLM calls, embeddings, and retrievals
- Cost tracking and attribution
- Tools: LangSmith, Honeycomb, Arize Phoenix

**Lecture 16: Quality Monitoring and Incident Response**
- Detecting quality regressions in production
- Building quality dashboards
- Alert design for AI systems
- Incident response playbooks
- Post-mortem practices for AI failures
- Hands-on: Implementing an observability stack

**Assignment 4 Due / Assignment 5 (Final) Released:** End-to-End Production AI System

---

### Week 9: Project Development

**No Formal Lectures**
- Extended office hours
- Project architecture reviews
- Technical deep-dives on request
- Guest speaker (industry practitioner)

---

### Week 10: Final Presentations

**Day 1: Final Project Demonstrations (Groups 1-8)**
- 15-minute demo + 5-minute Q&A per group

**Day 2: Final Project Demonstrations (Groups 9-16) + Course Wrap-up**
- Remaining demonstrations
- Course retrospective
- Career paths in Production AI
- Continuing education resources

**Assignment 5 Due:** Complete system demonstration and code submission

---

## Assignments Overview

| Assignment | Topic | Weight | Duration |
|------------|-------|--------|----------|
| 1 | Production Environment & LLM Integration | 10% | Weeks 1-3 |
| 2 | AI Evaluation Suite | 15% | Weeks 3-5 |
| 3 | Secure RAG Pipeline | 15% | Weeks 5-7 |
| 4 | Optimized Deployment Pipeline | 15% | Weeks 7-8 |
| 5 | End-to-End Production System (Final) | 25% | Weeks 8-10 |

---

## Grading

| Component | Weight |
|-----------|--------|
| Assignments (1-4) | 55% |
| Midterm Exam | 10% |
| Final Project | 25% |
| Participation & Labs | 10% |

---

## Technology Stack

### Required
- **Python 3.11+** (Expert level expected)
- **FastAPI + Pydantic v2** (Backend framework)
- **Docker** (Containerization)
- **Git** (Version control)

### Used Throughout Course
- **Orchestration:** Celery, Redis
- **Frameworks:** LangGraph, LangChain, Haystack
- **Vector Databases:** Qdrant, Weaviate, Pinecone
- **Serving:** vLLM, TensorRT-LLM
- **Observability:** LangSmith, Arize Phoenix, OpenTelemetry
- **Cloud:** Azure OpenAI, AWS Bedrock (optional)

---

## Required Reading

1. **Primary Text:** Course Notes (provided)
2. **Reference:** "AI Engineering" by Chip Huyen (O'Reilly, 2025)
3. **Reference:** "Designing Machine Learning Systems" by Chip Huyen
4. **Security:** OWASP Top 10 for LLM Applications
5. **Papers:** Selected papers provided per week

---

## Office Hours

- **Instructor:** 2 hours/week (scheduled)
- **TAs:** 4 hours/week (rotating schedule)
- **Week 9:** Extended office hours for final project support

---

## Academic Integrity

AI assistants may be used for learning and coding assistance. All submitted work must be understood and explainable by the student. Plagiarism detection tools will be used on code submissions.

---

## Accessibility

Please contact the instructor within the first week if you require accommodations.
