# Assignment 5: End-to-End Production AI System (Final Project)

**Released:** Week 8
**Due:** Week 10 (Demonstration Day)
**Weight:** 25%

---

## Overview

For the final project, you will build a complete, production-ready AI application that demonstrates mastery of all concepts covered in the course. You will integrate evaluation, security, deployment, optimization, and observability into a cohesive system.

---

## Project Options

Choose ONE of the following project types:

### Option A: Intelligent Customer Support Agent
Build an AI agent that can:
- Answer questions using a knowledge base (RAG)
- Execute actions (refunds, order status, escalation)
- Maintain conversation context
- Handle multi-turn conversations

### Option B: Code Review Assistant
Build an AI assistant that can:
- Analyze code for issues
- Suggest improvements
- Explain code functionality
- Generate documentation

### Option C: Research Assistant with Tools
Build an AI agent that can:
- Search and summarize documents
- Use external tools (calculator, web search)
- Generate reports
- Track research progress

### Option D: Custom Project
Propose your own project (requires instructor approval by Week 9).

---

## Requirements

All projects must include ALL of the following components:

### 1. Core Application (20 points)

```
your-project/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── services/         # Business logic
│   ├── models/           # Pydantic models
│   ├── agents/           # Agent/workflow logic (if applicable)
│   ├── security/         # Guardrails
│   ├── evals/            # Evaluation framework
│   └── observability/    # Tracing and metrics
├── tests/
├── datasets/             # Evaluation datasets
├── docker/
├── k8s/
└── docs/
```

**Deliverables:**
- [ ] Functional application that serves its purpose
- [ ] Clean code with type hints
- [ ] Comprehensive error handling
- [ ] API documentation (auto-generated OpenAPI)

### 2. Evaluation System (15 points)

Implement a complete evaluation suite:

```python
# Minimum requirements:
# - 30+ golden examples
# - 15+ adversarial examples
# - 10+ edge cases
# - LLM-as-judge integration
# - CI/CD integration

async def run_full_evaluation():
    """
    Run complete evaluation and return pass/fail.
    Must pass 90% of golden, 85% of adversarial.
    """
    pass
```

**Deliverables:**
- [ ] Three evaluation datasets (golden, adversarial, edge cases)
- [ ] Custom metrics for your domain
- [ ] LLM-as-judge with rubrics
- [ ] Evaluation runs in CI (GitHub Actions or similar)
- [ ] Quality dashboard or report

### 3. Security Implementation (15 points)

Comprehensive security layer:

```python
# Must include:
# - Input sanitization with injection detection
# - Output validation with PII detection
# - Rate limiting per user
# - Budget controls
# - Audit logging
```

**Deliverables:**
- [ ] Input guardrails (10+ patterns detected)
- [ ] Output guardrails
- [ ] Rate limiting (configurable)
- [ ] Cost/budget controls
- [ ] Security test suite (10+ tests)

### 4. Deployment Configuration (15 points)

Production-ready deployment:

```yaml
# Must include:
# - Multi-stage Dockerfile
# - Docker Compose for development
# - Kubernetes manifests
# - HPA configuration
# - Health checks
```

**Deliverables:**
- [ ] Docker image < 500MB
- [ ] Docker Compose with all dependencies
- [ ] Kubernetes deployment + service + HPA
- [ ] ConfigMap and Secret references
- [ ] Health and readiness probes

### 5. Optimization (15 points)

Performance optimization:

```python
# Must include:
# - Response caching (exact match)
# - Semantic caching (similarity-based)
# - Intelligent model routing
# - Cost tracking
```

**Deliverables:**
- [ ] Caching with >20% hit rate on demo workload
- [ ] Model routing based on context
- [ ] Cost tracking per request
- [ ] Performance benchmarks documented

### 6. Observability (10 points)

Complete observability stack:

```python
# Must include:
# - Distributed tracing (OpenTelemetry or LangSmith)
# - Metrics (latency, tokens, costs)
# - Structured logging
# - Quality monitoring (sampled evaluation)
```

**Deliverables:**
- [ ] Traces for all AI operations
- [ ] Metrics dashboard (Grafana or similar)
- [ ] Quality monitoring with sampling
- [ ] Alert configuration (at least 3 alerts)

### 7. Documentation (10 points)

Professional documentation:

```markdown
# README.md should include:
- Project overview
- Architecture diagram
- Setup instructions
- API documentation
- Security considerations
- Performance benchmarks
- Known limitations
```

**Deliverables:**
- [ ] README with setup instructions
- [ ] Architecture diagram
- [ ] API documentation
- [ ] Security documentation
- [ ] Runbook for common operations

---

## Demonstration Requirements

During the final demonstration (Week 10), you must show:

1. **Live Demo (10 minutes)**
   - Application working end-to-end
   - Normal use cases
   - Edge case handling
   - Security controls in action

2. **Architecture Walkthrough (5 minutes)**
   - System components
   - Data flow
   - Security boundaries

3. **Observability Demo (3 minutes)**
   - Show traces
   - Show metrics
   - Show quality scores

4. **Q&A (2 minutes)**
   - Answer technical questions
   - Explain design decisions

---

## Example Architecture (Option A: Customer Support)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CUSTOMER SUPPORT AGENT ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │     Client      │
                              │   (Web/API)     │
                              └────────┬────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY                                    │
│   [Rate Limit] [Auth] [Input Validation] [Request Tracing]                 │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENT ORCHESTRATOR                                │
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│   │ Intent          │───▶│ Tool Selection  │───▶│ Response        │       │
│   │ Classification  │    │ & Execution     │    │ Generation      │       │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘       │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│      RAG        │         │     TOOLS       │         │   LLM SERVICE   │
│   Knowledge     │         │  - Order API    │         │   - GPT-4       │
│   Base Search   │         │  - Refund API   │         │   - Routing     │
│                 │         │  - Escalation   │         │   - Caching     │
└─────────────────┘         └─────────────────┘         └─────────────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│    Qdrant       │         │   Mock APIs     │         │    OpenAI /     │
│  Vector Store   │         │  (Simulation)   │         │    Anthropic    │
└─────────────────┘         └─────────────────┘         └─────────────────┘

                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OBSERVABILITY                                     │
│   [OpenTelemetry Traces] [Prometheus Metrics] [Quality Monitor]            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Core Application | 20 | Functional, clean code, well-structured |
| Evaluation System | 15 | Comprehensive datasets, CI integration |
| Security | 15 | Robust guardrails, security tests |
| Deployment | 15 | Docker + K8s, proper configuration |
| Optimization | 15 | Caching, routing, benchmarks |
| Observability | 10 | Tracing, metrics, monitoring |
| Documentation | 10 | Complete, professional quality |

**Total: 100 points**

**Bonus Points (up to +15):**
- +5: Multi-agent coordination
- +5: Custom fine-tuned model integration
- +5: Production deployment (actual cloud deployment)

---

## Timeline

| Week | Milestone |
|------|-----------|
| 8 | Project kickoff, architecture design |
| 9 | Core implementation, initial evaluation |
| 9 (end) | Feature complete, testing |
| 10 | Final polish, documentation, demo prep |

---

## Submission

**Code Submission (Before Demo):**
1. GitHub repository URL
2. Working Docker Compose setup
3. All tests passing
4. Documentation complete

**Demo Day (Week 10):**
1. Live demonstration
2. Presentation slides (optional but recommended)
3. Be prepared for questions

---

## Evaluation Criteria for Demo

| Aspect | Weight | Criteria |
|--------|--------|----------|
| Functionality | 30% | Does it work? Does it solve the problem? |
| Architecture | 25% | Clean design, appropriate patterns |
| Production-Readiness | 25% | Security, observability, resilience |
| Presentation | 20% | Clear communication, handles questions |

---

## Resources

- All course notes and assignments
- Office hours (extended in Week 9)
- Peer review sessions
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Haystack Documentation](https://docs.haystack.deepset.ai/)

---

## FAQ

**Q: Can I work in a team?**
A: Yes, teams of 2 are allowed. Expectations scale accordingly (more features, more comprehensive testing).

**Q: What if my demo fails during presentation?**
A: Have a backup video recording of your demo working. Technical issues happen.

**Q: Can I use frameworks like LangChain?**
A: Yes, but you must understand what they do. Be prepared to explain any framework code.

**Q: How much of the previous assignments can I reuse?**
A: You can reuse all of your previous work. In fact, you should—that's the point.
