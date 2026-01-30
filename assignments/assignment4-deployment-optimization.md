# Assignment 4: Optimized Model Deployment Pipeline

**Released:** Week 7
**Due:** Week 8
**Weight:** 15%

---

## Learning Objectives

By completing this assignment, you will:
1. Containerize an AI application with Docker best practices
2. Implement prompt caching for cost and latency reduction
3. Build an intelligent model router for hybrid deployments
4. Set up monitoring and cost tracking
5. Create deployment configurations for Kubernetes

---

## Overview

You will optimize and containerize the application you've been building, implementing caching strategies, intelligent routing, and preparing it for production deployment with proper observability.

---

## Requirements

### Part 1: Containerization (20 points)

Create production-ready Docker configuration:

```dockerfile
# Dockerfile

# Multi-stage build for smaller image size
FROM python:3.11-slim as builder
# ... build stage

FROM python:3.11-slim as production
# ... production stage with:
# - Non-root user
# - Health check
# - Proper signal handling
```

```yaml
# docker-compose.yaml

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]

  redis:
    image: redis:7-alpine
    # ... configuration

  worker:
    build: .
    command: celery -A src.tasks worker
    # ... configuration
```

**Deliverables:**
- [ ] Multi-stage Dockerfile
- [ ] Docker Compose for local development
- [ ] Non-root user in container
- [ ] Health checks configured
- [ ] Image size < 500MB

### Part 2: Prompt Caching (25 points)

Implement multi-layer caching:

```python
# src/optimization/caching.py

class ResponseCache:
    """Exact match response caching."""

    async def get(self, messages: list, model: str, **kwargs) -> CachedResponse | None:
        """Get cached response if exists."""
        pass

    async def set(self, messages: list, model: str, response: str, **kwargs):
        """Cache a response."""
        pass


class SemanticCache:
    """Similarity-based caching for similar queries."""

    def __init__(self, similarity_threshold: float = 0.95):
        pass

    async def get(self, query: str, context_hash: str) -> tuple[str, float] | None:
        """Get similar cached response."""
        pass

    async def set(self, query: str, response: str, context_hash: str):
        """Cache query-response pair."""
        pass


class CachedLLMService:
    """LLM service with caching layers."""

    async def generate(self, messages: list, use_cache: bool = True, **kwargs):
        """
        Generate with caching:
        1. Check exact match cache
        2. Check semantic cache
        3. Call LLM if cache miss
        4. Store in both caches
        """
        pass
```

**Deliverables:**
- [ ] Exact match cache with Redis
- [ ] Semantic cache with configurable threshold
- [ ] Cache hit/miss metrics
- [ ] Cache invalidation support
- [ ] Tests showing cache effectiveness

### Part 3: Intelligent Routing (25 points)

Build a model router for hybrid deployments:

```python
# src/optimization/routing.py

class ModelRouter:
    """Route requests to appropriate model/provider."""

    def __init__(self):
        self.providers = {}
        self.provider_stats = {}

    def route(self, context: RoutingContext) -> ModelSelection:
        """
        Select model based on:
        - User tier (free/paid/enterprise)
        - Task complexity
        - Latency requirements
        - Current provider health
        - Cost optimization
        """
        pass

    def update_stats(self, provider: str, latency: float, success: bool):
        """Update provider statistics."""
        pass

    def get_fallback(self, primary: str) -> str | None:
        """Get fallback provider if primary unavailable."""
        pass


@dataclass
class RoutingContext:
    user_tier: str
    task_type: str
    estimated_tokens: int
    latency_budget_ms: int
    quality_requirement: str


@dataclass
class ModelSelection:
    provider: str
    model: str
    reason: str
```

**Deliverables:**
- [ ] Routing based on user tier
- [ ] Routing based on task type
- [ ] Latency-based routing
- [ ] Fallback mechanism
- [ ] Provider health tracking

### Part 4: Observability (20 points)

Implement cost tracking and monitoring:

```python
# src/observability/metrics.py

class CostTracker:
    """Track AI costs by user, feature, and model."""

    async def record(self, event: CostEvent):
        """Record a cost event."""
        pass

    async def get_daily_cost(self, date: datetime) -> float:
        """Get total cost for a day."""
        pass

    async def get_cost_by_model(self, start: datetime, end: datetime) -> dict:
        """Get costs broken down by model."""
        pass

    async def check_budget(self, user_id: str | None = None) -> tuple[bool, float]:
        """Check if within budget limits."""
        pass


class PerformanceMonitor:
    """Track latency and throughput metrics."""

    async def record_request(
        self,
        endpoint: str,
        latency_ms: float,
        tokens: int,
        cache_hit: bool,
    ):
        """Record request metrics."""
        pass

    async def get_p99_latency(self, endpoint: str, window_minutes: int) -> float:
        """Get p99 latency for endpoint."""
        pass
```

**Deliverables:**
- [ ] Cost tracking by user and model
- [ ] Budget alerts when approaching limit
- [ ] Latency percentile tracking (p50, p95, p99)
- [ ] Cache hit rate monitoring
- [ ] Metrics exposed for Prometheus scraping

### Part 5: Kubernetes Configuration (10 points)

Create K8s deployment manifests:

```yaml
# k8s/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-api
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: api
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          # ... full configuration

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-api-hpa
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

**Deliverables:**
- [ ] Deployment with resource limits
- [ ] Service configuration
- [ ] HPA for autoscaling
- [ ] ConfigMap for configuration
- [ ] Secret references (not actual secrets)

---

## Testing Requirements

```python
# tests/test_caching.py

async def test_exact_cache_hit():
    """Test that identical requests hit cache."""
    pass

async def test_semantic_cache_similar_queries():
    """Test that similar queries hit semantic cache."""
    pass

async def test_cache_miss_calls_llm():
    """Test that cache misses call the LLM."""
    pass


# tests/test_routing.py

def test_enterprise_gets_best_model():
    """Enterprise users should get GPT-4."""
    pass

def test_latency_constrained_uses_fast_model():
    """Low latency budget should use fast models."""
    pass

def test_fallback_on_provider_failure():
    """Should fall back when primary provider fails."""
    pass
```

---

## Performance Benchmarks

Your implementation should achieve:

| Metric | Target |
|--------|--------|
| Cache hit rate (exact) | >20% on test workload |
| Cache hit rate (semantic) | >10% additional |
| Cold start time | <5 seconds |
| p99 latency (cached) | <100ms |
| Container image size | <500MB |

Include a benchmark report showing your results.

---

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Containerization | 20 | Multi-stage, health checks, small image |
| Caching | 25 | Both caches work, metrics tracked |
| Routing | 25 | Smart routing with fallbacks |
| Observability | 20 | Cost tracking, latency metrics |
| K8s Config | 10 | Valid manifests, HPA configured |
| **Bonus** | +10 | vLLM integration, distributed caching |

---

## Submission

1. All code in Git repository
2. Docker image builds successfully
3. Docker Compose starts all services
4. Benchmark report included
5. K8s manifests in `k8s/` directory

---

## Resources

- Course Notes: Week 6 (Deployment) and Week 7 (Optimization)
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/guidelines/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Redis Caching Patterns](https://redis.io/docs/manual/patterns/)
