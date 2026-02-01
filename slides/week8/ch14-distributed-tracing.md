---
marp: true
theme: course-theme
paginate: true
---

<!-- _class: lead -->

# Chapter 14
## Distributed Tracing for AI

---

## 14.1 Why AI Systems Need Specialized Observability

### War Story: Silent Degradation in Production AI

- A 2025 study of **156 high-impact production AI incidents** found the majority of failures were **silent**
  - Systems appeared healthy by traditional metrics (latency, error rates, uptime)
  - But were producing **wrong outputs**
- ~75% of incidents involved non-reasoning LLMs; the rest affected embedding models

> Traditional observability misses these failures because there are no errors, timeouts, or obvious crashes.

---

## 14.1 The Silent Degradation Problem

### Why Traditional Monitoring Fails for AI

- **91% of ML models degrade over time** in production due to:
  - Data drift
  - Feature mismatches between training and serving
  - Stale retrieval layers

- A poisoned or stale retrieval pipeline can be **worse than no retrieval at all**
  - Feeds the model wrong context with high confidence

- Only **AI-specific observability** catches silent degradation:
  - Embedding dimensions & drift
  - Retrieval relevance scores
  - Response quality metrics
  - Data lineage across the full pipeline

---

<!-- _class: diagram -->

## 14.1 The AI Observability Pyramid

![AI Observability Pyramid](../diagrams/ch14-observability-pyramid.svg)

**Figure 14.1:** Traditional vs AI observability and the AI observability pyramid

---

## 14.1 Traditional vs AI Observability

| Traditional Observability | AI Observability (adds) |
|---|---|
| Request latency | Token usage and costs |
| Error rates | Model response quality |
| Throughput | Retrieval relevance |
| CPU/Memory usage | Embedding drift |
| Database queries | Prompt variations |
| | Hallucination detection |
| | Safety violations |
| | A/B test metrics |

---

## 14.1 The AI Observability Pyramid

Four layers, bottom to top:

1. **Infrastructure** -- Logs, metrics, traditional APM
2. **Business Metrics** -- Tokens, costs, latency, throughput
3. **AI Traces** -- LLM calls, RAG steps, tool invocations
4. **Quality Metrics** -- LLM-as-judge scores, user feedback, accuracy

> Each layer builds on the one below. You need all four for full AI observability.

---

## 14.2 OpenTelemetry for AI Workloads

### Setting Up OTel Tracing

```python
# src/observability/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

def setup_tracing(
    service_name: str = "ai-service",
    otlp_endpoint: str = "localhost:4317",
):
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
    })
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)
```

---

## 14.2 Tracing LLM Calls

```python
def trace_llm_call(model: str | None = None):
    """Decorator for tracing LLM calls.
    Captures: model name, input/output tokens, latency, cost."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(
                "llm.generate", kind=trace.SpanKind.CLIENT,
            ) as span:
                span.set_attribute("llm.model",
                    model or kwargs.get("model", "unknown"))
                span.set_attribute("llm.provider", "openai")
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    if hasattr(result, "input_tokens"):
                        span.set_attribute("llm.tokens.input", result.input_tokens)
                    if hasattr(result, "output_tokens"):
                        span.set_attribute("llm.tokens.output", result.output_tokens)
                    if hasattr(result, "cost_usd"):
                        span.set_attribute("llm.cost_usd", result.cost_usd)
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("llm.latency_ms", latency_ms)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator
```

---

## 14.2 Tracing Retrieval Operations

```python
def trace_retrieval(index_name: str | None = None):
    """Decorator for tracing retrieval operations.
    Captures: index/collection, query, num results, top score."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(
                "retrieval.search", kind=trace.SpanKind.CLIENT,
            ) as span:
                span.set_attribute("retrieval.index",
                    index_name or "unknown")
                query = kwargs.get("query") or (args[0] if args else "")
                if isinstance(query, str):
                    span.set_attribute("retrieval.query", query[:500])
                try:
                    result = await func(*args, **kwargs)
                    if isinstance(result, list):
                        span.set_attribute("retrieval.num_results", len(result))
                        if result and hasattr(result[0], "score"):
                            span.set_attribute("retrieval.top_score",
                                result[0].score)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator
```

---

## 14.2 Tracing Embedding Operations

```python
def trace_embedding(model: str | None = None):
    """Decorator for tracing embedding operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(
                "embedding.generate", kind=trace.SpanKind.CLIENT,
            ) as span:
                span.set_attribute("embedding.model", model or "unknown")
                inputs = (kwargs.get("texts") or kwargs.get("input")
                          or args[0] if args else [])
                if isinstance(inputs, str):
                    span.set_attribute("embedding.num_inputs", 1)
                elif isinstance(inputs, list):
                    span.set_attribute("embedding.num_inputs", len(inputs))
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        return wrapper
    return decorator
```

---

## 14.2 AI-Specific Span Attributes Summary

| Decorator | Span Name | Key Attributes |
|---|---|---|
| `trace_llm_call` | `llm.generate` | `llm.model`, `llm.tokens.input`, `llm.tokens.output`, `llm.cost_usd`, `llm.latency_ms` |
| `trace_retrieval` | `retrieval.search` | `retrieval.index`, `retrieval.query`, `retrieval.num_results`, `retrieval.top_score` |
| `trace_embedding` | `embedding.generate` | `embedding.model`, `embedding.num_inputs` |

All decorators follow the same pattern:
- Start a span with `SpanKind.CLIENT`
- Record domain-specific attributes
- Set status OK or ERROR
- Record exceptions on failure

---

## 14.2 Context Manager for Complex Traces

```python
class AITraceContext:
    """Context manager for tracing complex AI operations."""
    def __init__(self, operation_name: str,
                 user_id: str | None = None,
                 session_id: str | None = None):
        self.operation_name = operation_name
        self.user_id = user_id
        self.session_id = session_id
        self._metrics = {}

    def __enter__(self):
        self.span = tracer.start_span(
            self.operation_name, kind=trace.SpanKind.SERVER)
        self.span.__enter__()
        if self.user_id:
            self.span.set_attribute("user.id", self.user_id)
        if self.session_id:
            self.span.set_attribute("session.id", self.session_id)
        self.start_time = time.perf_counter()
        return self

    def record_quality_score(self, score: float):
        self._metrics["ai.quality_score"] = score

    def record_tokens(self, input_tokens: int, output_tokens: int):
        self._metrics["ai.tokens.input"] = input_tokens
        self._metrics["ai.tokens.output"] = output_tokens
        self._metrics["ai.tokens.total"] = input_tokens + output_tokens
```

---

## 14.3 LangSmith Integration

### Setup

```python
# src/observability/langsmith.py
from langsmith import Client
from langsmith.run_trees import RunTree
import os

def setup_langsmith(
    api_key: str | None = None,
    project_name: str = "production-ai",
):
    """Configure LangSmith for tracing."""
    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project_name
    os.environ["LANGSMITH_TRACING"] = "true"
    return Client()
```

- Three environment variables to set:
  - `LANGSMITH_API_KEY` -- authentication
  - `LANGSMITH_PROJECT` -- project grouping
  - `LANGSMITH_TRACING` -- enable/disable

---

## 14.3 LangSmith Tracer Class

```python
class LangSmithTracer:
    """LangSmith-based tracing for AI operations."""
    def __init__(self, project_name: str = "production-ai"):
        self.client = Client()
        self.project = project_name

    def trace_chain(self, name: str):
        """Decorator for tracing chain/pipeline executions."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                run = RunTree(
                    name=name, run_type="chain",
                    project_name=self.project,
                )
                try:
                    run.inputs = {"args": str(args)[:1000],
                                  "kwargs": str(kwargs)[:1000]}
                    result = await func(*args, **kwargs)
                    run.outputs = {"result": str(result)[:1000]}
                    run.end()
                    return result
                except Exception as e:
                    run.error = str(e)
                    run.end()
                    raise
            return wrapper
        return decorator
```

---

## 14.3 LangSmith Feedback & LangGraph Integration

```python
    async def log_feedback(
        self, run_id: str, score: float,
        feedback_type: str = "user_rating",
        comment: str | None = None,
    ):
        """Log user feedback for a run."""
        self.client.create_feedback(
            run_id=run_id, key=feedback_type,
            score=score, comment=comment,
        )
```

### LangGraph Auto-Tracing

```python
def create_traced_graph():
    from langgraph.graph import StateGraph
    from langchain_openai import ChatOpenAI

    # LangSmith automatically traces LangChain/LangGraph
    llm = ChatOpenAI(model="gpt-4o-mini")
    # Build graph...
    # All nodes automatically traced in LangSmith
```

> LangSmith provides automatic tracing for LangChain and LangGraph -- no decorators needed.

---

## 14.4 Cost Tracking and Attribution

### Cost Event & Rate Table

```python
@dataclass
class CostEvent:
    """A single cost event."""
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    user_id: str | None = None
    feature: str | None = None
    request_id: str | None = None
```

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|---|---|---|
| gpt-4o | $0.0025 | $0.010 |
| gpt-4o-mini | $0.00015 | $0.0006 |
| claude-3-opus | $0.015 | $0.075 |
| claude-3.5-sonnet | $0.003 | $0.015 |
| claude-3-haiku | $0.00025 | $0.00125 |
| text-embedding-3-small | $0.00002 | $0 |

---

## 14.4 Cost Calculation

```python
COST_RATES = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "text-embedding-3-small": {"input": 0.00002, "output": 0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0},
}

def calculate_cost(
    model: str, input_tokens: int, output_tokens: int,
) -> float:
    """Calculate cost for a request."""
    rates = COST_RATES.get(model, {"input": 0.01, "output": 0.03})
    return (
        (input_tokens / 1000) * rates["input"] +
        (output_tokens / 1000) * rates["output"]
    )
```

---

## 14.4 CostTracker: Recording Events

```python
class CostTracker:
    """Track and attribute AI costs."""
    def __init__(self, redis: Redis):
        self.redis = redis

    async def record(self, event: CostEvent):
        """Record a cost event."""
        key = f"costs:{event.timestamp.strftime('%Y-%m-%d')}"
        await self.redis.lpush(key, json.dumps({
            "timestamp": event.timestamp.isoformat(),
            "model": event.model,
            "input_tokens": event.input_tokens,
            "output_tokens": event.output_tokens,
            "cost_usd": event.cost_usd,
            "user_id": event.user_id,
            "feature": event.feature,
            "request_id": event.request_id,
        }))
        await self.redis.expire(key, 60 * 60 * 24 * 30)  # 30 days
        await self._update_aggregates(event)
```

**Storage strategy:** Redis with 30-day TTL, plus real-time aggregates

---

## 14.4 CostTracker: Aggregation Dimensions

```python
    async def _update_aggregates(self, event: CostEvent):
        """Update cost aggregates."""
        date = event.timestamp.strftime('%Y-%m-%d')

        # Daily total
        await self.redis.incrbyfloat(f"costs:daily:{date}", event.cost_usd)

        # By model
        await self.redis.incrbyfloat(
            f"costs:model:{event.model}:{date}", event.cost_usd)

        # By user
        if event.user_id:
            await self.redis.incrbyfloat(
                f"costs:user:{event.user_id}:{date}", event.cost_usd)

        # By feature
        if event.feature:
            await self.redis.incrbyfloat(
                f"costs:feature:{event.feature}:{date}", event.cost_usd)
```

Four aggregation dimensions: **daily total**, **by model**, **by user**, **by feature**

---

## 14.4 CostTracker: Budget Checks

```python
    async def check_budget(
        self,
        user_id: str | None = None,
        daily_limit: float = 100.0,
    ) -> tuple[bool, float]:
        """Check if within budget. Returns (within_budget, current_spend)"""
        today = datetime.utcnow()
        if user_id:
            key = f"costs:user:{user_id}:{today.strftime('%Y-%m-%d')}"
        else:
            key = f"costs:daily:{today.strftime('%Y-%m-%d')}"
        current = await self.redis.get(key)
        current_spend = float(current) if current else 0.0
        return current_spend < daily_limit, current_spend
```

### Cost Attribution Strategy

- **Per-request**: Every LLM call records model, tokens, cost
- **Per-user**: Track spending per user for rate limiting
- **Per-feature**: Attribute costs to product features
- **Per-model**: Compare cost efficiency across providers

---

<!-- _class: lead -->

## Key Takeaways -- Chapter 14

- **Traditional observability misses AI-specific failures** -- silent degradation is the norm
- **OpenTelemetry** provides vendor-neutral tracing with AI-specific span attributes for LLM calls, retrieval, and embeddings
- **LangSmith** offers turnkey AI observability with automatic LangChain/LangGraph tracing
- **Cost tracking** requires attribution across multiple dimensions: user, feature, model, and time
- **Budget checks** should be real-time using Redis aggregates to prevent cost overruns
