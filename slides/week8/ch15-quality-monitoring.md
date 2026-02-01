---
marp: true
theme: course-theme
paginate: true
---

<!-- _class: lead -->

# Chapter 15
## Quality Monitoring and Incident Response

---

## 15.1 Quality Monitoring Dashboard

### The AI Quality Dashboard

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AI QUALITY DASHBOARD                           │
│  Time Range: [Last 24h]    Refresh: [Auto]    Compare: [Previous]  │
├─────────────────────────────────────────────────────────────────────┤
│  HEALTH SCORE                                                       │
│    [████████████████████░░░░░] 87/100  (↓3 from yesterday)         │
│                                                                     │
│    Components:                                                      │
│    • Response Quality:  92/100  ✓                                  │
│    • Latency Score:     85/100  ⚠                                  │
│    • Error Rate:        95/100  ✓                                  │
│    • Cost Efficiency:   78/100  ⚠                                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Figure 15.1:** Top section of an AI quality monitoring dashboard

---

## 15.1 Dashboard: Quality Trends & Error Distribution

```
┌───────────────────────────────┬─────────────────────────────────┐
│  RESPONSE QUALITY TREND       │  ERROR DISTRIBUTION              │
│                               │                                  │
│  Score                        │  ┌────────────┐                  │
│   100│     ╭─╮   ╭───╮       │  │ Timeout    │████████ 34%     │
│    90│╭───╯  ╰───╯    ╰──    │  │ Rate Limit │██████ 28%       │
│    80│                        │  │ Quality    │████ 22%         │
│    70│                        │  │ Other      │██ 16%           │
│      └────────────────────    │  └────────────┘                  │
│       6h  12h  18h  24h       │                                  │
└───────────────────────────────┴─────────────────────────────────┘
```

Key metrics to track:
- **Response quality** over time (rolling window)
- **Error distribution** by type (timeout, rate limit, quality, other)

---

## 15.1 Dashboard: Issues, Costs & Alerts

```
┌───────────────────────────────┬─────────────────────────────────┐
│  TOP ISSUES (Last 24h)        │  COST BREAKDOWN                  │
│                               │                                  │
│  1. Slow retrieval (p99: 2.3s)│  Total: $847.23                 │
│     23 occurrences            │  gpt-4o     │████████ $412      │
│                               │  gpt-4o-mini│████ $234          │
│  2. Quality drop in RAG       │  embeddings │██ $89             │
│     "refund policy" queries   │  other      │█ $112             │
│     12 low scores             │                                  │
│                               │  Budget: $1000/day               │
│  3. Rate limiting spikes      │  Remaining: $152.77              │
│     Peak: 14:00 UTC           │                                  │
├───────────────────────────────┴─────────────────────────────────┤
│  RECENT ALERTS                                                    │
│  🔴 14:23 Quality score dropped below 80% for 5 minutes         │
│  🟡 12:45 Latency p99 exceeded 3s threshold                     │
│  🟢 10:30 Rate limit warning resolved                            │
│  🔴 08:15 Daily cost exceeded 80% of budget                     │
└───────────────────────────────────────────────────────────────────┘
```

---

## 15.1 Quality Data Structures

```python
# src/observability/quality_monitor.py
from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class QualitySnapshot:
    """Point-in-time quality measurement."""
    timestamp: datetime
    overall_score: float
    correctness: float
    relevance: float
    helpfulness: float
    safety: float
    sample_size: int

@dataclass
class QualityAlert:
    """Quality alert definition."""
    name: str
    severity: str       # "critical", "warning", "info"
    threshold: float
    window_minutes: int
    metric: str          # "overall", "correctness", "relevance", etc.
```

---

## 15.1 QualityMonitor: Initialization

```python
class QualityMonitor:
    """Continuous quality monitoring for AI systems."""

    def __init__(
        self,
        redis: Redis,
        judge: LLMJudge,
        sample_rate: float = 0.01,  # Sample 1% of requests
    ):
        self.redis = redis
        self.judge = judge
        self.sample_rate = sample_rate
        self.alerts: List[QualityAlert] = []

    def add_alert(self, alert: QualityAlert):
        """Add a quality alert."""
        self.alerts.append(alert)

    async def should_sample(self) -> bool:
        """Determine if this request should be sampled."""
        import random
        return random.random() < self.sample_rate
```

**Key design decision:** Sample only 1% of requests to control evaluation cost while maintaining statistical significance.

---

## 15.1 QualityMonitor: Evaluate and Record

```python
    async def evaluate_and_record(
        self,
        query: str,
        response: str,
        request_id: str,
        reference: str | None = None,
    ):
        """Evaluate a response and record the quality score."""
        # Run LLM-as-judge evaluation
        result = await self.judge.single_rating(
            query=query,
            response=response,
            reference=reference,
        )

        # Store in Redis
        timestamp = datetime.utcnow()
        await self._store_score(timestamp, result, request_id)

        # Check alerts
        await self._check_alerts(result)

        return result
```

Three-step pipeline: **evaluate** -> **store** -> **check alerts**

---

## 15.1 QualityMonitor: Score Storage

```python
    async def _store_score(self, timestamp, result, request_id):
        """Store quality score in Redis."""
        key = f"quality:{timestamp.strftime('%Y-%m-%d:%H')}"
        score_data = {
            "timestamp": timestamp.isoformat(),
            "request_id": request_id,
            "overall": result.overall,
            "correctness": result.correctness,
            "relevance": result.relevance,
            "helpfulness": result.helpfulness,
            "safety": result.safety,
        }
        await self.redis.lpush(key, str(score_data))
        await self.redis.expire(key, 60 * 60 * 24 * 7)  # 7 days
        await self._update_averages(result)
```

- **Key format:** `quality:YYYY-MM-DD:HH` -- hourly buckets
- **TTL:** 7 days
- Running averages updated in parallel

---

## 15.1 QualityMonitor: Running Averages

```python
    async def _update_averages(self, result: SingleRatingResult):
        """Update running average scores."""
        window_key = f"quality:avg:{datetime.utcnow().strftime('%Y-%m-%d:%H')}"

        pipe = self.redis.pipeline()
        pipe.lpush(f"{window_key}:overall", result.overall)
        pipe.lpush(f"{window_key}:correctness", result.correctness)
        pipe.lpush(f"{window_key}:relevance", result.relevance)
        pipe.lpush(f"{window_key}:helpfulness", result.helpfulness)
        pipe.lpush(f"{window_key}:safety", result.safety)

        # Set expiry for each
        for metric in ["overall", "correctness", "relevance",
                       "helpfulness", "safety"]:
            pipe.expire(f"{window_key}:{metric}", 60 * 60 * 25)

        await pipe.execute()
```

Uses **Redis pipelining** for efficient batch writes across 5 metric dimensions.

---

## 15.1 QualityMonitor: Alert System

```python
    async def _check_alerts(self, result: SingleRatingResult):
        """Check if any alerts should fire."""
        for alert in self.alerts:
            score = getattr(result, alert.metric, result.overall)
            if score < alert.threshold:
                await self._fire_alert(alert, score)

    async def _fire_alert(self, alert: QualityAlert, current_score: float):
        """Fire a quality alert."""
        alert_key = f"alert:quality:{alert.name}"

        # Debounce: don't fire same alert within window
        if await self.redis.exists(alert_key):
            return

        await self.redis.setex(
            alert_key,
            alert.window_minutes * 60,
            str({"name": alert.name, "severity": alert.severity,
                 "threshold": alert.threshold,
                 "current_score": current_score,
                 "fired_at": datetime.utcnow().isoformat()})
        )
        # In production: send to PagerDuty, Slack, etc.
```

**Debouncing** via Redis TTL prevents alert storms.

---

## 15.1 QualityMonitor: Quality Trends

```python
    async def get_quality_trend(self, hours: int = 24) -> List[QualitySnapshot]:
        """Get quality trend over time."""
        snapshots = []
        now = datetime.utcnow()

        for hour_offset in range(hours):
            timestamp = now - timedelta(hours=hour_offset)
            window_key = f"quality:avg:{timestamp.strftime('%Y-%m-%d:%H')}"
            overall_scores = await self.redis.lrange(
                f"{window_key}:overall", 0, -1)
            if not overall_scores:
                continue
            overall = [float(s) for s in overall_scores]
            snapshots.append(QualitySnapshot(
                timestamp=timestamp,
                overall_score=sum(overall) / len(overall),
                correctness=await self._get_avg(f"{window_key}:correctness"),
                relevance=await self._get_avg(f"{window_key}:relevance"),
                helpfulness=await self._get_avg(f"{window_key}:helpfulness"),
                safety=await self._get_avg(f"{window_key}:safety"),
                sample_size=len(overall),
            ))
        return sorted(snapshots, key=lambda s: s.timestamp)
```

---

## 15.2 Incident Response Playbooks

### Incident Classification

| Severity | Description | Response Time |
|---|---|---|
| **SEV1 -- Critical** | Complete outage, data breach, >50% requests failing | Immediate, all-hands |
| **SEV2 -- High** | >20% quality drop, major feature broken, >200% cost spike | Within 1 hour |
| **SEV3 -- Medium** | Minor quality issues, elevated latency, partial degradation | Within 4 hours |
| **SEV4 -- Low** | Cosmetic issues, non-critical bugs | Next business day |

---

<!-- _class: diagram -->

## 15.2 Incident Response Flow

![Incident Response Playbook](../diagrams/ch15-incident-response.svg)

**Figure 15.2:** Full incident response playbook from classification to post-mortem

---

## 15.2 Step 1: Assess (0-5 min)

**Trigger:** Quality score drops >15% for >10 minutes

Immediate assessment actions:

1. **Check quality dashboard** for affected dimensions
   - Is it correctness, relevance, safety, or all?
2. **Identify scope** -- global or segment-specific?
   - Specific queries, users, or features?
3. **Review recent deployments** (last 24h)
   - Code changes, config changes, model updates?
4. **Check external dependencies**
   - OpenAI status page, vector DB health, network issues?

> Goal: Understand the blast radius and likely cause within 5 minutes.

---

## 15.2 Step 2: Mitigate (5-15 min)

Choose the right mitigation based on root cause:

| Root Cause | Mitigation |
|---|---|
| Recent deployment | **Rollback** to last known good version |
| External dependency down | **Switch to fallback model** |
| General degradation | **Enable degraded mode** (cached responses) |
| Rate limiting | **Scale up workers** |
| Overload | **Enable request shedding** for non-critical traffic |

> Priority: Stop the bleeding first, fix root cause later.

---

## 15.2 Steps 3-5: Communicate, Resolve, Post-Mortem

### Step 3: Communicate (15-30 min)
- Update **status page**
- Notify affected customers if user-facing
- Post in **#incidents** Slack channel
- Page additional responders if needed

### Step 4: Resolve (30+ min)
- Identify **root cause**
- Implement fix with proper testing
- Deploy through **normal pipeline** (not hotfix shortcuts)
- Verify quality scores recovered

### Step 5: Post-Mortem (24-48h)
- Schedule **blameless post-mortem**
- Document timeline and actions
- Define **action items** to prevent recurrence
- Add **regression test cases**
- Update runbooks

---

## Exercises

**Exercise 8.1:** Implement an OpenTelemetry span that tracks the complete RAG pipeline from query to response.

**Exercise 8.2:** Build a cost alert that fires when spending exceeds $X in a 1-hour window.

**Exercise 8.3:** Create a quality monitoring job that samples 1% of requests and stores scores in Redis.

**Exercise 8.4:** Write an incident response playbook for "Model provider API is down."

---

<!-- _class: lead -->

## Key Takeaways -- Chapter 15

- **Quality dashboards** should combine health scores, quality trends, error distribution, cost breakdown, and alerts in one view
- **Continuous quality monitoring** uses LLM-as-judge on sampled production traffic (e.g., 1% sample rate)
- **Alert debouncing** via Redis TTL prevents alert fatigue during extended incidents
- **Incident classification** (SEV1-SEV4) determines response urgency and team involvement
- **Post-mortems are essential** -- blameless, with concrete action items and regression tests
