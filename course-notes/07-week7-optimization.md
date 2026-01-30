# Week 7: Optimization Techniques

---

## Chapter 12: Model Optimization

### 12.1 The Optimization Landscape

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION TECHNIQUE OVERVIEW                          │
└─────────────────────────────────────────────────────────────────────────────┘

                          Quality
                             ▲
                             │
                   ┌─────────┼─────────┐
                   │         │         │
         FP32 ●────┼─────────┼─────────┼──────  Best quality, highest cost
                   │         │         │
         FP16 ●────┼─────────┼─────────┼──────  Minimal quality loss
                   │         │         │
          FP8 ●────┼─────────┼─────────┼──────  Good balance
                   │         │         │
         INT8 ●────┼─────────┼─────────┼──────  Significant speedup
                   │         │         │
         INT4 ●────┼─────────┼─────────┼──────  Max compression, quality loss
                   │         │         │
                   └─────────┼─────────┘
                             │
   Cost/Latency ◄────────────┴────────────► Low

   TECHNIQUE COMPARISON:
   ┌─────────────────┬──────────────┬─────────────┬──────────────┬───────────┐
   │ Technique       │ Speedup      │ Memory      │ Quality Loss │ Effort    │
   ├─────────────────┼──────────────┼─────────────┼──────────────┼───────────┤
   │ FP16            │ 2x           │ 50%         │ Negligible   │ Trivial   │
   │ INT8 (PTQ)      │ 2-4x         │ 75%         │ Low          │ Low       │
   │ INT8 (QAT)      │ 2-4x         │ 75%         │ Minimal      │ Medium    │
   │ INT4 (GPTQ)     │ 3-4x         │ 87%         │ Moderate     │ Medium    │
   │ Pruning         │ 1.5-3x       │ 40-80%      │ Variable     │ High      │
   │ Distillation    │ 5-100x       │ 90%+        │ Moderate     │ High      │
   │ Flash Attention │ 2-4x         │ 50%+        │ None         │ Trivial   │
   │ Prompt Caching  │ 10-100x*     │ -           │ None         │ Low       │
   └─────────────────┴──────────────┴─────────────┴──────────────┴───────────┘
   * For repeated prefixes
```

**Figure 12.1:** Optimization techniques comparison

### 12.2 Quantization Deep Dive

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUANTIZATION EXPLAINED                              │
└─────────────────────────────────────────────────────────────────────────────┘

   PRECISION FORMATS:

   FP32 (32-bit float):     ████████████████████████████████  32 bits
   Sign: 1 bit, Exponent: 8 bits, Mantissa: 23 bits
   Range: ±3.4 × 10^38

   FP16 (16-bit float):     ████████████████  16 bits
   Sign: 1 bit, Exponent: 5 bits, Mantissa: 10 bits
   Range: ±65504

   BF16 (bfloat16):         ████████████████  16 bits
   Sign: 1 bit, Exponent: 8 bits, Mantissa: 7 bits
   Range: Same as FP32, less precision

   INT8 (8-bit integer):    ████████  8 bits
   Range: -128 to 127 (or 0-255 unsigned)


   QUANTIZATION PROCESS:

   Original FP32 weights:
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  -0.234  0.891  -0.012  0.567  -0.999  0.123  0.456  -0.789            │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
   Determine scale and zero-point:
   scale = (max - min) / 255 = (0.891 - (-0.999)) / 255 = 0.0074
   zero_point = round(-min / scale) = 135

                                    │
                                    ▼
   Quantized INT8 weights:
   ┌─────────────────────────────────────────────────────────────────────────┐
   │     103    255    133    212      0    152    197     28                │
   └─────────────────────────────────────────────────────────────────────────┘


   QUANTIZATION TYPES:

   1. Post-Training Quantization (PTQ)
      ─────────────────────────────────
      • Quantize after training
      • No retraining needed
      • Faster to implement
      • May lose more quality

      [Trained Model] ───▶ [Calibration] ───▶ [Quantized Model]


   2. Quantization-Aware Training (QAT)
      ──────────────────────────────────
      • Simulate quantization during training
      • Model learns to be robust to quantization
      • Better quality preservation
      • Requires retraining

      [Training with Fake Quantization] ───▶ [Quantized Model]
```

**Figure 12.2:** Quantization fundamentals

```python
# src/optimization/quantization.py
"""
Model quantization utilities.
"""

from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_quantized_model(
    model_name: str,
    quantization: str = "int8",  # "int8", "int4", "fp16", "none"
    device_map: str = "auto",
) -> tuple:
    """
    Load a model with the specified quantization.

    Args:
        model_name: HuggingFace model name or path
        quantization: Quantization level
        device_map: Device placement strategy

    Returns:
        (model, tokenizer) tuple
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure quantization
    quantization_config = None

    if quantization == "int8":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,  # Outlier threshold
            llm_int8_has_fp16_weight=False,
        )
    elif quantization == "int4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Normalized float 4-bit
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Nested quantization
        )
    elif quantization == "fp16":
        pass  # Will use torch_dtype below

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.float16 if quantization in ["fp16", "none"] else None,
        trust_remote_code=True,
    )

    return model, tokenizer


def measure_memory_usage(model) -> dict:
    """
    Measure model memory usage.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    torch.cuda.reset_peak_memory_stats()

    # Get model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    return {
        "parameter_memory_mb": param_size / (1024 ** 2),
        "buffer_memory_mb": buffer_size / (1024 ** 2),
        "total_mb": (param_size + buffer_size) / (1024 ** 2),
        "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
    }


# GPTQ Quantization for maximum compression
def load_gptq_model(
    model_name: str,
    bits: int = 4,
    group_size: int = 128,
):
    """
    Load a GPTQ-quantized model.

    GPTQ provides better quality than naive INT4 by using
    second-order information during quantization.
    """
    from auto_gptq import AutoGPTQForCausalLM

    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        use_safetensors=True,
        device_map="auto",
        use_triton=False,  # Set True for faster inference with Triton
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
```

### 12.3 Knowledge Distillation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      KNOWLEDGE DISTILLATION                                 │
└─────────────────────────────────────────────────────────────────────────────┘

   CONCEPT:
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                                                                         │
   │   Teacher Model (Large)                Student Model (Small)            │
   │   ┌─────────────────────┐             ┌─────────────────────┐          │
   │   │                     │             │                     │          │
   │   │   GPT-4 / Claude    │───────────▶│    GPT-4-mini       │          │
   │   │   (175B params)     │  Transfer  │    (8B params)      │          │
   │   │                     │  Knowledge │                     │          │
   │   └─────────────────────┘             └─────────────────────┘          │
   │                                                                         │
   │   Capabilities:                       Capabilities:                     │
   │   • Complex reasoning                 • Simpler reasoning               │
   │   • Broad knowledge                   • Focused knowledge               │
   │   • High accuracy                     • Good accuracy                   │
   │                                                                         │
   │   Cost: $$$$$                         Cost: $                           │
   │   Latency: 2000ms                     Latency: 200ms                    │
   └─────────────────────────────────────────────────────────────────────────┘


   DISTILLATION PROCESS:
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                                                                         │
   │   1. Generate training data with teacher                                │
   │      ┌─────────────┐                                                    │
   │      │   Query     │───▶ Teacher ───▶ High-quality response            │
   │      └─────────────┘                                                    │
   │                                                                         │
   │   2. Train student on teacher outputs                                   │
   │      ┌─────────────────────────────────────────────────────────────┐   │
   │      │ Input: Query                                                 │   │
   │      │ Target: Teacher's response (soft labels)                     │   │
   │      │ Loss: KL divergence + task loss                             │   │
   │      └─────────────────────────────────────────────────────────────┘   │
   │                                                                         │
   │   3. Evaluate and iterate                                               │
   │      • Compare student vs teacher on eval set                          │
   │      • Add hard examples where student fails                           │
   │      • Repeat until quality target met                                 │
   └─────────────────────────────────────────────────────────────────────────┘


   DISTILLATION STRATEGIES:

   ┌─────────────────────┬─────────────────────────────────────────────────┐
   │ Strategy            │ Description                                     │
   ├─────────────────────┼─────────────────────────────────────────────────┤
   │ Response Distill    │ Train on teacher's final outputs               │
   │ Logit Distillation  │ Match teacher's probability distributions      │
   │ Feature Distillation│ Match intermediate representations             │
   │ Chain-of-Thought    │ Include teacher's reasoning in training        │
   │ Selective Distill   │ Focus on hard/important examples               │
   └─────────────────────┴─────────────────────────────────────────────────┘
```

**Figure 12.3:** Knowledge distillation overview

```python
# src/optimization/distillation.py
"""
Knowledge distillation for creating efficient models.
"""

import asyncio
from dataclasses import dataclass
from typing import Iterator
import json
from pathlib import Path

from src.services.llm import LLMService


@dataclass
class DistillationExample:
    """A single distillation training example."""
    input: str
    teacher_output: str
    teacher_reasoning: str | None = None
    metadata: dict = None


class TeacherDataGenerator:
    """
    Generate training data from a teacher model.
    """

    def __init__(
        self,
        teacher_service: LLMService,
        teacher_model: str = "gpt-4o",
    ):
        self.teacher = teacher_service
        self.model = teacher_model

    async def generate_example(
        self,
        prompt: str,
        include_reasoning: bool = True,
    ) -> DistillationExample:
        """
        Generate a single training example.
        """
        if include_reasoning:
            # Use chain-of-thought prompting
            system_prompt = """You are a helpful assistant. When answering:
1. First, explain your reasoning step by step in <reasoning> tags
2. Then provide your final answer in <answer> tags

Example:
<reasoning>
The user is asking about X. I need to consider Y and Z...
</reasoning>
<answer>
The answer is...
</answer>"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = await self.teacher.generate(
            messages=messages,
            model=self.model,
            temperature=0.3,  # Lower temperature for consistency
        )

        # Parse reasoning and answer
        content = response.content
        reasoning = None
        answer = content

        if include_reasoning:
            if "<reasoning>" in content and "</reasoning>" in content:
                reasoning_start = content.find("<reasoning>") + len("<reasoning>")
                reasoning_end = content.find("</reasoning>")
                reasoning = content[reasoning_start:reasoning_end].strip()

            if "<answer>" in content and "</answer>" in content:
                answer_start = content.find("<answer>") + len("<answer>")
                answer_end = content.find("</answer>")
                answer = content[answer_start:answer_end].strip()

        return DistillationExample(
            input=prompt,
            teacher_output=answer,
            teacher_reasoning=reasoning,
            metadata={
                "model": self.model,
                "tokens_used": response.input_tokens + response.output_tokens,
            }
        )

    async def generate_dataset(
        self,
        prompts: list[str],
        output_path: Path,
        include_reasoning: bool = True,
        max_concurrent: int = 5,
    ):
        """
        Generate a full distillation dataset.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_generate(prompt: str) -> DistillationExample:
            async with semaphore:
                return await self.generate_example(prompt, include_reasoning)

        # Generate all examples
        tasks = [bounded_generate(p) for p in prompts]
        examples = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors and save
        valid_examples = [e for e in examples if isinstance(e, DistillationExample)]

        # Save as JSONL
        with output_path.open("w") as f:
            for ex in valid_examples:
                f.write(json.dumps({
                    "input": ex.input,
                    "output": ex.teacher_output,
                    "reasoning": ex.teacher_reasoning,
                    "metadata": ex.metadata,
                }) + "\n")

        return len(valid_examples)


# Training data format for fine-tuning
def format_for_openai_finetuning(
    examples: list[DistillationExample],
    include_reasoning: bool = False,
) -> list[dict]:
    """
    Format distillation examples for OpenAI fine-tuning.
    """
    formatted = []

    for ex in examples:
        if include_reasoning and ex.teacher_reasoning:
            assistant_content = f"Let me think through this:\n{ex.teacher_reasoning}\n\nAnswer: {ex.teacher_output}"
        else:
            assistant_content = ex.teacher_output

        formatted.append({
            "messages": [
                {"role": "user", "content": ex.input},
                {"role": "assistant", "content": assistant_content},
            ]
        })

    return formatted
```

---

## Chapter 13: Inference Optimization

### 13.1 Prompt Caching

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROMPT CACHING STRATEGIES                          │
└─────────────────────────────────────────────────────────────────────────────┘

   WITHOUT CACHING:
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                                                                         │
   │  Request 1: [System Prompt (2000 tokens)][User: "Hi"]                  │
   │             ────────────────────────────────────────▶ Process all      │
   │                                                                         │
   │  Request 2: [System Prompt (2000 tokens)][User: "Help"]                │
   │             ────────────────────────────────────────▶ Process all      │
   │                                                                         │
   │  Request 3: [System Prompt (2000 tokens)][User: "Thanks"]              │
   │             ────────────────────────────────────────▶ Process all      │
   │                                                                         │
   │  Total tokens processed: 6000+ system prompt tokens (redundant!)       │
   └─────────────────────────────────────────────────────────────────────────┘


   WITH PROMPT CACHING:
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                                                                         │
   │  Request 1: [System Prompt (2000 tokens)][User: "Hi"]                  │
   │             ─────────────────────────────────────────▶ Process & Cache │
   │                    │                                                    │
   │                    ▼                                                    │
   │             ┌─────────────────┐                                         │
   │             │  KV Cache       │  Stores computed attention states      │
   │             │  (System only)  │                                         │
   │             └────────┬────────┘                                         │
   │                      │                                                  │
   │  Request 2: [Cached──┘][User: "Help"]                                  │
   │             ─────────────────────────▶ Process only new tokens         │
   │                                                                         │
   │  Request 3: [Cached──┘][User: "Thanks"]                                │
   │             ─────────────────────────▶ Process only new tokens         │
   │                                                                         │
   │  Tokens saved: 4000+ (66% reduction!)                                  │
   └─────────────────────────────────────────────────────────────────────────┘


   CACHING STRATEGIES:

   1. PREFIX CACHING (Same prefix across requests)
      ┌──────────────────────────────────────────────────────────────────────┐
      │  Best for: System prompts, few-shot examples, shared context        │
      │  Savings: 50-90% token reduction                                    │
      │  Implementation: Most LLM APIs support this natively                │
      └──────────────────────────────────────────────────────────────────────┘

   2. SEMANTIC CACHING (Similar queries)
      ┌──────────────────────────────────────────────────────────────────────┐
      │  Best for: FAQ-style queries, repeated questions                    │
      │  Savings: 100% for cache hits                                       │
      │  Implementation: Embed query → search cache → return if similar     │
      └──────────────────────────────────────────────────────────────────────┘

   3. RESPONSE CACHING (Exact matches)
      ┌──────────────────────────────────────────────────────────────────────┐
      │  Best for: Deterministic queries, idempotent operations             │
      │  Savings: 100% for cache hits                                       │
      │  Implementation: Hash request → lookup → return cached response     │
      └──────────────────────────────────────────────────────────────────────┘
```

**Figure 13.1:** Prompt caching strategies

```python
# src/optimization/caching.py
"""
Caching strategies for LLM inference.
"""

import hashlib
import json
from typing import Any
from dataclasses import dataclass
import time
import numpy as np
from redis.asyncio import Redis


@dataclass
class CacheEntry:
    """A cached response."""
    response: str
    model: str
    created_at: float
    hit_count: int = 0
    tokens_saved: int = 0


class ResponseCache:
    """
    Exact-match response caching.
    """

    def __init__(
        self,
        redis: Redis,
        ttl_seconds: int = 3600,
        prefix: str = "llm_cache:",
    ):
        self.redis = redis
        self.ttl = ttl_seconds
        self.prefix = prefix

    def _hash_request(self, messages: list[dict], model: str, **kwargs) -> str:
        """Create deterministic hash of request."""
        content = json.dumps({
            "messages": messages,
            "model": model,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    async def get(
        self,
        messages: list[dict],
        model: str,
        **kwargs
    ) -> CacheEntry | None:
        """Get cached response if exists."""
        key = self.prefix + self._hash_request(messages, model, **kwargs)
        data = await self.redis.get(key)

        if data:
            entry = CacheEntry(**json.loads(data))
            # Update hit count
            entry.hit_count += 1
            await self.redis.setex(
                key,
                self.ttl,
                json.dumps(entry.__dict__)
            )
            return entry

        return None

    async def set(
        self,
        messages: list[dict],
        model: str,
        response: str,
        tokens_used: int,
        **kwargs
    ):
        """Cache a response."""
        key = self.prefix + self._hash_request(messages, model, **kwargs)
        entry = CacheEntry(
            response=response,
            model=model,
            created_at=time.time(),
            tokens_saved=tokens_used,
        )
        await self.redis.setex(
            key,
            self.ttl,
            json.dumps(entry.__dict__)
        )


class SemanticCache:
    """
    Semantic similarity-based caching.

    Returns cached responses for semantically similar queries.
    """

    def __init__(
        self,
        redis: Redis,
        embedding_service,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
    ):
        self.redis = redis
        self.embedder = embedding_service
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds

    async def get(
        self,
        query: str,
        context_hash: str = "",  # Hash of system prompt/context
    ) -> tuple[str, float] | None:
        """
        Get semantically similar cached response.

        Returns (response, similarity_score) or None.
        """
        # Get query embedding
        query_embedding = await self.embedder.embed(query)

        # Search cache for similar queries
        # In production, use vector DB for this
        cache_key = f"semantic_cache:{context_hash}:*"
        keys = await self.redis.keys(cache_key)

        best_match = None
        best_score = 0.0

        for key in keys:
            data = await self.redis.get(key)
            if not data:
                continue

            entry = json.loads(data)
            cached_embedding = np.array(entry["embedding"])

            # Cosine similarity
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )

            if similarity > self.threshold and similarity > best_score:
                best_match = entry["response"]
                best_score = similarity

        if best_match:
            return best_match, best_score

        return None

    async def set(
        self,
        query: str,
        response: str,
        context_hash: str = "",
    ):
        """Cache a query-response pair with embedding."""
        query_embedding = await self.embedder.embed(query)

        # Use query hash as key
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        key = f"semantic_cache:{context_hash}:{query_hash}"

        entry = {
            "query": query,
            "response": response,
            "embedding": query_embedding.tolist(),
            "created_at": time.time(),
        }

        await self.redis.setex(key, self.ttl, json.dumps(entry))


class CachedLLMService:
    """
    LLM service with multi-layer caching.
    """

    def __init__(
        self,
        llm_service,
        response_cache: ResponseCache,
        semantic_cache: SemanticCache | None = None,
    ):
        self.llm = llm_service
        self.response_cache = response_cache
        self.semantic_cache = semantic_cache

    async def generate(
        self,
        messages: list[dict],
        model: str = "gpt-4o-mini",
        use_cache: bool = True,
        **kwargs
    ):
        """Generate with caching."""

        # Layer 1: Exact match cache
        if use_cache:
            cached = await self.response_cache.get(messages, model, **kwargs)
            if cached:
                return type('CachedResponse', (), {
                    'content': cached.response,
                    'model': model,
                    'from_cache': True,
                    'tokens_saved': cached.tokens_saved,
                })()

        # Layer 2: Semantic cache (for user message)
        if use_cache and self.semantic_cache and len(messages) > 0:
            last_user_msg = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                None
            )
            if last_user_msg:
                # Hash context (all messages except last user message)
                context_hash = hashlib.sha256(
                    json.dumps(messages[:-1]).encode()
                ).hexdigest()[:16]

                semantic_hit = await self.semantic_cache.get(
                    last_user_msg,
                    context_hash
                )
                if semantic_hit:
                    response, similarity = semantic_hit
                    return type('SemanticCacheResponse', (), {
                        'content': response,
                        'model': model,
                        'from_cache': True,
                        'semantic_similarity': similarity,
                    })()

        # Cache miss - call LLM
        response = await self.llm.generate(messages, model=model, **kwargs)

        # Store in caches
        if use_cache:
            await self.response_cache.set(
                messages,
                model,
                response.content,
                response.input_tokens + response.output_tokens,
                **kwargs
            )

            if self.semantic_cache and len(messages) > 0:
                last_user_msg = next(
                    (m["content"] for m in reversed(messages) if m["role"] == "user"),
                    None
                )
                if last_user_msg:
                    context_hash = hashlib.sha256(
                        json.dumps(messages[:-1]).encode()
                    ).hexdigest()[:16]
                    await self.semantic_cache.set(
                        last_user_msg,
                        response.content,
                        context_hash
                    )

        response.from_cache = False
        return response
```

### 13.2 Batching Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BATCHING STRATEGIES                                 │
└─────────────────────────────────────────────────────────────────────────────┘

   1. STATIC BATCHING
      ────────────────
      Wait for fixed batch size, then process together.

      ┌─────────────────────────────────────────────────────────────────────┐
      │  Time ──▶                                                           │
      │                                                                     │
      │  Requests:  R1 ─┐                                                   │
      │             R2 ─┼─┐                                                 │
      │             R3 ─┼─┼─┐                                               │
      │             R4 ─┼─┼─┼──▶ [Batch Process] ──▶ All responses         │
      │                 │ │ │                                               │
      │                 Wait for batch                                      │
      │                                                                     │
      │  Pros: Simple, predictable                                         │
      │  Cons: Latency for early requests, underutilized for sparse traffic│
      └─────────────────────────────────────────────────────────────────────┘


   2. DYNAMIC BATCHING
      ─────────────────
      Process available requests after timeout or size threshold.

      ┌─────────────────────────────────────────────────────────────────────┐
      │  Time ──▶                                                           │
      │                                                                     │
      │  Requests:  R1 ─┐                                                   │
      │                 │ timeout                                           │
      │             R2 ─┼──▶ [Batch R1,R2] ──▶ Responses                   │
      │                                                                     │
      │             R3 ─┐                                                   │
      │             R4 ─┼                                                   │
      │             R5 ─┼                                                   │
      │             R6 ─┼──▶ [Batch R3-R6] ──▶ Responses (size threshold)  │
      │                                                                     │
      │  Pros: Better latency, adapts to traffic                           │
      │  Cons: More complex, variable batch sizes                          │
      └─────────────────────────────────────────────────────────────────────┘


   3. CONTINUOUS BATCHING (vLLM/TensorRT-LLM)
      ───────────────────────────────────────
      Dynamically add/remove requests from batch as they complete.

      ┌─────────────────────────────────────────────────────────────────────┐
      │  Time ──▶                                                           │
      │                                                                     │
      │  GPU Batch:  [R1][R2][R3][__][__]                                   │
      │                    │                                                │
      │              R2 completes, R4 joins                                 │
      │                    ▼                                                │
      │              [R1][R4][R3][__][__]                                   │
      │                    │                                                │
      │              R1 completes, R5 joins                                 │
      │                    ▼                                                │
      │              [R5][R4][R3][__][__]                                   │
      │                                                                     │
      │  Pros: Maximum GPU utilization, best throughput                    │
      │  Cons: Requires specialized serving engine                         │
      └─────────────────────────────────────────────────────────────────────┘
```

**Figure 13.2:** Batching strategies comparison

```python
# src/optimization/batching.py
"""
Request batching for LLM inference.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any
import time
from collections import deque


@dataclass
class BatchRequest:
    """A request waiting to be batched."""
    id: str
    messages: list[dict]
    kwargs: dict
    future: asyncio.Future
    submitted_at: float = field(default_factory=time.time)


class DynamicBatcher:
    """
    Dynamic batching for LLM requests.
    """

    def __init__(
        self,
        llm_service,
        max_batch_size: int = 8,
        max_wait_ms: int = 50,
    ):
        self.llm = llm_service
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: deque[BatchRequest] = deque()
        self.lock = asyncio.Lock()
        self._running = False

    async def start(self):
        """Start the batching loop."""
        self._running = True
        asyncio.create_task(self._batch_loop())

    async def stop(self):
        """Stop the batching loop."""
        self._running = False

    async def generate(
        self,
        messages: list[dict],
        **kwargs
    ) -> Any:
        """
        Submit a request for batched processing.
        """
        future = asyncio.Future()
        request = BatchRequest(
            id=f"req_{time.time_ns()}",
            messages=messages,
            kwargs=kwargs,
            future=future,
        )

        async with self.lock:
            self.queue.append(request)

        return await future

    async def _batch_loop(self):
        """Main batching loop."""
        while self._running:
            batch = await self._collect_batch()

            if batch:
                # Process batch
                results = await self._process_batch(batch)

                # Resolve futures
                for request, result in zip(batch, results):
                    if isinstance(result, Exception):
                        request.future.set_exception(result)
                    else:
                        request.future.set_result(result)
            else:
                await asyncio.sleep(0.01)  # Small sleep if no requests

    async def _collect_batch(self) -> list[BatchRequest]:
        """Collect requests into a batch."""
        batch = []
        start_time = time.time()

        while len(batch) < self.max_batch_size:
            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.max_wait_ms and batch:
                break

            # Try to get a request
            async with self.lock:
                if self.queue:
                    batch.append(self.queue.popleft())
                elif batch:
                    break  # No more requests, process what we have
                else:
                    break  # Nothing to process

            if not batch:
                await asyncio.sleep(0.001)

        return batch

    async def _process_batch(
        self,
        batch: list[BatchRequest]
    ) -> list[Any]:
        """
        Process a batch of requests.

        In production, this would call a batched inference endpoint.
        For API-based LLMs, we run them concurrently.
        """
        tasks = [
            self.llm.generate(req.messages, **req.kwargs)
            for req in batch
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)


class EmbeddingBatcher:
    """
    Specialized batcher for embedding requests.

    Embeddings benefit greatly from batching because:
    1. Single API call for multiple texts
    2. Better GPU utilization
    3. Lower per-item latency
    """

    def __init__(
        self,
        embedding_service,
        max_batch_size: int = 100,
        max_wait_ms: int = 100,
    ):
        self.embedder = embedding_service
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: list[tuple[str, asyncio.Future]] = []
        self.lock = asyncio.Lock()
        self._task = None

    async def embed(self, text: str) -> list[float]:
        """Embed a single text with batching."""
        future = asyncio.Future()

        async with self.lock:
            self.queue.append((text, future))

            # Start batch task if not running
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._process_after_wait())

        return await future

    async def _process_after_wait(self):
        """Wait then process batch."""
        await asyncio.sleep(self.max_wait_ms / 1000)

        async with self.lock:
            if not self.queue:
                return

            # Take up to max_batch_size
            batch = self.queue[:self.max_batch_size]
            self.queue = self.queue[self.max_batch_size:]

        texts = [t for t, _ in batch]
        futures = [f for _, f in batch]

        try:
            # Batch embed
            embeddings = await self.embedder.embed_batch(texts)

            # Resolve futures
            for future, embedding in zip(futures, embeddings):
                future.set_result(embedding)

        except Exception as e:
            for future in futures:
                future.set_exception(e)
```

---

### Summary: Week 7

In this week, we covered:

1. **Quantization**: FP16, INT8, INT4, GPTQ techniques
2. **Knowledge distillation**: Creating smaller models from larger ones
3. **Prompt caching**: Prefix, semantic, and response caching
4. **Batching strategies**: Static, dynamic, and continuous batching
5. **Optimization trade-offs**: When optimization hurts quality

**Key Takeaways:**

- Quantization offers easy wins: INT8 is often "free" quality-wise
- Distillation enables custom small models for specific tasks
- Caching can reduce costs by 50-90% for repetitive workloads
- Batching is essential for throughput but adds latency complexity

---

### Exercises

**Exercise 7.1:** Measure the quality difference between FP16 and INT8 quantized versions of a model on your evaluation set.

**Exercise 7.2:** Implement a semantic cache with a 0.92 similarity threshold and measure the hit rate on production-like queries.

**Exercise 7.3:** Build a dynamic batcher and measure the throughput improvement vs. individual requests.

---

*Next Week: Observability and Operations—Making AI systems understandable*
