# Prerequisite 1: Python Proficiency

> **Course:** Building Production AI Systems
> **Purpose:** Refresh the Python features you will use daily in this course -- classes, type hints, and async/await. This is not a full Python tutorial; it assumes you already write Python and need a targeted refresher on the patterns that matter for production AI work.

---

## Table of Contents

1. [Classes and OOP](#1-classes-and-oop)
2. [Type Hints](#2-type-hints)
3. [Async/Await](#3-asyncawait)
4. [Cheat Sheet](#4-cheat-sheet)

---

## 1. Classes and OOP

Production AI systems are built from layers of abstraction -- provider clients, prompt managers, evaluation harnesses, pipeline stages. All of these are classes. You need to be comfortable with every feature below.

### 1.1 Basic Class Definition

```python
class LLMRequest:
    """A single request to an LLM provider."""

    def __init__(self, model: str, prompt: str, max_tokens: int = 1024):
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self._created_at = time.time()

    def __repr__(self) -> str:
        return (
            f"LLMRequest(model={self.model!r}, "
            f"prompt={self.prompt[:40]!r}..., "
            f"max_tokens={self.max_tokens})"
        )

    def __str__(self) -> str:
        return f"[{self.model}] {self.prompt[:80]}"
```

**Key points:**

- `__init__` is the initializer, not the constructor (`__new__` is the constructor -- you almost never override it).
- `__repr__` is for developers (unambiguous). `__str__` is for users (readable).
- Use `!r` inside f-strings to get the `repr()` of a value (adds quotes around strings).

### 1.2 Instance Methods, Class Methods, Static Methods

```python
import time
from datetime import datetime


class LLMResponse:
    _request_count: int = 0  # class variable, shared across all instances

    def __init__(self, text: str, model: str, latency_ms: float):
        self.text = text
        self.model = model
        self.latency_ms = latency_ms
        self.timestamp = time.time()
        LLMResponse._request_count += 1

    # Instance method -- operates on self
    def token_count(self) -> int:
        """Rough token estimate (whitespace split)."""
        return len(self.text.split())

    # Class method -- operates on the class, not an instance
    @classmethod
    def get_request_count(cls) -> int:
        return cls._request_count

    # Static method -- no access to self or cls, just logically grouped here
    @staticmethod
    def estimate_cost(token_count: int, price_per_1k: float) -> float:
        return (token_count / 1000) * price_per_1k
```

**When to use each:**

| Decorator | First arg | Access to instance? | Access to class? | Use case |
|---|---|---|---|---|
| *(none)* | `self` | Yes | Via `self.__class__` | Most methods |
| `@classmethod` | `cls` | No | Yes | Factory methods, class-level state |
| `@staticmethod` | *(none)* | No | No | Utility functions scoped to the class |

### 1.3 Properties

Properties let you expose computed values as attributes and add validation to setters.

```python
class TokenBudget:
    def __init__(self, max_tokens: int):
        self._max_tokens = max_tokens
        self._used_tokens = 0

    @property
    def remaining(self) -> int:
        return self._max_tokens - self._used_tokens

    @property
    def used_tokens(self) -> int:
        return self._used_tokens

    @used_tokens.setter
    def used_tokens(self, value: int) -> None:
        if value < 0:
            raise ValueError("used_tokens cannot be negative")
        if value > self._max_tokens:
            raise ValueError(f"used_tokens ({value}) exceeds max ({self._max_tokens})")
        self._used_tokens = value


budget = TokenBudget(4096)
budget.used_tokens = 1000
print(budget.remaining)  # 3096
```

### 1.4 Inheritance and Abstract Base Classes

In production AI code, you define abstract interfaces for providers and implement them concretely. This is the dominant pattern in this course.

```python
from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Abstract interface every LLM provider must implement."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Send a completion request. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def stream(self, prompt: str, **kwargs):
        """Stream a completion. Must be implemented by subclasses."""
        ...

    def _build_headers(self) -> dict[str, str]:
        """Shared helper -- concrete method on the base class."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


class OpenAIProvider(BaseLLMProvider):
    BASE_URL = "https://api.openai.com/v1"

    def complete(self, prompt: str, **kwargs) -> str:
        # In reality, you'd call the API here
        print(f"Calling OpenAI ({self.model}) with: {prompt[:50]}")
        return "OpenAI response placeholder"

    def stream(self, prompt: str, **kwargs):
        print(f"Streaming from OpenAI ({self.model})...")
        yield "chunk1"
        yield "chunk2"


class AnthropicProvider(BaseLLMProvider):
    BASE_URL = "https://api.anthropic.com/v1"

    def complete(self, prompt: str, **kwargs) -> str:
        print(f"Calling Anthropic ({self.model}) with: {prompt[:50]}")
        return "Anthropic response placeholder"

    def stream(self, prompt: str, **kwargs):
        print(f"Streaming from Anthropic ({self.model})...")
        yield "chunk1"
        yield "chunk2"


# Usage -- program against the interface
def run_evaluation(provider: BaseLLMProvider, prompts: list[str]) -> list[str]:
    return [provider.complete(p) for p in prompts]


# Swap providers without changing evaluation code
provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
results = run_evaluation(provider, ["Explain transformers.", "What is RLHF?"])
```

**Why this matters:** Every major AI framework (LangChain, LlamaIndex, Haystack) uses this pattern. You will build your own provider abstractions in this course.

### 1.5 Dataclasses

Dataclasses remove boilerplate for classes that are primarily data containers. You will use them (and Pydantic models, covered later) constantly.

```python
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    prompt: str
    expected: str
    actual: str
    score: float
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.score >= 0.8


# __init__, __repr__, __eq__ are auto-generated
result = EvalResult(
    prompt="What is 2+2?",
    expected="4",
    actual="4",
    score=1.0,
    metadata={"model": "gpt-4o"},
)
print(result)
# EvalResult(prompt='What is 2+2?', expected='4', actual='4', score=1.0, metadata={'model': 'gpt-4o'})
print(result.passed)  # True
```

**Important:** Use `field(default_factory=dict)` for mutable defaults (lists, dicts, sets). A bare `metadata: dict = {}` would share the same dict object across all instances -- a classic Python bug.

### 1.6 Practical Example: LLM Provider Abstraction

Bringing it all together -- a realistic provider abstraction you might build in this course.

```python
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class CompletionResult:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


class BaseLLMProvider(ABC):
    def __init__(self, model: str, api_key: str, timeout: float = 30.0):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._call_count = 0

    @abstractmethod
    def chat(self, messages: list[Message], **kwargs) -> CompletionResult:
        ...

    def _track_call(self) -> float:
        """Start tracking a call. Returns start time."""
        self._call_count += 1
        return time.time()

    @classmethod
    def from_env(cls, model: str) -> "BaseLLMProvider":
        """Factory: build a provider pulling the API key from env vars."""
        import os
        key_var = cls.ENV_KEY_NAME  # subclasses define this
        api_key = os.environ.get(key_var)
        if not api_key:
            raise EnvironmentError(f"Set {key_var} in your environment")
        return cls(model=model, api_key=api_key)


class OpenAIProvider(BaseLLMProvider):
    ENV_KEY_NAME = "OPENAI_API_KEY"

    def chat(self, messages: list[Message], **kwargs) -> CompletionResult:
        start = self._track_call()
        # -- In production, you'd call openai.ChatCompletion.create() here --
        response_text = f"[OpenAI {self.model}] placeholder response"
        latency = (time.time() - start) * 1000
        return CompletionResult(
            text=response_text,
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            latency_ms=latency,
        )


class AnthropicProvider(BaseLLMProvider):
    ENV_KEY_NAME = "ANTHROPIC_API_KEY"

    def chat(self, messages: list[Message], **kwargs) -> CompletionResult:
        start = self._track_call()
        response_text = f"[Anthropic {self.model}] placeholder response"
        latency = (time.time() - start) * 1000
        return CompletionResult(
            text=response_text,
            model=self.model,
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            latency_ms=latency,
        )


# --- Usage ---
provider = OpenAIProvider(model="gpt-4o", api_key="sk-test")
result = provider.chat([
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Explain attention mechanisms."),
])
print(f"Response: {result.text}")
print(f"Tokens used: {result.total_tokens}")
print(f"Latency: {result.latency_ms:.1f}ms")
```

---

## 2. Type Hints

Python type hints don't enforce types at runtime by default, but they enable:

- Editor autocompletion and error detection.
- Static analysis with `mypy` or `pyright`.
- Runtime validation when paired with Pydantic.
- Self-documenting code that is easier to maintain across a team.

In production AI systems, type hints are not optional. They catch bugs before deployment.

### 2.1 Basic Types

```python
name: str = "gpt-4o"
temperature: float = 0.7
max_tokens: int = 4096
stream: bool = False
```

### 2.2 Collection Types

Since Python 3.9 you can use built-in collection types directly. Since Python 3.10 you can use `X | Y` union syntax.

```python
# Lists, dicts, tuples, sets
models: list[str] = ["gpt-4o", "claude-sonnet-4-20250514"]
token_limits: dict[str, int] = {"gpt-4o": 128_000, "claude-sonnet-4-20250514": 200_000}
coordinates: tuple[float, float] = (37.7749, -122.4194)
unique_tags: set[str] = {"production", "eval", "v2"}

# Variable-length tuples
scores: tuple[float, ...] = (0.9, 0.85, 0.92, 0.88)

# Nested types
conversations: list[list[dict[str, str]]] = [
    [{"role": "user", "content": "Hello"}],
    [{"role": "user", "content": "Hi"}],
]
```

### 2.3 Optional, Union, and the Modern `X | Y` Syntax

```python
from typing import Optional, Union

# These two are equivalent:
api_key: Optional[str] = None
api_key: str | None = None  # preferred (Python 3.10+)

# Union for multiple types
token_count: Union[int, float] = 100
token_count: int | float = 100  # preferred (Python 3.10+)
```

### 2.4 Type Aliases

```python
from typing import TypeAlias

# Simple alias
MessageDict: TypeAlias = dict[str, str]

# Complex alias -- makes signatures readable
ConversationHistory: TypeAlias = list[list[MessageDict]]
ModelConfig: TypeAlias = dict[str, str | int | float | bool]


def format_conversation(history: ConversationHistory) -> str:
    lines = []
    for turn in history:
        for msg in turn:
            lines.append(f"{msg['role']}: {msg['content']}")
    return "\n".join(lines)
```

### 2.5 Callable

```python
from typing import Callable

# A function that takes a string and returns a float
Scorer: TypeAlias = Callable[[str, str], float]


def run_eval(prompt: str, response: str, scorer: Scorer) -> float:
    return scorer(prompt, response)


# Usage
def simple_scorer(prompt: str, response: str) -> float:
    return 1.0 if len(response) > 0 else 0.0

score = run_eval("test", "answer", simple_scorer)
```

### 2.6 Generics and TypeVar

Generics let you write functions and classes that work with any type while preserving type information.

```python
from typing import TypeVar, Generic

T = TypeVar("T")


class Cache(Generic[T]):
    def __init__(self) -> None:
        self._store: dict[str, T] = {}

    def get(self, key: str) -> T | None:
        return self._store.get(key)

    def set(self, key: str, value: T) -> None:
        self._store[key] = value


# Type checker knows response_cache stores strings
response_cache: Cache[str] = Cache()
response_cache.set("prompt_1", "The answer is 42.")
result: str | None = response_cache.get("prompt_1")

# Type checker knows embedding_cache stores list[float]
embedding_cache: Cache[list[float]] = Cache()
embedding_cache.set("doc_1", [0.1, 0.2, 0.3])
```

### 2.7 Literal and Annotated

```python
from typing import Literal, Annotated

# Literal constrains to specific values
Role = Literal["system", "user", "assistant"]
Model = Literal["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"]


def create_message(role: Role, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


create_message("user", "Hello")        # OK
create_message("moderator", "Hello")   # mypy error: not a valid Literal


# Annotated adds metadata (used heavily by Pydantic, FastAPI)
from pydantic import Field

Temperature = Annotated[float, Field(ge=0.0, le=2.0, description="Sampling temperature")]
MaxTokens = Annotated[int, Field(ge=1, le=128_000, description="Max output tokens")]
```

### 2.8 Protocol (Structural Subtyping)

Protocol defines an interface based on structure, not inheritance. If a class has the right methods, it satisfies the protocol -- no base class needed.

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class Completable(Protocol):
    def complete(self, prompt: str) -> str: ...


class OpenAIClient:
    def complete(self, prompt: str) -> str:
        return "openai response"


class LocalModel:
    def complete(self, prompt: str) -> str:
        return "local response"


# Both satisfy Completable without inheriting from it
def run(client: Completable, prompt: str) -> str:
    return client.complete(prompt)


run(OpenAIClient(), "test")   # OK
run(LocalModel(), "test")     # OK

# runtime_checkable lets you do isinstance checks
print(isinstance(OpenAIClient(), Completable))  # True
```

### 2.9 Pydantic BaseModel

Pydantic is the de facto standard for validated data models in production Python. This course uses it extensively for API schemas, configuration, and structured outputs from LLMs.

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, description="Message text")


class ChatRequest(BaseModel):
    model: str = Field(default="gpt-4o", description="Model identifier")
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=128_000)
    stream: bool = False

    @field_validator("model")
    @classmethod
    def model_must_be_known(cls, v: str) -> str:
        allowed = {"gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"}
        if v not in allowed:
            raise ValueError(f"Unknown model: {v}. Must be one of {allowed}")
        return v


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    text: str
    model: str
    usage: TokenUsage
    latency_ms: float


# --- Usage ---

# Pydantic validates on construction
request = ChatRequest(
    messages=[
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is RLHF?"),
    ],
    temperature=0.5,
)
print(request.model_dump_json(indent=2))

# Validation errors are clear and structured
try:
    bad_request = ChatRequest(
        messages=[],  # min_length=1 violated
        temperature=5.0,  # le=2.0 violated
    )
except Exception as e:
    print(e)

# Parse from JSON (common when reading API responses)
raw_json = '{"id": "abc-123", "text": "RLHF is...", "model": "gpt-4o", "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60}, "latency_ms": 432.1}'
response = ChatResponse.model_validate_json(raw_json)
print(f"Tokens used: {response.usage.total_tokens}")
```

**Key Pydantic features you will use in this course:**

| Feature | Purpose |
|---|---|
| `Field(...)` | Validation constraints and metadata |
| `@field_validator` | Custom validation logic |
| `.model_dump()` | Convert to dict |
| `.model_dump_json()` | Convert to JSON string |
| `.model_validate()` | Parse from dict |
| `.model_validate_json()` | Parse from JSON string |
| `model_config` | Class-level config (e.g., `extra = "forbid"`) |

---

## 3. Async/Await

### 3.1 Why Async Matters for AI Workloads

LLM API calls are I/O-bound operations. A typical call to GPT-4o or Claude takes 500ms to 30s. If you make 100 calls sequentially, you wait for the sum of all latencies. With async, you wait for the *maximum* latency instead.

```
Sequential:  call1 (2s) -> call2 (1s) -> call3 (3s) = 6 seconds total
Async:       call1 (2s) |
             call2 (1s) | = 3 seconds total (limited by slowest call)
             call3 (3s) |
```

This is the single biggest performance lever in production AI systems.

### 3.2 Asyncio Basics

```python
import asyncio


async def fetch_completion(prompt: str) -> str:
    """Simulate an LLM API call with network latency."""
    print(f"Sending: {prompt[:40]}...")
    await asyncio.sleep(1.0)  # simulate 1s API latency
    print(f"Received response for: {prompt[:40]}")
    return f"Response to: {prompt}"


async def main():
    # Sequential -- takes ~3 seconds
    r1 = await fetch_completion("What is attention?")
    r2 = await fetch_completion("What is RLHF?")
    r3 = await fetch_completion("What is RAG?")
    print(f"Sequential results: {len([r1, r2, r3])} responses")


asyncio.run(main())
```

**Key rules:**

- `async def` defines a coroutine function.
- `await` pauses the coroutine and yields control to the event loop.
- `asyncio.run()` is the entry point -- call it once from synchronous code.
- You can only use `await` inside an `async def` function.

### 3.3 asyncio.gather for Parallel Calls

This is the pattern you will use most often.

```python
import asyncio
import time


async def call_llm(prompt: str, delay: float = 1.0) -> str:
    """Simulate an LLM API call."""
    await asyncio.sleep(delay)
    return f"Response to: {prompt}"


async def main():
    prompts = [
        "Explain transformers.",
        "What is RLHF?",
        "How does RAG work?",
        "What are embeddings?",
        "Explain chain-of-thought prompting.",
    ]

    # --- Sequential ---
    start = time.perf_counter()
    sequential_results = []
    for p in prompts:
        result = await call_llm(p)
        sequential_results.append(result)
    seq_time = time.perf_counter() - start
    print(f"Sequential: {seq_time:.2f}s for {len(sequential_results)} calls")

    # --- Parallel with gather ---
    start = time.perf_counter()
    parallel_results = await asyncio.gather(*[call_llm(p) for p in prompts])
    par_time = time.perf_counter() - start
    print(f"Parallel:   {par_time:.2f}s for {len(parallel_results)} calls")
    print(f"Speedup:    {seq_time / par_time:.1f}x")


asyncio.run(main())
# Sequential: 5.00s for 5 calls
# Parallel:   1.00s for 5 calls
# Speedup:    5.0x
```

**`asyncio.gather` behavior:**

- Returns results in the same order as the input coroutines.
- By default, if one task fails, all results are lost. Use `return_exceptions=True` to get exceptions as return values instead.

```python
results = await asyncio.gather(
    call_llm("prompt1"),
    call_llm("prompt2"),
    call_llm("prompt3"),
    return_exceptions=True,  # don't let one failure kill everything
)

for r in results:
    if isinstance(r, Exception):
        print(f"Error: {r}")
    else:
        print(f"Success: {r}")
```

### 3.4 Controlling Concurrency with Semaphores

API providers have rate limits. You need to cap concurrency.

```python
import asyncio


async def call_llm_api(prompt: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:  # only N concurrent calls allowed
        print(f"  -> Calling API for: {prompt[:30]}")
        await asyncio.sleep(1.0)  # simulate API latency
        return f"Response to: {prompt}"


async def main():
    max_concurrent = 3  # respect API rate limits
    semaphore = asyncio.Semaphore(max_concurrent)

    prompts = [f"Prompt {i}" for i in range(10)]

    results = await asyncio.gather(
        *[call_llm_api(p, semaphore) for p in prompts]
    )
    print(f"Got {len(results)} results")


asyncio.run(main())
```

### 3.5 Async Context Managers

Context managers that need to do async setup/teardown (like opening HTTP connections).

```python
import asyncio


class AsyncLLMClient:
    """An async client that manages an HTTP session lifecycle."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self._session = None

    async def __aenter__(self):
        # In production: self._session = aiohttp.ClientSession(...)
        print("Opening HTTP session...")
        self._session = "mock_session"
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # In production: await self._session.close()
        print("Closing HTTP session...")
        self._session = None
        return False  # don't suppress exceptions

    async def complete(self, prompt: str) -> str:
        if not self._session:
            raise RuntimeError("Client not initialized. Use 'async with'.")
        await asyncio.sleep(0.5)
        return f"Response to: {prompt}"


async def main():
    async with AsyncLLMClient("https://api.example.com", "sk-test") as client:
        result = await client.complete("What is attention?")
        print(result)
    # session is automatically closed here


asyncio.run(main())
```

### 3.6 Async Generators and Iterators

Async generators are essential for streaming LLM responses.

```python
import asyncio
from typing import AsyncIterator


async def stream_response(prompt: str) -> AsyncIterator[str]:
    """Simulate a streaming LLM response."""
    tokens = ["The", " answer", " to", " your", " question", " is", " 42", "."]
    for token in tokens:
        await asyncio.sleep(0.1)  # simulate token-by-token arrival
        yield token


async def main():
    full_response = ""
    async for token in stream_response("What is the meaning of life?"):
        print(token, end="", flush=True)
        full_response += token
    print()  # newline
    print(f"Full response: {full_response}")


asyncio.run(main())
```

### 3.7 Async HTTP with httpx

`httpx` is the recommended async HTTP library (it also has a sync API, so you only need one library). `aiohttp` is an alternative.

```python
import asyncio
import httpx


async def call_openai(prompt: str, api_key: str) -> dict:
    """Make an async call to the OpenAI API."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
            },
        )
        response.raise_for_status()
        return response.json()


async def batch_call(prompts: list[str], api_key: str) -> list[dict]:
    """Call the API for all prompts in parallel."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Reuse the same client for connection pooling
        async def single_call(prompt: str) -> dict:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                },
            )
            response.raise_for_status()
            return response.json()

        return await asyncio.gather(*[single_call(p) for p in prompts])
```

### 3.8 Practical Example: Parallel LLM Evaluation

A realistic async evaluation harness -- the kind of thing you will build in this course.

```python
import asyncio
import time
from dataclasses import dataclass


@dataclass
class EvalCase:
    prompt: str
    expected: str


@dataclass
class EvalResult:
    prompt: str
    expected: str
    actual: str
    score: float
    latency_ms: float


async def call_llm(prompt: str) -> str:
    """Simulate an async LLM call with variable latency."""
    latency = 0.5 + (hash(prompt) % 10) / 10  # 0.5-1.5s
    await asyncio.sleep(latency)
    return f"Simulated answer to: {prompt}"


def score_response(expected: str, actual: str) -> float:
    """Simple scoring function."""
    return 1.0 if expected.lower() in actual.lower() else 0.0


async def evaluate_single(
    case: EvalCase,
    semaphore: asyncio.Semaphore,
) -> EvalResult:
    async with semaphore:
        start = time.perf_counter()
        actual = await call_llm(case.prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        score = score_response(case.expected, actual)
        return EvalResult(
            prompt=case.prompt,
            expected=case.expected,
            actual=actual,
            score=score,
            latency_ms=latency_ms,
        )


async def run_evaluation(
    cases: list[EvalCase],
    max_concurrent: int = 5,
) -> list[EvalResult]:
    semaphore = asyncio.Semaphore(max_concurrent)
    results = await asyncio.gather(
        *[evaluate_single(case, semaphore) for case in cases],
        return_exceptions=True,
    )

    # Separate successes and failures
    successes = [r for r in results if isinstance(r, EvalResult)]
    failures = [r for r in results if isinstance(r, Exception)]

    if failures:
        print(f"WARNING: {len(failures)} calls failed")
        for f in failures:
            print(f"  Error: {f}")

    return successes


async def main():
    cases = [
        EvalCase("What is 2+2?", "4"),
        EvalCase("Capital of France?", "Paris"),
        EvalCase("Largest planet?", "Jupiter"),
        EvalCase("Speed of light?", "299"),
        EvalCase("H2O is?", "water"),
        EvalCase("Pi to 2 decimals?", "3.14"),
        EvalCase("Boiling point of water in C?", "100"),
        EvalCase("Author of Hamlet?", "Shakespeare"),
    ]

    print(f"Running {len(cases)} evaluations...")
    start = time.perf_counter()

    results = await run_evaluation(cases, max_concurrent=3)

    total_time = time.perf_counter() - start
    avg_score = sum(r.score for r in results) / len(results) if results else 0
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

    print(f"\nResults:")
    print(f"  Total time:      {total_time:.2f}s")
    print(f"  Cases evaluated: {len(results)}")
    print(f"  Average score:   {avg_score:.2%}")
    print(f"  Average latency: {avg_latency:.0f}ms")

    for r in results:
        status = "PASS" if r.score >= 0.8 else "FAIL"
        print(f"  [{status}] {r.prompt[:40]} ({r.latency_ms:.0f}ms)")


asyncio.run(main())
```

### 3.9 Common Pitfalls

**Pitfall 1: Blocking the event loop**

```python
import asyncio
import time


# BAD -- time.sleep() blocks the entire event loop
async def bad_example():
    await asyncio.gather(
        blocking_call("a"),
        blocking_call("b"),
    )

async def blocking_call(name: str) -> str:
    time.sleep(2)  # BLOCKS! No other coroutine can run during this.
    return name


# GOOD -- use asyncio.sleep() or run blocking code in an executor
async def good_example():
    await asyncio.gather(
        non_blocking_call("a"),
        non_blocking_call("b"),
    )

async def non_blocking_call(name: str) -> str:
    await asyncio.sleep(2)  # yields control to event loop
    return name


# If you MUST call blocking code (e.g., a sync library), use run_in_executor:
async def run_blocking_in_executor():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, time.sleep, 2)
    return result
```

**Pitfall 2: Forgetting `await`**

```python
import asyncio


async def get_response() -> str:
    await asyncio.sleep(1)
    return "hello"


async def main():
    # BAD -- result is a coroutine object, not a string!
    result = get_response()
    print(type(result))  # <class 'coroutine'>
    # Python will warn: "RuntimeWarning: coroutine 'get_response' was never awaited"

    # GOOD
    result = await get_response()
    print(type(result))  # <class 'str'>


asyncio.run(main())
```

**Pitfall 3: Creating the client outside `async with`**

```python
import httpx

# BAD -- client created at module level, never properly closed
client = httpx.AsyncClient()

# GOOD -- use async with to manage lifecycle
async def make_request():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com")
        return response.json()

# ALSO GOOD -- create once, close explicitly (for long-lived services)
class MyService:
    def __init__(self):
        self.client = httpx.AsyncClient()

    async def close(self):
        await self.client.aclose()
```

**Pitfall 4: Mixing sync and async incorrectly**

```python
import asyncio

# BAD -- calling asyncio.run() inside an already-running event loop
async def outer():
    # This will raise RuntimeError
    # asyncio.run(inner())  # DON'T DO THIS
    await inner()  # Do this instead


async def inner():
    return "result"
```

---

## 4. Cheat Sheet

### Classes

```python
# Basic class
class Foo:
    class_var: int = 0                  # shared across instances

    def __init__(self, x: int):         # initializer
        self.x = x                      # instance variable

    def method(self) -> int:            # instance method
        return self.x

    @classmethod
    def from_str(cls, s: str) -> "Foo": # factory method
        return cls(int(s))

    @staticmethod
    def validate(x: int) -> bool:       # utility, no self/cls
        return x > 0

    @property
    def doubled(self) -> int:           # computed attribute
        return self.x * 2

    def __repr__(self) -> str:          # developer string
        return f"Foo(x={self.x})"

    def __str__(self) -> str:           # user-facing string
        return f"Foo with value {self.x}"

# Abstract base class
from abc import ABC, abstractmethod
class Base(ABC):
    @abstractmethod
    def do_thing(self) -> str: ...

# Dataclass
from dataclasses import dataclass, field
@dataclass
class Item:
    name: str
    tags: list[str] = field(default_factory=list)
```

### Type Hints

```python
# Primitives
x: int = 1
y: float = 1.0
s: str = "hi"
b: bool = True

# Collections (Python 3.9+)
items: list[str] = []
mapping: dict[str, int] = {}
pair: tuple[int, str] = (1, "a")
var_tuple: tuple[int, ...] = (1, 2, 3)
tags: set[str] = set()

# Optional / Union (Python 3.10+)
val: str | None = None                  # Optional[str]
num: int | float = 1                    # Union[int, float]

# Callable
from typing import Callable
fn: Callable[[str], int] = len

# Literal
from typing import Literal
role: Literal["user", "system"] = "user"

# TypeVar and Generic
from typing import TypeVar, Generic
T = TypeVar("T")
class Box(Generic[T]):
    def __init__(self, val: T): self.val = val

# Protocol
from typing import Protocol
class HasName(Protocol):
    name: str

# TypeAlias
from typing import TypeAlias
JsonDict: TypeAlias = dict[str, "JsonValue"]

# Pydantic
from pydantic import BaseModel, Field
class Config(BaseModel):
    model: str = Field(default="gpt-4o")
    temp: float = Field(ge=0.0, le=2.0, default=0.7)

cfg = Config(model="gpt-4o", temp=0.5)
cfg.model_dump()                        # -> dict
cfg.model_dump_json()                   # -> JSON string
Config.model_validate({"model": "x"})   # dict -> Config
Config.model_validate_json('{"model":"x"}')  # JSON -> Config
```

### Async/Await

```python
import asyncio

# Define a coroutine
async def fetch(url: str) -> str:
    await asyncio.sleep(1)
    return f"data from {url}"

# Run from sync code
asyncio.run(fetch("https://example.com"))

# Parallel execution
async def main():
    results = await asyncio.gather(
        fetch("url1"),
        fetch("url2"),
        fetch("url3"),
        return_exceptions=True,         # don't lose all results on one error
    )

# Concurrency limit
sem = asyncio.Semaphore(5)
async def limited_fetch(url: str) -> str:
    async with sem:
        return await fetch(url)

# Async context manager
class Client:
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass

# Async generator (streaming)
async def stream() -> AsyncIterator[str]:
    for chunk in ["a", "b", "c"]:
        await asyncio.sleep(0.1)
        yield chunk

async def consume():
    async for chunk in stream():
        print(chunk)

# Async HTTP (httpx)
import httpx
async with httpx.AsyncClient() as client:
    resp = await client.get("https://api.example.com")
    resp.raise_for_status()
    data = resp.json()

# Run blocking code without blocking the event loop
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, blocking_function, arg1)
```

### Quick Reference Table

| What | Sync | Async |
|---|---|---|
| Define function | `def f():` | `async def f():` |
| Call and wait | `result = f()` | `result = await f()` |
| Sleep | `time.sleep(1)` | `await asyncio.sleep(1)` |
| Context manager | `with X() as x:` | `async with X() as x:` |
| Iterate | `for x in gen():` | `async for x in gen():` |
| Run parallel | `ThreadPoolExecutor` | `asyncio.gather()` |
| HTTP GET | `httpx.get(url)` | `await client.get(url)` |
| Entry point | `main()` | `asyncio.run(main())` |

---

**Next:** [Prerequisite 2 -- to be added]
