# Assignment 1: Production Environment & LLM Integration

**Released:** Week 1
**Due:** Week 3
**Weight:** 10%

---

## Learning Objectives

By completing this assignment, you will:
1. Set up a production-grade Python development environment
2. Build a FastAPI application with proper configuration management
3. Implement an LLM service with error handling and retries
4. Create structured request/response models with Pydantic v2
5. Implement basic rate limiting and cost tracking

---

## Overview

You will build the foundation of a production AI service: a FastAPI application that exposes a chat completion endpoint with proper error handling, configuration management, and observability hooks.

---

## Requirements

### Part 1: Project Setup (15 points)

Create a new Python project with the following structure:

```
assignment1/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py      # Pydantic Settings
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py          # Chat endpoints
│   │   └── health.py        # Health check
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py      # Request models
│   │   └── responses.py     # Response models
│   └── services/
│       ├── __init__.py
│       └── llm.py           # LLM service
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_llm_service.py
├── .env.example
├── pyproject.toml
└── README.md
```

**Deliverables:**
- [ ] Project uses `pyproject.toml` with dependencies
- [ ] `.env.example` documents all required environment variables
- [ ] `README.md` includes setup instructions

### Part 2: Configuration Management (20 points)

Implement settings management using Pydantic Settings:

```python
# src/config/settings.py

from pydantic_settings import BaseSettings
from pydantic import SecretStr, Field

class Settings(BaseSettings):
    # Application settings
    app_name: str = "Production AI Assignment 1"
    debug: bool = False

    # LLM Provider
    openai_api_key: SecretStr
    default_model: str = "gpt-4o-mini"

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Cost controls
    max_tokens_per_request: int = 4096
    daily_cost_limit: float = 10.0

    class Config:
        env_file = ".env"
```

**Deliverables:**
- [ ] Settings load from environment variables
- [ ] Secrets are properly protected with `SecretStr`
- [ ] Settings are cached using `@lru_cache`
- [ ] Invalid configuration raises clear errors

### Part 3: Request/Response Models (20 points)

Create Pydantic models for the chat API:

```python
# src/models/requests.py

from pydantic import BaseModel, Field, field_validator
from typing import Literal

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, max_length=32000)

class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=100)
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)

    @field_validator('messages')
    @classmethod
    def validate_conversation(cls, v):
        # Implement validation logic
        pass
```

**Deliverables:**
- [ ] Proper type hints and validation
- [ ] Custom validators for business rules
- [ ] Clear error messages for validation failures
- [ ] Response models include usage statistics

### Part 4: LLM Service (25 points)

Implement an LLM service with:
- Provider abstraction
- Retry logic with exponential backoff
- Error handling for common API errors
- Cost calculation

```python
# src/services/llm.py

class LLMService:
    async def generate(
        self,
        messages: list[dict],
        model: str | None = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion with:
        - Automatic retries on rate limits
        - Proper error classification
        - Token and cost tracking
        """
        pass
```

**Deliverables:**
- [ ] Implements retry with exponential backoff (3 retries)
- [ ] Handles rate limits, timeouts, and API errors distinctly
- [ ] Returns structured response with token counts
- [ ] Calculates estimated cost

### Part 5: API Endpoint (20 points)

Create a production-ready chat endpoint:

```python
# src/api/chat.py

@router.post("/completions", response_model=ChatResponse)
async def create_completion(
    request: ChatRequest,
    llm: LLMServiceDep,
    settings: SettingsDep,
) -> ChatResponse:
    """
    Create a chat completion with:
    - Request validation
    - Token limit enforcement
    - Structured error responses
    """
    pass
```

**Deliverables:**
- [ ] POST `/api/v1/chat/completions` endpoint
- [ ] GET `/health` endpoint with dependency checks
- [ ] Proper HTTP status codes for different errors
- [ ] Request ID tracking in headers

---

## Testing Requirements

Write tests covering:

```python
# tests/test_api.py

def test_chat_completion_success():
    """Test successful completion request."""
    pass

def test_chat_completion_validation_error():
    """Test validation errors return 422."""
    pass

def test_chat_completion_rate_limit():
    """Test rate limiting behavior."""
    pass

def test_health_check():
    """Test health endpoint."""
    pass
```

**Deliverables:**
- [ ] At least 5 unit tests for models
- [ ] At least 5 integration tests for API
- [ ] Tests use mocking for external APIs
- [ ] All tests pass

---

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Project Setup | 15 | Proper structure, dependencies, documentation |
| Configuration | 20 | Settings work correctly, secrets protected |
| Models | 20 | Validation works, clear error messages |
| LLM Service | 25 | Retries work, errors handled, costs tracked |
| API Endpoint | 20 | Endpoint works, proper responses |
| **Bonus** | +5 | Streaming support, additional tests |

---

## Submission

1. Push your code to a Git repository
2. Ensure all tests pass: `pytest tests/`
3. Ensure the application starts: `uvicorn src.main:app`
4. Submit the repository URL

---

## Starter Code

```python
# src/main.py
from fastapi import FastAPI
from src.api import chat, health
from src.config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
)

app.include_router(health.router)
app.include_router(chat.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```python
# src/api/health.py
from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
```

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- Course Notes: Week 1-2
