# Prerequisite 3: REST APIs

This tutorial covers everything you need to know about REST APIs before starting the **Building Production AI Systems** course. You will learn to consume APIs with `httpx` and build them with `FastAPI` -- both are used heavily throughout the course.

---

## Table of Contents

1. [What is a REST API](#1-what-is-a-rest-api)
2. [Consuming APIs with Python](#2-consuming-apis-with-python)
3. [Building APIs with FastAPI](#3-building-apis-with-fastapi)
4. [API Testing](#4-api-testing)
5. [Common Patterns for AI APIs](#5-common-patterns-for-ai-apis)
6. [Cheat Sheet](#6-cheat-sheet)

---

## 1. What is a REST API

A REST (Representational State Transfer) API is a way for two programs to communicate over HTTP. One program (the **client**) sends a **request**, and the other (the **server**) sends back a **response**. Data is almost always exchanged as JSON.

### HTTP Methods

Each request uses a **method** that signals its intent:

| Method   | Purpose                        | Example                          |
|----------|--------------------------------|----------------------------------|
| `GET`    | Read a resource                | Fetch a user's profile           |
| `POST`   | Create a resource              | Submit a new chat completion     |
| `PUT`    | Replace a resource entirely    | Update an entire user record     |
| `PATCH`  | Partially update a resource    | Change just a user's email       |
| `DELETE` | Remove a resource              | Delete a conversation            |

### Status Codes

The server responds with a numeric status code:

| Code  | Meaning                | When You See It                              |
|-------|------------------------|----------------------------------------------|
| `200` | OK                     | Successful GET, PUT, PATCH, or DELETE         |
| `201` | Created                | Successful POST that created a resource       |
| `400` | Bad Request            | Malformed JSON or missing required fields     |
| `401` | Unauthorized           | Missing or invalid authentication credentials |
| `403` | Forbidden              | Valid credentials but insufficient permissions|
| `404` | Not Found              | The resource does not exist                   |
| `422` | Unprocessable Entity   | Request is valid JSON but fails validation    |
| `500` | Internal Server Error  | Something broke on the server                 |

### Anatomy of a Request

```
POST /api/v1/chat/completions HTTP/1.1    <-- method + path
Host: api.example.com                      <-- header
Authorization: Bearer sk-abc123            <-- header
Content-Type: application/json             <-- header

{                                          <-- request body (JSON)
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Hello"}]
}
```

### Anatomy of a Response

```
HTTP/1.1 200 OK                            <-- status code
Content-Type: application/json             <-- header

{                                          <-- response body (JSON)
  "id": "chatcmpl-abc123",
  "choices": [{"message": {"role": "assistant", "content": "Hi!"}}]
}
```

---

## 2. Consuming APIs with Python

We use `httpx` throughout the course because it supports both sync and async calls, and it integrates cleanly with FastAPI's test client.

### Installation

```bash
pip install httpx
```

### Basic GET Request

```python
import httpx

response = httpx.get("https://jsonplaceholder.typicode.com/posts/1")

print(response.status_code)   # 200
print(response.json())        # {'userId': 1, 'id': 1, 'title': '...', 'body': '...'}
```

### Basic POST Request

```python
import httpx

payload = {
    "title": "My Post",
    "body": "Hello, world!",
    "userId": 1,
}

response = httpx.post(
    "https://jsonplaceholder.typicode.com/posts",
    json=payload,
)

print(response.status_code)   # 201
print(response.json())        # {'id': 101, 'title': 'My Post', ...}
```

### Setting Headers

```python
import httpx

headers = {
    "Authorization": "Bearer sk-abc123",
    "Content-Type": "application/json",
}

response = httpx.get("https://api.example.com/data", headers=headers)
```

### Authentication Patterns

**API Key in a header:**

```python
headers = {"X-API-Key": "your-api-key-here"}
response = httpx.get("https://api.example.com/data", headers=headers)
```

**Bearer token (most common for AI APIs):**

```python
headers = {"Authorization": "Bearer sk-your-token-here"}
response = httpx.post("https://api.example.com/chat", headers=headers, json=payload)
```

### Handling Responses

```python
import httpx

response = httpx.get("https://api.example.com/users/42")

# Check if the request succeeded
if response.status_code == 200:
    user = response.json()
    print(user["name"])
elif response.status_code == 404:
    print("User not found")
else:
    print(f"Unexpected status: {response.status_code}")
```

### Error Handling

```python
import httpx

try:
    response = httpx.get("https://api.example.com/data", timeout=10.0)
    response.raise_for_status()  # Raises an exception for 4xx/5xx responses
    data = response.json()
except httpx.TimeoutException:
    print("Request timed out")
except httpx.HTTPStatusError as e:
    print(f"HTTP error {e.response.status_code}: {e.response.text}")
except httpx.RequestError as e:
    print(f"Network error: {e}")
```

### Practical Example: Calling an OpenAI-Compatible API

This pattern appears constantly in the course. Many AI services expose an OpenAI-compatible chat completions endpoint.

```python
import httpx

API_BASE = "https://api.openai.com/v1"
API_KEY = "sk-your-key-here"

def chat(messages: list[dict], model: str = "gpt-4") -> str:
    """Send a chat completion request and return the assistant's reply."""
    response = httpx.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.7,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# Usage
reply = chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is FastAPI?"},
])
print(reply)
```

### Async Requests with httpx

When you need to make multiple API calls concurrently (common in AI pipelines):

```python
import asyncio
import httpx

async def fetch_all():
    async with httpx.AsyncClient() as client:
        urls = [
            "https://api.example.com/model-a/predict",
            "https://api.example.com/model-b/predict",
            "https://api.example.com/model-c/predict",
        ]
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)

        for resp in responses:
            print(resp.status_code, resp.json())

asyncio.run(fetch_all())
```

---

## 3. Building APIs with FastAPI

FastAPI is the framework we use to build AI service endpoints throughout the course. It is fast, has automatic docs, and uses Python type hints for validation.

### Installation

```bash
pip install "fastapi[standard]"
```

This installs FastAPI along with `uvicorn` (the ASGI server) and other useful defaults.

### Your First App

Create a file called `main.py`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, world!"}
```

Run it:

```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000` to see the response. Visit `http://127.0.0.1:8000/docs` for the auto-generated Swagger UI.

### Path Parameters

```python
@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}
```

FastAPI automatically validates that `user_id` is an integer. Sending `/users/abc` returns a `422` error.

### Query Parameters

```python
@app.get("/search")
def search(q: str, limit: int = 10):
    return {"query": q, "limit": limit}
```

Call it: `GET /search?q=fastapi&limit=5`

### Request Body with Pydantic Models

Pydantic models define the shape of your request and response data. FastAPI validates incoming JSON against these models automatically.

```python
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: float = 0.7

@app.post("/chat")
def chat(request: ChatRequest):
    return {
        "model": request.model,
        "message_count": len(request.messages),
        "temperature": request.temperature,
    }
```

Sending invalid JSON (wrong types, missing required fields) automatically returns a `422` with a detailed error message.

### Response Models

Use `response_model` to control what gets returned and documented:

```python
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

class UserOut(BaseModel):
    id: int
    name: str
    email: str
    # Note: password is NOT here, so it's never exposed

class UserIn(BaseModel):
    name: str
    email: str
    password: str

@app.post("/users", response_model=UserOut, status_code=201)
def create_user(user: UserIn):
    # In real code, save to database here
    return {"id": 1, "name": user.name, "email": user.email, "password": user.password}
    # password is automatically stripped from the response because of response_model
```

### Status Codes

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

items = {"wrench": {"name": "Wrench", "price": 9.99}}

@app.get("/items/{item_id}")
def get_item(item_id: str):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]

@app.post("/items", status_code=201)
def create_item(item_id: str, name: str, price: float):
    items[item_id] = {"name": name, "price": price}
    return items[item_id]
```

### Dependency Injection

Dependencies let you extract shared logic (authentication, database connections, configuration) into reusable functions.

```python
from fastapi import FastAPI, Depends, HTTPException, Header

app = FastAPI()

def verify_api_key(x_api_key: str = Header(...)):
    """Dependency that checks for a valid API key."""
    if x_api_key != "secret-key-123":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.get("/protected")
def protected_route(api_key: str = Depends(verify_api_key)):
    return {"message": "You have access!", "key_used": api_key}
```

You can also apply dependencies to the entire app or a router:

```python
from fastapi import APIRouter

router = APIRouter(
    prefix="/api/v1",
    dependencies=[Depends(verify_api_key)],  # All routes in this router require the key
)

@router.get("/models")
def list_models():
    return {"models": ["gpt-4", "claude-3"]}

app.include_router(router)
```

### Middleware

Middleware runs code before and after every request. Useful for logging, timing, and adding headers.

```python
import time
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Response-Time"] = f"{elapsed:.4f}s"
    return response
```

### Practical Example: A Simple AI Service Endpoint

Putting it all together -- a minimal AI inference service:

```python
import time
from pydantic import BaseModel, Field
from fastapi import FastAPI, Depends, HTTPException, Header

app = FastAPI(title="AI Inference Service", version="0.1.0")

# --- Models ---

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    model_name: str = "sentiment-v1"

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model_name: str
    latency_ms: float

# --- Dependencies ---

VALID_API_KEYS = {"key-abc123", "key-def456"}

def verify_api_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.removeprefix("Bearer ")
    if token not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token

# --- Routes ---

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, api_key: str = Depends(verify_api_key)):
    start = time.perf_counter()

    # Simulate model inference
    label = "positive" if "good" in request.text.lower() else "negative"
    confidence = 0.95

    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        label=label,
        confidence=confidence,
        model_name=request.model_name,
        latency_ms=round(latency_ms, 2),
    )
```

Run it and test:

```bash
# Start the server
uvicorn main:app --reload

# Test the health endpoint
curl http://127.0.0.1:8000/health

# Test the predict endpoint
curl -X POST http://127.0.0.1:8000/predict \
  -H "Authorization: Bearer key-abc123" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is really good!", "model_name": "sentiment-v1"}'
```

---

## 4. API Testing

The course includes extensive API testing. FastAPI provides a built-in test client powered by `httpx`.

### Installation

```bash
pip install pytest httpx
```

### Basic Test Structure

Given the app from the previous section saved as `app.py`:

```python
# test_app.py
import pytest
from httpx import AsyncClient, ASGITransport
from app import app

@pytest.mark.anyio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.anyio
async def test_predict_success():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/predict",
            headers={"Authorization": "Bearer key-abc123"},
            json={"text": "This is good", "model_name": "sentiment-v1"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "positive"
    assert data["model_name"] == "sentiment-v1"
    assert "latency_ms" in data

@pytest.mark.anyio
async def test_predict_unauthorized():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/predict",
            headers={"Authorization": "Bearer wrong-key"},
            json={"text": "test"},
        )

    assert response.status_code == 401

@pytest.mark.anyio
async def test_predict_validation_error():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/predict",
            headers={"Authorization": "Bearer key-abc123"},
            json={},  # missing required "text" field
        )

    assert response.status_code == 422
```

### Using a Shared Fixture

Avoid repeating the client setup with a pytest fixture:

```python
# conftest.py
import pytest
from httpx import AsyncClient, ASGITransport
from app import app

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
```

```python
# test_app.py
import pytest

@pytest.mark.anyio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200

@pytest.mark.anyio
async def test_predict(client):
    response = await client.post(
        "/predict",
        headers={"Authorization": "Bearer key-abc123"},
        json={"text": "good stuff"},
    )
    assert response.status_code == 200
    assert response.json()["label"] == "positive"
```

Run the tests:

```bash
# Install the anyio pytest plugin
pip install anyio pytest-anyio

# Run
pytest test_app.py -v
```

---

## 5. Common Patterns for AI APIs

These patterns come up repeatedly when building production AI services.

### Health Check Endpoint

Every service needs a health check. Load balancers and orchestrators (Kubernetes, etc.) use it to know if your service is alive.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="1.2.0",
    )
```

### Versioned APIs

Prefix your routes so you can evolve the API without breaking existing clients:

```python
from fastapi import FastAPI, APIRouter

app = FastAPI()

v1_router = APIRouter(prefix="/api/v1")
v2_router = APIRouter(prefix="/api/v2")

@v1_router.post("/chat")
def chat_v1(message: str):
    return {"response": f"v1 echo: {message}"}

@v2_router.post("/chat")
def chat_v2(message: str):
    return {"response": f"v2 echo: {message}", "version": 2}

app.include_router(v1_router)
app.include_router(v2_router)
```

### Streaming Responses (Server-Sent Events)

LLM APIs stream tokens back to the client as they are generated. FastAPI supports this with `StreamingResponse`.

```python
import asyncio
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

async def generate_tokens(prompt: str):
    """Simulate an LLM generating tokens one at a time."""
    tokens = ["Hello", " there", "!", " How", " can", " I", " help", " you", "?"]
    for token in tokens:
        chunk = {"choices": [{"delta": {"content": token}}]}
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.1)  # Simulate generation delay
    yield "data: [DONE]\n\n"

@app.post("/chat/stream")
async def chat_stream(prompt: str = "hello"):
    return StreamingResponse(
        generate_tokens(prompt),
        media_type="text/event-stream",
    )
```

**Consuming an SSE stream from the client side:**

```python
import httpx

with httpx.stream("POST", "http://127.0.0.1:8000/chat/stream?prompt=hello") as response:
    for line in response.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            import json
            chunk = json.loads(line.removeprefix("data: "))
            token = chunk["choices"][0]["delta"]["content"]
            print(token, end="", flush=True)
    print()  # newline at the end
```

### CORS (Cross-Origin Resource Sharing)

If a frontend (running on `localhost:3000`) calls your API (running on `localhost:8000`), browsers block it by default. CORS middleware fixes this.

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/data")
def get_data():
    return {"value": 42}
```

### Background Tasks

Offload work that the client does not need to wait for (logging, sending emails, updating caches):

```python
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

def log_prediction(request_id: str, result: str):
    # Simulate writing to a log or database
    with open("predictions.log", "a") as f:
        f.write(f"{request_id}: {result}\n")

@app.post("/predict")
def predict(text: str, background_tasks: BackgroundTasks):
    result = "positive"  # Simulate inference
    background_tasks.add_task(log_prediction, "req-001", result)
    return {"label": result}
```

---

## 6. Cheat Sheet

### HTTP Methods

| Method   | CRUD Operation | Idempotent | Has Body |
|----------|---------------|------------|----------|
| `GET`    | Read          | Yes        | No       |
| `POST`   | Create        | No         | Yes      |
| `PUT`    | Replace       | Yes        | Yes      |
| `PATCH`  | Update        | No         | Yes      |
| `DELETE` | Delete        | Yes        | No       |

### Status Codes

| Range | Category      | Common Codes                                    |
|-------|---------------|-------------------------------------------------|
| 2xx   | Success       | `200 OK`, `201 Created`, `204 No Content`       |
| 4xx   | Client Error  | `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `404 Not Found`, `422 Unprocessable Entity`, `429 Too Many Requests` |
| 5xx   | Server Error  | `500 Internal Server Error`, `503 Service Unavailable` |

### httpx Quick Reference

```python
import httpx

# GET
resp = httpx.get(url, headers=headers, params={"key": "value"}, timeout=10.0)

# POST with JSON
resp = httpx.post(url, headers=headers, json={"key": "value"}, timeout=10.0)

# Check response
resp.status_code        # 200
resp.json()             # parsed JSON
resp.text               # raw text
resp.headers            # response headers
resp.raise_for_status() # raise on 4xx/5xx

# Async
async with httpx.AsyncClient() as client:
    resp = await client.get(url)
```

### FastAPI Quick Reference

```python
from fastapi import FastAPI, Depends, HTTPException, Header, Query, Path
from pydantic import BaseModel

app = FastAPI()

# Route decorators
@app.get("/path")
@app.post("/path", status_code=201)
@app.put("/path")
@app.patch("/path")
@app.delete("/path", status_code=204)

# Parameters
@app.get("/items/{item_id}")
def read(item_id: int = Path(..., ge=1)):              # path param
    ...

@app.get("/search")
def search(q: str = Query(..., min_length=1)):          # query param
    ...

# Request body
class Body(BaseModel):
    field: str

@app.post("/endpoint")
def create(body: Body):                                  # JSON body
    ...

# Response model
@app.get("/endpoint", response_model=SomeModel)
def read():
    ...

# Dependency injection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/items")
def read(db: Session = Depends(get_db)):
    ...

# Error responses
raise HTTPException(status_code=404, detail="Not found")

# Include router
from fastapi import APIRouter
router = APIRouter(prefix="/api/v1")
app.include_router(router)
```

### Common Patterns

```python
# Health check
@app.get("/health")
def health():
    return {"status": "healthy"}

# CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Streaming response (SSE)
from fastapi.responses import StreamingResponse
@app.post("/stream")
async def stream():
    return StreamingResponse(token_generator(), media_type="text/event-stream")

# Background tasks
from fastapi import BackgroundTasks
@app.post("/task")
def run(bg: BackgroundTasks):
    bg.add_task(some_function, arg1, arg2)
    return {"status": "accepted"}
```

### Testing Pattern

```python
# conftest.py
import pytest
from httpx import AsyncClient, ASGITransport
from app import app

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

# test_app.py
@pytest.mark.anyio
async def test_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
```

---

## Next Steps

You are ready for the course if you can:

- Make GET and POST requests with `httpx` and handle errors
- Build a FastAPI app with path/query parameters and Pydantic models
- Add authentication via dependency injection
- Write async tests using `httpx.AsyncClient`
- Understand streaming responses and CORS

If any of these feel unfamiliar, build a small project: create a FastAPI service with two or three endpoints, add authentication, write tests for each endpoint, and call it from a separate Python script using `httpx`.
