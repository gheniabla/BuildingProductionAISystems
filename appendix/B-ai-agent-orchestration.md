# Appendix B: AI Agent Orchestration

A comprehensive technical reference supporting the Building Production AI Systems course.

---

## Table of Contents

1. [RAG in Agentic Context](#1-rag-in-agentic-context)
2. [MCP (Model Context Protocol)](#2-mcp-model-context-protocol)
3. [DSPy](#3-dspy)
4. [Chain-of-Thought (CoT)](#4-chain-of-thought)
5. [ReAct (Reasoning + Acting)](#5-react)
6. [Context Engineering](#6-context-engineering)
7. [Frameworks: LangGraph and PydanticAI](#7-frameworks-langgraph-and-pydanticai)

---

## 1. RAG in Agentic Context

**Course Connection:** Week 5 (RAG Systems), Week 4 (Security — secure retrieval)

### 1.1 From Naive RAG to Agentic RAG

**Naive RAG** follows a fixed pipeline: retrieve → augment prompt → generate. The process is one-shot — there is no feedback loop.

**Agentic RAG** treats retrieval as a tool the agent can invoke conditionally, iteratively, and with self-reflection:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#4F46E5', 'primaryTextColor': '#FFFFFF', 'primaryBorderColor': '#3730A3', 'secondaryColor': '#D1FAE5', 'secondaryTextColor': '#065F46', 'secondaryBorderColor': '#059669', 'tertiaryColor': '#FEF3C7', 'tertiaryTextColor': '#92400E', 'tertiaryBorderColor': '#D97706', 'lineColor': '#6B7280', 'textColor': '#1F2937', 'fontSize': '14px'}}}%%
flowchart TD
    subgraph naive["Naive RAG"]
        direction LR
        NQ["Query"] --> NR["Retrieve"] --> NG["Generate"] --> ND["Done"]
    end

    subgraph agentic["Agentic RAG"]
        AQ["Query"] --> AT{"Think: Do I need to retrieve?"}
        AT -- "No" --> AG["Generate"]
        AT -- "Yes" --> AR["Retrieve"]
        AR --> AE{"Evaluate: Are results sufficient?"}
        AE -- "Yes" --> AG
        AE -- "No" --> AF["Reformulate query"] --> AR
        AG --> SC{"Self-check: Is answer grounded in sources?"}
        SC -- "Yes" --> AD["Done"]
        SC -- "No" --> AT
    end

    classDef client fill:#3B82F6,stroke:#1D4ED8,color:#FFFFFF
    classDef gateway fill:#EF4444,stroke:#B91C1C,color:#FFFFFF
    classDef service fill:#4F46E5,stroke:#3730A3,color:#FFFFFF
    classDef data fill:#10B981,stroke:#047857,color:#FFFFFF
    classDef external fill:#F59E0B,stroke:#D97706,color:#FFFFFF
    classDef observability fill:#8B5CF6,stroke:#6D28D9,color:#FFFFFF

    class NQ,AQ client
    class NR,AR service
    class NG,AG observability
    class ND,AD data
    class AT,AE,SC gateway
    class AF external
```

### 1.2 Adaptive Retrieval

Not all queries need retrieval. An agent can decide whether to retrieve based on the query type:

```python
async def should_retrieve(query: str, agent_state: dict) -> bool:
    """Decide whether retrieval is needed."""
    # Simple heuristic approach
    if is_factual_question(query):
        return True
    if is_conversational(query):
        return False
    if has_sufficient_context(agent_state):
        return False

    # LLM-based decision
    decision = await llm.classify(
        f"Does this query require external knowledge? Query: {query}"
    )
    return decision == "yes"
```

This avoids unnecessary retrieval latency and cost for queries the model can handle from its training data.

### 1.3 Self-Reflective RAG

**CRAG (Corrective RAG)** (Yan et al., 2024): After retrieval, evaluates the relevance of retrieved documents and takes corrective action:

```
Query --> Retrieve --> Evaluate relevance of each document
  |
  +--> If CORRECT: Use retrieved documents
  +--> If AMBIGUOUS: Refine query and retrieve again
  +--> If INCORRECT: Fall back to web search or generate without retrieval
```

**Self-RAG** (Asai et al., 2023): The model generates special tokens that indicate:
- Whether retrieval is needed (`[Retrieve]` / `[No Retrieve]`)
- Whether a retrieved passage is relevant (`[Relevant]` / `[Irrelevant]`)
- Whether the generated response is supported (`[Supported]` / `[Not Supported]`)

### 1.4 Multi-Step Retrieval

Complex queries may require breaking down into sub-queries:

```python
async def multi_step_retrieve(complex_query: str) -> list[Document]:
    """Decompose a complex query and retrieve iteratively."""
    # Step 1: Decompose
    sub_queries = await llm.decompose(complex_query)
    # e.g., "Compare Q4 revenue for Apple and Google"
    # --> ["Apple Q4 revenue 2024", "Google Q4 revenue 2024"]

    # Step 2: Retrieve for each sub-query
    all_docs = []
    for sub_q in sub_queries:
        docs = await retriever.search(sub_q, top_k=3)
        all_docs.extend(docs)

    # Step 3: Deduplicate and rerank against original query
    unique_docs = deduplicate(all_docs)
    reranked = await reranker.rerank(complex_query, unique_docs)

    return reranked[:5]
```

### 1.5 Security Considerations

When RAG operates within an agent that has tool access, security becomes critical:

- **Retrieved documents may contain prompt injection** (Week 4). An attacker could inject instructions into a document that, when retrieved, causes the agent to take unintended actions
- **Data isolation**: Ensure retrieval respects user permissions — user A should not retrieve user B's documents
- **Tool escalation**: A compromised retrieval result should not be able to trigger tool execution

---

## 2. MCP (Model Context Protocol)

**Course Connection:** Week 2 (Architecture — API design), Week 4 (Security)

### 2.1 What Problem MCP Solves

Before MCP, every AI application had to build custom integrations for each data source and tool. An application using 5 tools needed 5 custom integrations. If you switched LLM providers, those integrations might need rewriting.

MCP standardizes the interface between LLM applications and external capabilities:

```
Before MCP:                          With MCP:
App <-> Custom GitHub integration    App <-> MCP Client <-> MCP Server (GitHub)
App <-> Custom Slack integration                        <-> MCP Server (Slack)
App <-> Custom DB integration                           <-> MCP Server (Database)
(N×M integrations)                   (Standard protocol, plug-and-play)
```

### 2.2 Architecture

MCP follows a client-server architecture:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#4F46E5', 'primaryTextColor': '#FFFFFF', 'primaryBorderColor': '#3730A3', 'secondaryColor': '#D1FAE5', 'secondaryTextColor': '#065F46', 'secondaryBorderColor': '#059669', 'tertiaryColor': '#FEF3C7', 'tertiaryTextColor': '#92400E', 'tertiaryBorderColor': '#D97706', 'lineColor': '#6B7280', 'textColor': '#1F2937', 'fontSize': '14px'}}}%%
flowchart TD
    subgraph Host["Host Application #40;Claude Desktop, IDE, Custom App#41;"]
        C1["MCP Client"]
        C2["MCP Client"]
        C3["MCP Client"]
    end

    C1 --> S1["MCP Server #40;GitHub#41;"]
    C2 --> S2["MCP Server #40;Database#41;"]
    C3 --> S3["MCP Server #40;Slack#41;"]

    classDef client fill:#3B82F6,stroke:#1D4ED8,color:#FFFFFF
    classDef gateway fill:#EF4444,stroke:#B91C1C,color:#FFFFFF
    classDef service fill:#4F46E5,stroke:#3730A3,color:#FFFFFF
    classDef data fill:#10B981,stroke:#047857,color:#FFFFFF
    classDef external fill:#F59E0B,stroke:#D97706,color:#FFFFFF
    classDef observability fill:#8B5CF6,stroke:#6D28D9,color:#FFFFFF

    class C1,C2,C3 service
    class S1 external
    class S2 data
    class S3 external
```

**Host**: The application containing the LLM (e.g., Claude Desktop, a custom FastAPI app)
**Client**: Maintains a 1:1 connection with a server, handles protocol negotiation
**Server**: Exposes capabilities (tools, resources, prompts) through the standard protocol

### 2.3 Core Primitives

MCP defines four main primitives:

**Tools** — Functions the LLM can invoke:
```json
{
  "name": "search_codebase",
  "description": "Search the codebase for files matching a pattern",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query"},
      "file_pattern": {"type": "string", "description": "Glob pattern"}
    },
    "required": ["query"]
  }
}
```

**Resources** — Data the LLM can read (files, database records, API responses):
```json
{
  "uri": "file:///project/src/main.py",
  "name": "Main application file",
  "mimeType": "text/x-python"
}
```

**Prompts** — Reusable prompt templates:
```json
{
  "name": "code_review",
  "description": "Review code for bugs and improvements",
  "arguments": [
    {"name": "code", "description": "The code to review", "required": true}
  ]
}
```

**Sampling** — Allows servers to request LLM completions through the client (enabling agentic server behaviors).

### 2.4 Transport Mechanisms

**stdio** (Standard I/O): Server runs as a subprocess. Communication via stdin/stdout. Best for local tools.

```python
# Starting an MCP server via stdio
import subprocess
proc = subprocess.Popen(
    ["python", "mcp_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)
```

**HTTP + SSE** (Server-Sent Events): Server runs as a web service. Client connects via HTTP. Best for remote/shared servers.

```
Client --> POST /message --> Server (request)
Client <-- SSE stream   <-- Server (response stream)
```

### 2.5 Building an MCP Server

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("my-server")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        weather = await fetch_weather(arguments["city"])
        return [TextContent(type="text", text=f"Weather: {weather}")]

# Run with stdio transport
from mcp.server.stdio import stdio_server
async with stdio_server() as (read, write):
    await server.run(read, write)
```

### 2.6 Production Considerations

**Security:**
- Validate all tool inputs on the server side
- Implement authentication for remote MCP servers
- Apply least-privilege: servers should only expose necessary capabilities
- Rate limit tool calls to prevent abuse

**Tool discovery**: In production, dynamically loading MCP servers allows extending agent capabilities without redeployment.

**Error handling**: MCP servers should return structured errors that help the LLM understand failures and retry appropriately.

---

## 3. DSPy

**Course Connection:** Week 3 (Evaluation — systematic optimization), Week 7 (Optimization)

### 3.1 Programming, Not Prompting

DSPy (Khattab et al., 2023) takes a fundamentally different approach to building LLM applications. Instead of manually writing prompts, you:

1. **Declare** what the LLM should do (Signatures)
2. **Compose** modules that implement patterns (ChainOfThought, ReAct)
3. **Optimize** prompts automatically using training data (Teleprompters/Optimizers)

The key insight: prompts are compiled artifacts, not hand-crafted instructions. DSPy compiles high-level declarations into optimized prompts.

### 3.2 Signatures

A Signature declares the input/output behavior of an LLM call:

```python
import dspy

# Simple signature using shorthand notation
classify = dspy.Predict("sentence -> sentiment: str")

# Detailed signature using a class
class Summarize(dspy.Signature):
    """Summarize the given document in 2-3 sentences."""
    document: str = dspy.InputField(desc="The document to summarize")
    summary: str = dspy.OutputField(desc="A concise summary")

# Usage
summarizer = dspy.Predict(Summarize)
result = summarizer(document="Long article text here...")
print(result.summary)
```

### 3.3 Modules

Modules wrap signatures with specific prompting strategies:

```python
# Chain of Thought — adds reasoning before the answer
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="What is 15% of 240?")
# result.reasoning = "15% of 240 = 0.15 × 240 = 36"
# result.answer = "36"

# ReAct — interleaves reasoning with tool use
class SearchAndAnswer(dspy.Module):
    def __init__(self):
        self.react = dspy.ReAct(
            "question -> answer",
            tools=[dspy.Tool(search_wikipedia)]
        )

    def forward(self, question):
        return self.react(question=question)
```

### 3.4 Optimizers (Teleprompters)

Optimizers automatically tune prompts using training data:

```python
# Define a metric
def answer_accuracy(example, prediction, trace=None):
    return prediction.answer.lower() == example.answer.lower()

# Create training data
trainset = [
    dspy.Example(question="Capital of France?", answer="Paris"),
    dspy.Example(question="Capital of Japan?", answer="Tokyo"),
    # ... more examples
]

# Optimize
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=answer_accuracy,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)

optimized_module = optimizer.compile(
    student=dspy.ChainOfThought("question -> answer"),
    trainset=trainset
)
```

**Key optimizers:**
- **BootstrapFewShot**: Generates and selects effective few-shot examples
- **MIPRO**: Optimizes both instructions and few-shot examples using Bayesian optimization
- **BootstrapFinetune**: Compiles the optimized program into fine-tuning data

### 3.5 Why DSPy Matters for Production

**Reproducibility**: Prompts are generated by an optimization process, not hand-crafted. The process is repeatable.

**Systematic optimization**: Instead of A/B testing prompt variants manually, DSPy searches the space systematically using metrics.

**Reduced fragility**: When you change models (e.g., GPT-4 to Claude), re-running the optimizer adapts the prompts automatically rather than requiring manual rewriting.

**Evaluation integration (Week 3)**: DSPy's optimizers require metrics — this naturally enforces building evaluation systems.

```python
# Compile for a different model
dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-4-20250514"))
reoptimized = optimizer.compile(
    student=dspy.ChainOfThought("question -> answer"),
    trainset=trainset
)
# Prompts are automatically adapted for the new model
```

---

## 4. Chain-of-Thought (CoT)

**Course Connection:** Week 3 (Evaluation — evaluating reasoning), Week 7 (Optimization — cost/latency trade-offs)

### 4.1 The Core Technique

Chain-of-Thought prompting (Wei et al., 2022) instructs the model to produce intermediate reasoning steps before giving the final answer.

**Without CoT:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3. How many does he have?
A: 11
```

**With CoT:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3. How many does he have?
A: Roger started with 5 balls. He bought 2 cans of 3 balls each, which is
   2 × 3 = 6 balls. So he now has 5 + 6 = 11 tennis balls.
```

The explicit reasoning steps improve accuracy on math, logic, and multi-step problems.

### 4.2 Zero-Shot CoT

Simply appending "Let's think step by step" triggers CoT reasoning without any examples:

```python
prompt = f"""{question}

Let's think step by step."""
```

This works because RLHF-trained models have seen reasoning patterns during training.

### 4.3 Few-Shot CoT

Providing examples with reasoning chains as demonstrations:

```python
prompt = """Q: There are 15 trees in the grove. 5 more trees are planted. How many now?
A: There were originally 15 trees. Then 5 more were planted. 15 + 5 = 20.
The answer is 20.

Q: If there are 3 cars and each has 4 passengers, how many passengers total?
A: There are 3 cars with 4 passengers each. 3 × 4 = 12 passengers total.
The answer is 12.

Q: {user_question}
A: Let's think step by step."""
```

### 4.4 Self-Consistency

Self-Consistency (Wang et al., 2022) samples multiple reasoning paths and takes the majority vote:

```python
async def self_consistent_answer(question: str, n_samples: int = 5) -> str:
    """Sample multiple CoT reasoning paths and vote."""
    answers = []
    for _ in range(n_samples):
        response = await llm.generate(
            prompt=f"{question}\nLet's think step by step.",
            temperature=0.7  # Higher temperature for diverse reasoning
        )
        answer = extract_final_answer(response)
        answers.append(answer)

    # Majority vote
    return most_common(answers)
```

This trades increased cost (N API calls) for significantly improved accuracy.

### 4.5 Extensions

**Tree-of-Thought (Yao et al., 2023)**: Explores multiple reasoning branches in parallel, evaluates each branch, and prunes unpromising ones. Uses BFS or DFS search over thought trees.

**Graph-of-Thought**: Extends Tree-of-Thought to allow merging and combining insights from different reasoning branches.

### 4.6 Structured CoT for Production

In production, structure the reasoning trace for parseability:

```python
system_prompt = """When answering questions, use the following format:

<reasoning>
Step 1: [First reasoning step]
Step 2: [Second reasoning step]
...
</reasoning>

<answer>
[Final answer]
</answer>"""
```

This allows extracting both the reasoning trace (for logging/evaluation) and the answer (for the user).

### 4.7 Cost and Latency Implications

CoT generates significantly more tokens:
- **Without CoT**: ~10-50 output tokens
- **With CoT**: ~100-500 output tokens
- **Self-Consistency (5 samples)**: 5x the output tokens

**Production strategy**: Use CoT selectively:
- Simple lookups and greetings: No CoT
- Multi-step reasoning: CoT
- High-stakes decisions: Self-Consistency

**Evaluation connection (Week 3)**: Log reasoning traces to evaluate not just the final answer but the quality of reasoning. Incorrect reasoning that accidentally produces a correct answer is a reliability risk.

---

## 5. ReAct (Reasoning + Acting)

**Course Connection:** Week 2 (Async patterns), Week 8 (Observability — tracing agent loops)

### 5.1 The ReAct Paradigm

ReAct (Yao et al., 2022) interleaves reasoning traces with concrete actions:

```
Thought: I need to find the population of Paris to answer this question.
Action: search("population of Paris 2024")
Observation: Paris has a population of approximately 2.1 million in the city
             proper and 12.3 million in the metro area.
Thought: The question asks about the city proper, so the answer is 2.1 million.
Action: finish("2.1 million")
```

The key insight: reasoning without acting produces plans that may be wrong; acting without reasoning produces unfocused trial-and-error. Combining them produces grounded, directed problem-solving.

### 5.2 The Thought → Action → Observation Loop

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#4F46E5', 'primaryTextColor': '#FFFFFF', 'primaryBorderColor': '#3730A3', 'secondaryColor': '#D1FAE5', 'secondaryTextColor': '#065F46', 'secondaryBorderColor': '#059669', 'tertiaryColor': '#FEF3C7', 'tertiaryTextColor': '#92400E', 'tertiaryBorderColor': '#D97706', 'lineColor': '#6B7280', 'textColor': '#1F2937', 'fontSize': '14px'}}}%%
flowchart LR
    T["Thought #40;Reason#41;"] --> A["Action #40;Tool Use#41;"]
    A --> O["Observation #40;Tool Result#41;"]
    O --> D{"Action = finish?"}
    D -- "No" --> T
    D -- "Yes" --> F["Return answer"]

    classDef client fill:#3B82F6,stroke:#1D4ED8,color:#FFFFFF
    classDef gateway fill:#EF4444,stroke:#B91C1C,color:#FFFFFF
    classDef service fill:#4F46E5,stroke:#3730A3,color:#FFFFFF
    classDef data fill:#10B981,stroke:#047857,color:#FFFFFF
    classDef external fill:#F59E0B,stroke:#D97706,color:#FFFFFF
    classDef observability fill:#8B5CF6,stroke:#6D28D9,color:#FFFFFF

    class T observability
    class A service
    class O external
    class D gateway
    class F data
```

### 5.3 Implementation Pattern

```python
import json
from typing import Callable

REACT_PROMPT = """You are an assistant that solves problems step by step.

Available tools:
{tool_descriptions}

For each step, output exactly one of:
Thought: <your reasoning>
Action: <tool_name>(<arguments as JSON>)

When you have the final answer:
Action: finish(<your answer>)

Begin!

Question: {question}
"""

async def react_loop(
    question: str,
    tools: dict[str, Callable],
    max_steps: int = 10
) -> str:
    """Execute a ReAct loop."""
    messages = [{"role": "user", "content": format_prompt(question, tools)}]

    for step in range(max_steps):
        # Get model's thought + action
        response = await llm.generate(messages)
        messages.append({"role": "assistant", "content": response})

        # Parse the action
        action = parse_action(response)

        if action.name == "finish":
            return action.arguments

        # Execute the tool
        try:
            observation = await tools[action.name](**action.arguments)
        except Exception as e:
            observation = f"Error: {str(e)}"

        # Feed observation back
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}"
        })

    return "Max steps reached without conclusion"
```

### 5.4 Comparison with CoT-Only and Act-Only

| Approach | Reasoning | Actions | When to Use |
|----------|-----------|---------|-------------|
| CoT-only | Yes | No | Pure reasoning tasks (math, logic) |
| Act-only | No | Yes | Simple tool use, no planning needed |
| ReAct | Yes | Yes | Complex tasks needing both reasoning and information gathering |

### 5.5 Error Recovery and Retry

Production ReAct agents need robust error handling:

```python
async def react_with_recovery(question: str, tools: dict, max_retries: int = 2):
    """ReAct loop with error recovery."""
    for step in range(max_steps):
        response = await llm.generate(messages)
        action = parse_action(response)

        if action.name == "finish":
            return action.arguments

        retry_count = 0
        while retry_count <= max_retries:
            try:
                observation = await asyncio.wait_for(
                    tools[action.name](**action.arguments),
                    timeout=30.0  # 30s timeout per tool call
                )
                break
            except asyncio.TimeoutError:
                observation = f"Tool '{action.name}' timed out after 30s"
                retry_count += 1
            except KeyError:
                observation = f"Unknown tool: '{action.name}'. Available: {list(tools.keys())}"
                break
            except Exception as e:
                observation = f"Tool error: {str(e)}"
                retry_count += 1

        messages.append({"role": "user", "content": f"Observation: {observation}"})
```

### 5.6 Production Considerations

**Timeout handling**: Each tool call should have a timeout. The overall ReAct loop should also have a total time budget.

**Cost management**: Each loop iteration requires an LLM call. Set `max_steps` to prevent runaway costs. Monitor average steps per query.

**Observability (Week 8)**: Trace each Thought/Action/Observation step. This is critical for debugging:
```python
# OpenTelemetry tracing for ReAct
with tracer.start_span("react_loop") as span:
    span.set_attribute("question", question)
    for step in range(max_steps):
        with tracer.start_span(f"step_{step}") as step_span:
            step_span.set_attribute("thought", thought)
            step_span.set_attribute("action", action.name)
            step_span.set_attribute("observation_length", len(observation))
```

**Async patterns (Week 2)**: Tool calls are often I/O-bound (API calls, database queries). Use `asyncio` for concurrent tool execution when the agent dispatches multiple independent actions.

---

## 6. Context Engineering

**Course Connection:** Week 5 (RAG — context assembly), Week 7 (Optimization — token costs, caching)

### 6.1 What Is Context Engineering

Context engineering is the discipline of optimizing what goes into the LLM's context window. As systems grow beyond simple chat, the context window becomes a critical resource that must be managed programmatically.

### 6.2 Components of Context

A production LLM request may assemble context from many sources:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#4F46E5', 'primaryTextColor': '#FFFFFF', 'primaryBorderColor': '#3730A3', 'secondaryColor': '#D1FAE5', 'secondaryTextColor': '#065F46', 'secondaryBorderColor': '#059669', 'tertiaryColor': '#FEF3C7', 'tertiaryTextColor': '#92400E', 'tertiaryBorderColor': '#D97706', 'lineColor': '#6B7280', 'textColor': '#1F2937', 'fontSize': '14px'}}}%%
flowchart TD
    subgraph window ["Context Window"]
        SP["System Prompt — Static"]:::service
        TD["Tool Definitions — Semi-static"]:::service
        FS["Few-Shot Examples — Semi-static"]:::service
        RD["Retrieved Documents #40;RAG#41; — Dynamic"]:::client
        MU["Memory / User Profile — Session-dependent"]:::observability
        CH["Conversation History — Dynamic, growing"]:::client
        TR["Tool Results — Dynamic"]:::client
        CM["Current User Message — Dynamic"]:::client

        SP --> TD --> FS --> RD --> MU --> CH --> TR --> CM
    end

    CM --> RO["Reserved for Output — Must leave room"]:::gateway

    classDef client fill:#3B82F6,stroke:#1D4ED8,color:#FFFFFF
    classDef gateway fill:#EF4444,stroke:#B91C1C,color:#FFFFFF
    classDef service fill:#4F46E5,stroke:#3730A3,color:#FFFFFF
    classDef data fill:#10B981,stroke:#047857,color:#FFFFFF
    classDef external fill:#F59E0B,stroke:#D97706,color:#FFFFFF
    classDef observability fill:#8B5CF6,stroke:#6D28D9,color:#FFFFFF
```

### 6.3 Context Window Management Strategies

**Sliding window**: Keep only the most recent N messages of conversation history. Simple but loses earlier context.

**Summarization**: Periodically summarize older conversation turns:
```python
async def manage_history(messages: list, max_tokens: int) -> list:
    """Keep recent messages, summarize older ones."""
    if count_tokens(messages) <= max_tokens:
        return messages

    # Keep system prompt and last N messages
    system = messages[0]
    recent = messages[-6:]  # Last 3 turns
    old = messages[1:-6]

    # Summarize old messages
    summary = await llm.summarize(old)
    summary_msg = {"role": "system", "content": f"Previous conversation summary: {summary}"}

    return [system, summary_msg] + recent
```

**Priority-based truncation**: Assign priorities to context components and drop low-priority items first when the budget is exceeded:

```python
def assemble_context(blocks: list[ContextBlock], budget: int) -> list[ContextBlock]:
    """Assemble context within token budget, respecting priorities."""
    # Sort by priority (highest first)
    sorted_blocks = sorted(blocks, key=lambda b: b.priority, reverse=True)

    selected = []
    remaining_budget = budget

    for block in sorted_blocks:
        if block.token_count <= remaining_budget:
            selected.append(block)
            remaining_budget -= block.token_count

    return selected
```

### 6.4 Token Budget Allocation

```python
def compute_budget(model_context_limit: int, reserved_for_output: int = 4096) -> dict:
    """Compute token budgets for each context component."""
    available = model_context_limit - reserved_for_output
    return {
        "system_prompt": min(2000, int(available * 0.05)),
        "tool_definitions": min(3000, int(available * 0.08)),
        "few_shot_examples": min(2000, int(available * 0.05)),
        "rag_documents": min(8000, int(available * 0.30)),
        "memory": min(2000, int(available * 0.05)),
        "conversation_history": int(available * 0.47),
        "output_reserved": reserved_for_output,
    }
```

### 6.5 Dynamic Context Assembly

Different queries need different context compositions:

```python
class DynamicContextAssembler:
    """Assemble context dynamically based on the query type."""

    async def assemble(
        self,
        query: str,
        conversation_history: list[dict],
        available_budget: int
    ) -> list[ContextBlock]:
        # Classify the query intent
        intent = await self.classify_intent(query)

        blocks = []

        # System prompt (always included)
        blocks.append(ContextBlock(
            content=self.get_system_prompt(intent),
            priority=10.0,
            category="system"
        ))

        # Tool definitions (only if intent requires tools)
        if intent in ("data_query", "action_request", "multi_step"):
            tools = self.get_relevant_tools(intent)
            blocks.append(ContextBlock(
                content=format_tools(tools),
                priority=9.0,
                category="tools"
            ))

        # RAG documents (only if intent requires knowledge)
        if intent in ("knowledge_question", "document_analysis"):
            docs = await self.retrieve(query, top_k=5)
            for i, doc in enumerate(docs):
                blocks.append(ContextBlock(
                    content=doc.content,
                    priority=7.0 - (i * 0.5),
                    category="rag"
                ))

        # Assemble within budget
        return assemble_context(blocks, available_budget)
```

### 6.6 From Prompt Engineering to Context Engineering

| Aspect | Prompt Engineering | Context Engineering |
|--------|-------------------|---------------------|
| **Scope** | Single prompt text | Entire context window |
| **Components** | Instructions + query | System + tools + RAG + history + memory + examples |
| **Optimization** | Wording and phrasing | Information selection, prioritization, budgeting |
| **Scale** | One-off interactions | Multi-turn, multi-tool, multi-source |
| **Tooling** | Manual iteration | Programmatic assembly with metrics |
| **Testing** | Trial and error | Evaluation suites measuring context utilization |

### 6.7 Cache-Friendly Context Ordering

Prompt caching (Week 7) is most effective when context has stable prefixes:

```python
# Cache-friendly ordering:
# 1. System prompt (static — always cached)
# 2. Tool definitions (static per session — usually cached)
# 3. Few-shot examples (static per task type — often cached)
# 4. RAG documents (dynamic — cache miss)
# 5. Conversation history (dynamic — cache miss)
# 6. Current message (dynamic — cache miss)
```

Placing static components first maximizes prefix cache hit rates, reducing both cost and latency.

---

## 7. Frameworks: LangGraph and PydanticAI

**Course Connection:** Week 2 (FastAPI, Pydantic), Week 5 (RAG), Week 8 (Observability)

### 7.1 LangGraph

LangGraph is a library for building stateful, multi-step agent applications as graphs. Agent behavior is modeled as a **state machine** where nodes perform actions and edges determine the next step.

#### 7.1.1 Core Concepts: Nodes, Edges, State

**State** is a typed object that flows through the graph:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Chat history (append-only)
    retrieved_docs: list[str]                # RAG results
    current_step: str                        # Tracking
    error_count: int                         # Error handling
```

**Nodes** are functions that take state and return updates:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

async def call_model(state: AgentState) -> dict:
    """Node: Call the LLM with current messages."""
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

async def retrieve_documents(state: AgentState) -> dict:
    """Node: Retrieve relevant documents."""
    query = state["messages"][-1].content
    docs = await vector_store.asimilarity_search(query, k=3)
    return {"retrieved_docs": [doc.page_content for doc in docs]}
```

**Conditional edges** allow branching:

```python
def should_retrieve(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tools"
    if needs_retrieval(last_message.content):
        return "retrieve"
    return "respond"
```

#### 7.1.2 Graph Construction

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("tools", execute_tools)

# Add edges
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", should_retrieve,
    {"retrieve": "retrieve", "use_tools": "tools", "respond": END}
)
workflow.add_edge("retrieve", "agent")
workflow.add_edge("tools", "agent")

app = workflow.compile()
```

#### 7.1.3 Cycles and Conditional Logic

Unlike DAG-only frameworks, LangGraph supports **cycles** — essential for agents that:
- Retrieve, reason, retrieve again
- Attempt an action, fail, retry
- Iterate until a quality threshold is met

```python
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return "end"
    if state.get("error_count", 0) > 3:
        return "end"
    return "continue"

workflow.add_conditional_edges(
    "tools", should_continue,
    {"continue": "agent", "end": END}
)
```

#### 7.1.4 Human-in-the-Loop

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tools"]  # Pause before tool execution
)

# Run until interrupt
config = {"configurable": {"thread_id": "user-123"}}
result = await app.ainvoke(
    {"messages": [HumanMessage(content="Delete all test data")]},
    config=config
)

# Human reviews, then resumes
result = await app.ainvoke(None, config=config)
```

#### 7.1.5 Checkpointing and Persistence

Saves full state at each node transition. Enables fault tolerance, time travel, and branching:

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/langgraph"
) as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)
```

#### 7.1.6 Subgraphs and Multi-Agent

```python
# Compose multiple agents
parent = StateGraph(OrchestratorState)
parent.add_node("research", research_agent)
parent.add_node("writing", writing_agent)
parent.add_edge("research", "writing")
parent.set_entry_point("research")
parent.add_edge("writing", END)
orchestrator = parent.compile()
```

#### 7.1.7 Code Example: ReAct Agent

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base."""
    return f"Results for: {query}"

@tool
def get_current_metrics(service_name: str) -> str:
    """Get performance metrics for a service."""
    return f"Metrics for {service_name}: p99=120ms, error_rate=0.1%"

tools = [search_knowledge_base, get_current_metrics]
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

async def agent(state: AgentState) -> dict:
    system = SystemMessage(content="You are an SRE assistant.")
    response = await llm.ainvoke([system] + state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_use_tools(state: AgentState) -> str:
    if getattr(state["messages"][-1], "tool_calls", None):
        return "tools"
    return "end"

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_use_tools, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

react_agent = graph.compile()
```

### 7.2 PydanticAI

PydanticAI is built by the creators of Pydantic. It emphasizes **type safety**, **dependency injection**, and **model-agnostic design** — inspired by FastAPI.

#### 7.2.1 Core Concepts

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class CityInfo(BaseModel):
    name: str
    country: str
    population: int
    notable_fact: str

agent = Agent(
    "openai:gpt-4o",
    system_prompt="You are a geography expert.",
    result_type=CityInfo  # Structured output validated by Pydantic
)

result = agent.run_sync("Tell me about Tokyo")
print(result.data)
# CityInfo(name='Tokyo', country='Japan', population=13960000, ...)
```

If validation fails, PydanticAI automatically retries with the error fed back to the model.

#### 7.2.2 Tools and Dependency Injection

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
import httpx

@dataclass
class SupportDeps:
    customer_id: int
    db_connection: any
    http_client: httpx.AsyncClient

agent = Agent(
    "openai:gpt-4o",
    deps_type=SupportDeps,
    system_prompt="You are a customer support agent."
)

@agent.tool
async def get_customer_orders(
    ctx: RunContext[SupportDeps], limit: int = 5
) -> str:
    """Retrieve the customer's recent orders."""
    orders = await ctx.deps.db_connection.fetch_orders(
        customer_id=ctx.deps.customer_id, limit=limit
    )
    return format_orders(orders)

# Run with injected dependencies
async def handle_request(customer_id: int, message: str):
    async with httpx.AsyncClient() as client:
        deps = SupportDeps(customer_id=customer_id, db_connection=db, http_client=client)
        result = await agent.run(message, deps=deps)
        return result.data
```

#### 7.2.3 Dynamic System Prompts

```python
@agent.system_prompt
async def build_system_prompt(ctx: RunContext[SupportDeps]) -> str:
    customer = await ctx.deps.db_connection.get_customer(ctx.deps.customer_id)
    return (
        f"You are assisting {customer.name}, a {customer.tier} tier customer. "
        f"Be {'extra attentive' if customer.tier == 'premium' else 'helpful'}."
    )
```

#### 7.2.4 Streaming

```python
async def stream_response(question: str):
    async with agent.run_stream(question) as result:
        async for text in result.stream_text():
            print(text, end="", flush=True)
```

#### 7.2.5 Model-Agnostic Design

```python
# Switch models easily
agent = Agent("openai:gpt-4o-mini")
result = await agent.run("Hello", model="anthropic:claude-sonnet-4-20250514")
```

#### 7.2.6 Integration with FastAPI

```python
from fastapi import FastAPI
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from dataclasses import dataclass
import httpx

app = FastAPI()

class SupportRequest(BaseModel):
    customer_id: int
    message: str

class SupportResponse(BaseModel):
    reply: str
    sentiment: str
    escalate: bool

@dataclass
class Deps:
    db: any
    http_client: httpx.AsyncClient

support_agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    result_type=SupportResponse,
    system_prompt="You are a customer support agent."
)

@support_agent.tool
async def lookup_order(ctx: RunContext[Deps], order_id: str) -> str:
    """Look up an order by ID."""
    return str(await ctx.deps.db.get_order(order_id))

@app.post("/support", response_model=SupportResponse)
async def handle_support(request: SupportRequest):
    async with httpx.AsyncClient() as client:
        deps = Deps(db=get_db(), http_client=client)
        result = await support_agent.run(request.message, deps=deps)
        return result.data  # Already a validated SupportResponse
```

### 7.3 Framework Comparison

| Dimension | LangGraph | PydanticAI |
|-----------|-----------|------------|
| **Core metaphor** | Graph / state machine | Type-safe function composition |
| **State management** | Explicit graph state | Implicit via messages |
| **Control flow** | Nodes + conditional edges + cycles | Sequential with tool calls |
| **Multi-agent** | First-class (subgraphs) | Composable but manual |
| **Human-in-the-loop** | Built-in (interrupt/resume) | Manual |
| **Checkpointing** | Built-in (Postgres, Redis, etc.) | Not built-in |
| **Type safety** | Moderate (TypedDict) | Strong (Pydantic v2) |
| **Dependency injection** | Not built-in | First-class |
| **Streaming** | Supported | Supported |
| **FastAPI integration** | Possible but separate | Natural (shared Pydantic) |
| **Learning curve** | Steeper (graph concepts) | Gentler (familiar patterns) |
| **Maturity** | More mature, larger ecosystem | Newer, rapidly evolving |
| **Best for** | Complex multi-step workflows, multi-agent | Type-safe agents, FastAPI apps |

**When to use LangGraph:**
- Complex control flow with cycles and conditional branching
- Multi-agent systems with handoffs
- Persistent, resumable workflows
- Human-in-the-loop with checkpointing

**When to use PydanticAI:**
- FastAPI applications with consistent typing
- Validated structured outputs
- Dependency injection for testability
- Straightforward agent workflows

**When to combine them:** Use PydanticAI for type-safe tool execution within LangGraph nodes. LangGraph handles orchestration; PydanticAI handles individual agent interactions.

---

## Further Reading

- **RAG:** Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020); Yan et al., "Corrective Retrieval Augmented Generation" (2024); Asai et al., "Self-RAG" (2023)
- **MCP:** Anthropic, "Model Context Protocol Specification" — https://modelcontextprotocol.io
- **DSPy:** Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2023) — https://github.com/stanfordnlp/dspy
- **CoT:** Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022); Wang et al., "Self-Consistency" (2022); Yao et al., "Tree of Thoughts" (2023)
- **ReAct:** Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
- **LangGraph:** https://langchain-ai.github.io/langgraph/
- **PydanticAI:** https://ai.pydantic.dev

---

*This appendix is part of the Building Production AI Systems course. For the core curriculum, see the weekly course notes in `course-notes/`.*
