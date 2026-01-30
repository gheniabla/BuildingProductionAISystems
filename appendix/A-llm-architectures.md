# Appendix A: LLM Architectures

A comprehensive technical reference supporting the Building Production AI Systems course.

---

## Table of Contents

1. [Transformers](#1-transformers)
2. [Flash Attention](#2-flash-attention)
3. [Quantization](#3-quantization)
4. [LoRA and QLoRA](#4-lora-and-qlora)
5. [Fine-Tuning](#5-fine-tuning)
6. [RLHF (Reinforcement Learning from Human Feedback)](#6-rlhf)
7. [DPO (Direct Preference Optimization)](#7-dpo)
8. [RAG (Retrieval-Augmented Generation)](#8-rag)
9. [Mixture of Experts (MoE)](#9-mixture-of-experts)
10. [Diffusion Models](#10-diffusion-models)
11. [DeepSeek Architecture](#11-deepseek-architecture)

---

## 1. Transformers

**Course Connection:** Foundational to all weeks — every component of the course builds on the Transformer architecture.

### 1.1 Overview

The Transformer architecture (Vaswani et al., 2017, "Attention Is All You Need") replaced recurrent neural networks as the dominant architecture for sequence modeling. Its key innovation is the **self-attention mechanism**, which allows every token in a sequence to attend to every other token in parallel, eliminating the sequential bottleneck of RNNs.

### 1.2 Self-Attention Mechanism

Self-attention computes a weighted sum of all values in a sequence, where the weights are determined by the similarity between queries and keys.

For an input sequence of embeddings `X` (shape: `[seq_len, d_model]`):

```
Q = X * W_Q    # Queries:  [seq_len, d_k]
K = X * W_K    # Keys:     [seq_len, d_k]
V = X * W_V    # Values:   [seq_len, d_v]

Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Where:
- `Q * K^T` computes pairwise similarity scores between all positions
- Division by `sqrt(d_k)` prevents the dot products from growing too large
- `softmax` converts scores to a probability distribution
- Multiplication by `V` produces the weighted output

**Computational complexity**: O(n² * d) where n is sequence length — this quadratic scaling with sequence length is a primary bottleneck for long contexts.

### 1.3 Multi-Head Attention

Rather than computing a single attention function, Multi-Head Attention runs `h` parallel attention heads with different learned projections:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

where head_i = Attention(Q * W_Qi, K * W_Ki, V * W_Vi)
```

Each head can attend to different types of relationships (syntactic, semantic, positional). Typical configurations: h=32 heads with d_k=128 for a d_model=4096 model.

### 1.4 Positional Encoding

Since self-attention is permutation-invariant (it has no inherent notion of token order), position information must be explicitly injected.

**Sinusoidal positional encoding** (original Transformer):
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Rotary Positional Embeddings (RoPE)** (used in most modern LLMs):
Encodes position by rotating the query and key vectors in 2D subspaces. Advantages:
- Relative position information is naturally captured
- Extrapolates better to unseen sequence lengths
- Compatible with KV caching

### 1.5 Encoder-Decoder vs. Decoder-Only

**Encoder-Decoder** (original Transformer, T5, BART):
- Encoder processes the full input with bidirectional attention
- Decoder generates output autoregressively with causal (left-to-right) attention
- Cross-attention connects decoder to encoder representations
- Used for: translation, summarization, seq2seq tasks

**Decoder-Only** (GPT, LLaMA, Mistral, DeepSeek):
- Single stack of Transformer blocks with causal attention mask
- Each token can only attend to itself and previous tokens
- Simpler architecture, scales better, dominates modern LLMs
- Used for: text generation, instruction following, reasoning

**Encoder-Only** (BERT, RoBERTa):
- Bidirectional attention over the full sequence
- Used for: classification, embeddings, token-level tasks

### 1.6 Key-Value (KV) Cache

During autoregressive generation, each new token requires attending to all previous tokens. Without optimization, this means recomputing the K and V projections for all previous tokens at every step.

The **KV cache** stores the computed K and V tensors from previous steps:

```
Step 1: Compute K_1, V_1 for token 1. Store in cache.
Step 2: Compute K_2, V_2 for token 2. Attend to [K_1, K_2], [V_1, V_2].
Step 3: Compute K_3, V_3 for token 3. Attend to [K_1, K_2, K_3], [V_1, V_2, V_3].
...
```

**KV cache size** per token per layer: `2 * n_heads * d_head * sizeof(dtype)`

For a 70B model (80 layers, 64 heads, d_head=128, FP16):
```
Per token: 2 * 64 * 128 * 2 bytes * 80 layers = 2.62 MB
For 4096 tokens: ~10.7 GB
For 128K tokens: ~335 GB
```

This is why KV cache management (Week 7) and architectures that compress the KV cache (like DeepSeek's MLA) are critical for production.

### 1.7 Context Windows

The context window is the maximum number of tokens the model can process in a single forward pass. Modern models range from 4K to 1M+ tokens.

**Implications for production:**
- Larger contexts enable more RAG documents, longer conversations, larger code files
- KV cache memory scales linearly with context length
- Attention computation scales quadratically (mitigated by Flash Attention)
- Quality may degrade for information in the middle of long contexts ("Lost in the Middle")

---

## 2. Flash Attention

**Course Connection:** Week 7 (Optimization), Week 6 (Deployment with vLLM)

### 2.1 The Memory Bottleneck

Standard attention computes the full `N x N` attention matrix, which must be materialized in GPU high-bandwidth memory (HBM). For a sequence of 8192 tokens:

```
Attention matrix size: 8192 * 8192 * 4 bytes (FP32) = 256 MB per head
With 32 heads: ~8 GB just for attention scores
```

The bottleneck is not compute (GPUs have plenty of FLOPs) but **memory I/O** — reading and writing this large matrix to and from HBM is slow.

### 2.2 How Flash Attention Works

Flash Attention (Dao et al., 2022) avoids materializing the full attention matrix by using **tiling** and **kernel fusion**:

**Tiling**: Instead of computing the full N×N matrix, Flash Attention processes the attention computation in blocks (tiles) that fit in GPU SRAM (on-chip memory, ~20MB). Each tile computes a partial softmax using the online softmax trick:

```
For each block of queries Q_block:
    For each block of keys K_block, values V_block:
        1. Compute S_block = Q_block * K_block^T / sqrt(d_k)
        2. Update running softmax statistics (max, sum)
        3. Accumulate output: O += corrected_softmax(S_block) * V_block

    Final output for this query block is ready (exact, not approximate)
```

**Kernel fusion**: The entire attention computation (matrix multiply, scaling, softmax, dropout, value weighting) is fused into a single GPU kernel. This eliminates intermediate reads/writes to HBM.

**Results**:
- **Memory**: O(N) instead of O(N²) — no materialized attention matrix
- **Speed**: 2-4x faster than standard attention due to reduced HBM I/O
- **Exact**: Not an approximation — produces identical results to standard attention

### 2.3 Flash Attention 2

Flash Attention 2 (Dao, 2023) improves on the original:
- Better work partitioning across GPU thread blocks
- Reduced non-matmul FLOPs (important on modern GPUs where matmul is disproportionately fast)
- Up to 2x faster than Flash Attention 1
- Support for head dimensions up to 256

### 2.4 Production Impact

Flash Attention is enabled by default in vLLM, HuggingFace Transformers, and most modern serving frameworks:

- **Longer contexts**: Makes 32K-128K contexts practical without OOM errors
- **Higher throughput**: Faster attention means more tokens per second per GPU
- **Lower cost**: Same hardware serves more requests
- **Paged Attention** (used in vLLM) extends Flash Attention's ideas to manage KV cache memory dynamically across requests

---

## 3. Quantization

**Course Connection:** Week 7 (Optimization — Quantization, Model Compression)

### 3.1 Numeric Precision Formats

Neural networks traditionally use 32-bit floating point (FP32). Quantization reduces the precision of weights (and sometimes activations) to smaller formats:

| Format | Bits | Range | Use Case |
|--------|------|-------|----------|
| FP32 | 32 | ±3.4×10³⁸ | Training (baseline) |
| BF16 | 16 | ±3.4×10³⁸ (reduced mantissa) | Training & inference |
| FP16 | 16 | ±65504 | Inference (standard) |
| FP8 (E4M3) | 8 | ±448 | Training & inference (H100+) |
| INT8 | 8 | -128 to 127 | Inference (weight + activation) |
| INT4 | 4 | -8 to 7 | Inference (weights only) |

**BF16 vs FP16**: BF16 has the same exponent range as FP32 (avoids overflow) but less precision. FP16 has more precision but smaller range. BF16 is generally preferred for LLM training.

### 3.2 Post-Training Quantization (PTQ)

Quantization applied after training is complete. No retraining required.

**Weight-only quantization**: Only model weights are quantized; activations remain in FP16/BF16. Simpler and often sufficient.

**Weight + activation quantization**: Both weights and activations are quantized. More complex (requires calibration data) but enables faster integer arithmetic.

**Calibration**: A small dataset is passed through the model to determine the dynamic range of activations, which informs the quantization scale factors.

### 3.3 Quantization-Aware Training (QAT)

Quantization is simulated during training, allowing the model to adapt to reduced precision. Produces better quality than PTQ but requires retraining.

**Straight-Through Estimator (STE)**: During forward pass, weights are quantized. During backward pass, gradients flow through the quantization operation as if it were the identity function.

### 3.4 Popular Quantization Methods

**GPTQ** (Frantar et al., 2023):
- Layer-wise weight quantization using approximate second-order information
- Typically INT4 weights with groupwise quantization
- Good balance of quality and compression
- Wide support in inference frameworks

**AWQ** (Activation-Aware Weight Quantization, Lin et al., 2023):
- Observes that not all weights are equally important — some channels carry more salient activations
- Protects salient weight channels with higher precision
- Often better quality than GPTQ at INT4

**GGUF** (llama.cpp format):
- CPU-friendly quantization format with multiple precision levels (Q2_K through Q8_0)
- Supports mixed precision within a model (important layers at higher precision)
- Enables running models on consumer hardware (CPU + limited GPU)

### 3.5 Trade-offs

```
Quality:     FP32 > BF16 ≈ FP16 > INT8 > INT4 (generally)
Speed:       INT4 > INT8 > FP16 > FP32 (for weight-only quantization*)
Memory:      INT4 (4x reduction) > INT8 (2x) > FP16 (2x vs FP32)
```

*Weight-only quantization speed gains come from reduced memory bandwidth, not faster compute. The weights are dequantized to FP16 before matrix multiplication.

**Quality degradation patterns:**
- Small models (< 7B) degrade more with quantization than large models
- INT8 is nearly lossless for most models > 13B
- INT4 can cause noticeable quality loss, especially in math and reasoning
- Perplexity metrics may not capture all degradation — evaluate on your specific task

### 3.6 Production Recommendations

1. **Start with BF16/FP16** — 2x compression with negligible quality loss
2. **Try INT8 (AWQ or GPTQ)** — 4x compression, usually < 1% quality loss for large models
3. **Use INT4 only when necessary** — significant memory savings but test quality carefully
4. **vLLM supports GPTQ, AWQ, and FP8** — integrates directly with deployment pipeline (Week 6)
5. **Always benchmark on your evaluation suite** (Week 3) before deploying quantized models

---

## 4. LoRA and QLoRA

**Course Connection:** Week 7 (Optimization), Week 6 (Deployment — serving multiple adapters)

### 4.1 The Problem: Full Fine-Tuning Cost

Fine-tuning a 70B parameter model requires:
- ~280 GB of GPU memory just for model weights (FP32) + optimizer states
- Multiple high-end GPUs (8x A100 80GB minimum)
- Significant compute time and cost

### 4.2 LoRA: Low-Rank Adaptation

LoRA (Hu et al., 2021) freezes the original model weights and adds small trainable "adapter" matrices alongside selected weight matrices.

**Key insight**: The weight updates during fine-tuning have low intrinsic rank — they can be well-approximated by low-rank matrices.

For a weight matrix `W` (shape: `d × d`):

```
Original:    Output = W * x
With LoRA:   Output = W * x + (B * A) * x

Where:
  W: frozen original weights       [d × d]      (not trained)
  A: down-projection adapter       [d × r]      (trained, r << d)
  B: up-projection adapter         [r × d]      (trained, r << d)
  r: rank (typically 8-64)
```

**Parameter savings**: For a d=4096 matrix with rank r=16:
```
Full fine-tuning:  4096 × 4096 = 16.7M parameters
LoRA:              4096 × 16 + 16 × 4096 = 131K parameters (128x fewer)
```

**Which layers**: LoRA is typically applied to the attention projection matrices (Q, K, V, O) and sometimes the FFN layers.

**Scaling factor**: The LoRA output is scaled by `alpha/r` where `alpha` is a hyperparameter (typically equal to r or 2r). This controls the magnitude of the adaptation.

### 4.3 QLoRA

QLoRA (Dettmers et al., 2023) combines quantization with LoRA for even greater efficiency:

1. **Base model is quantized to 4-bit** (NF4 — Normal Float 4, a data type designed for normally-distributed weights)
2. **LoRA adapters are trained in BF16** on top of the quantized base
3. **Double quantization**: The quantization constants themselves are quantized (saving additional memory)
4. **Paged optimizers**: Optimizer states are offloaded to CPU when GPU memory is insufficient

**Memory comparison for a 70B model:**
```
Full fine-tuning (FP32):           ~280 GB
Full fine-tuning (BF16):           ~140 GB
LoRA (BF16 base):                  ~140 GB + ~100 MB adapters
QLoRA (4-bit base + BF16 adapters): ~35 GB + ~100 MB adapters
```

QLoRA makes fine-tuning a 70B model possible on a single 48GB GPU.

### 4.4 Production Considerations

**Adapter merging**: After training, LoRA weights can be merged into the base model:
```python
W_merged = W + (B * A) * (alpha / r)
```
This produces a standard model with zero inference overhead.

**Serving multiple adapters**: With frameworks like vLLM and LoRAX, a single base model in GPU memory can serve multiple LoRA adapters simultaneously:
```
Base model (shared):     35 GB GPU memory
LoRA adapter 1:          ~100 MB
LoRA adapter 2:          ~100 MB
...
LoRA adapter N:          ~100 MB
```
Each request can specify which adapter to use. The base model KV cache is shared; only the adapter weights differ.

**When to use LoRA vs full fine-tuning:**
- LoRA: domain adaptation, instruction tuning, when you need multiple task-specific models
- Full fine-tuning: maximum quality, single-task deployment, when compute is not a constraint

---

## 5. Fine-Tuning

**Course Connection:** Week 7 (Optimization), supports evaluation decisions in Week 3

### 5.1 Fine-Tuning Approaches

**Full fine-tuning**: Update all model parameters. Highest quality ceiling but most expensive. Requires the full training infrastructure.

**Parameter-efficient fine-tuning (PEFT)**: Update only a small subset of parameters. Includes LoRA/QLoRA (see above), prefix tuning, prompt tuning, and adapter layers.

### 5.2 Supervised Fine-Tuning (SFT)

The most common fine-tuning approach for LLMs. Train on (input, desired_output) pairs using the standard language modeling loss:

```
L = -sum(log P(token_i | token_1, ..., token_{i-1}))
```

Typically, the loss is computed only on the output tokens (not the input/instruction tokens).

**Data format** (common conversation format):
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize this article: ..."},
    {"role": "assistant", "content": "The article discusses..."}
  ]
}
```

### 5.3 Instruction Tuning

A specific form of SFT focused on teaching the model to follow diverse instructions. Uses datasets of (instruction, response) pairs across many task types.

Key datasets: OpenAssistant, ShareGPT, Alpaca, UltraChat.

### 5.4 Domain Adaptation

Fine-tuning a general model on domain-specific data (legal, medical, financial). Can involve:

1. **Continued pre-training**: Train on domain-specific text with the language modeling objective (no instruction format). Teaches the model domain vocabulary and knowledge.
2. **Domain SFT**: Fine-tune on domain-specific instruction-response pairs. Teaches the model to apply domain knowledge in an instruction-following context.

### 5.5 Dataset Preparation

Quality of fine-tuning data is more important than quantity:

- **Minimum viable dataset**: Often 1,000-10,000 high-quality examples are sufficient for SFT
- **Data cleaning**: Remove duplicates, low-quality examples, and potentially harmful content
- **Format consistency**: Ensure consistent conversation formatting
- **Diversity**: Cover the range of tasks and edge cases you expect in production
- **Decontamination**: Remove examples that overlap with your evaluation set

### 5.6 When to Fine-Tune vs. Alternatives

| Approach | Best For | Cost | Freshness |
|----------|----------|------|-----------|
| Prompt engineering | Quick iteration, simple tasks | Lowest | Real-time |
| RAG | Knowledge-intensive, changing data | Medium | Near real-time |
| Fine-tuning (LoRA) | Style, format, domain behavior | Medium-High | Static |
| Full fine-tuning | Maximum quality, unique capabilities | Highest | Static |

**Decision framework:**
1. Try prompt engineering first
2. If the model lacks knowledge → add RAG
3. If the model's behavior/style/format is wrong → fine-tune
4. If the model lacks fundamental capability → consider fine-tuning on more data or a larger model

---

## 6. RLHF (Reinforcement Learning from Human Feedback)

**Course Connection:** Understanding model alignment for production quality (Week 3 Evaluation)

### 6.1 The Three Stages

RLHF aligns language models with human preferences through three stages:

**Stage 1: Supervised Fine-Tuning (SFT)**
Start with a pre-trained model and fine-tune it on high-quality demonstrations:
```
Input:  "Write a poem about autumn"
Output: [High-quality human-written poem]
```

**Stage 2: Reward Model Training**
Train a separate model to predict human preferences:
```
Given:   A prompt and two model responses (A and B)
Human:   Labels which response is preferred
Reward:  Learns to assign scalar scores reflecting preference

Loss = -log(sigmoid(r(preferred) - r(rejected)))
```

The reward model is typically initialized from the SFT model with a scalar output head replacing the language modeling head.

**Stage 3: PPO (Proximal Policy Optimization)**
Use the reward model to optimize the language model via reinforcement learning:
```
Objective = E[r(response)] - beta * KL(policy || reference)

Where:
  r(response):     Reward model score
  KL divergence:   Prevents the model from deviating too far from the SFT model
  beta:            Controls the KL penalty strength
```

PPO clips the policy update to prevent destructively large changes:
```
L_PPO = min(ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon) * advantage)
```

### 6.2 Challenges

**Reward hacking**: The model finds responses that score highly with the reward model but are not genuinely preferred by humans. Example: verbosely restating the question before answering (inflates reward scores without adding value).

**Distribution shift**: As the policy model changes during PPO, its outputs diverge from the distribution the reward model was trained on, potentially making reward scores unreliable.

**Annotation quality**: The entire pipeline depends on the quality and consistency of human preference annotations. Ambiguous comparisons, annotator disagreement, and cultural biases all propagate into the final model.

**Training instability**: PPO is notoriously difficult to tune — learning rates, KL penalty, clipping parameters, and batch sizes all interact in complex ways.

### 6.3 Role in Production

RLHF is primarily a model training technique (performed by model providers like OpenAI, Anthropic, Meta). For production engineers, the key implications are:

- RLHF-trained models are better at following instructions and refusing harmful requests
- The reward model can be repurposed as an evaluation metric (Week 3)
- Understanding RLHF helps explain model behavior patterns (e.g., why models sometimes refuse benign requests)

---

## 7. DPO (Direct Preference Optimization)

**Course Connection:** Alternative alignment approach relevant to model selection and evaluation (Week 3)

### 7.1 Simplifying RLHF

DPO (Rafailov et al., 2023) eliminates the need for a separate reward model and PPO training. It directly optimizes the language model on preference data.

**Key insight**: The optimal policy under the RLHF objective has a closed-form relationship to the reward function:

```
r(x, y) = beta * log(pi(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

This means we can substitute the policy directly into the preference optimization objective, bypassing the reward model entirely.

### 7.2 The DPO Loss

```
L_DPO = -E[log sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))))]

Where:
  y_w: the preferred (winning) response
  y_l: the rejected (losing) response
  pi:  the policy being trained
  pi_ref: the reference (SFT) policy
  beta: temperature parameter
```

In plain language: increase the probability of preferred responses relative to the reference model, and decrease the probability of rejected responses, with the balance controlled by beta.

### 7.3 DPO vs RLHF

| Aspect | RLHF | DPO |
|--------|------|-----|
| Complexity | 3-stage pipeline | Single-stage training |
| Reward model | Required (separate training) | Not required |
| Stability | PPO can be unstable | More stable (standard supervised loss) |
| Memory | Needs reward model + policy + reference | Needs policy + reference |
| Hyperparameters | Many (PPO + reward) | Few (mainly beta) |
| Quality ceiling | Potentially higher (iterative reward) | Comparable for most use cases |
| Online learning | Supports new data easily | Requires retraining |

### 7.4 When RLHF is Still Preferred

- When you need **online/iterative** alignment (generating new responses and getting feedback)
- When your reward function is **non-binary** (e.g., fine-grained quality scores)
- For **very large-scale** alignment where the reward model enables more efficient exploration
- When you want to **reuse the reward model** for evaluation or filtering

### 7.5 Production Implications

DPO has made alignment more accessible:
- Smaller teams can align models without the infrastructure overhead of PPO
- Fine-tuning services (Together, Fireworks) offer DPO as a standard option
- Combined with LoRA, DPO alignment can be done on modest hardware
- The preference data format is simpler to collect and curate

---

## 8. RAG (Retrieval-Augmented Generation)

**Course Connection:** Week 5 (RAG Systems Deep Dive), Week 4 (Security — secure retrieval)

### 8.1 Architecture Overview

RAG augments an LLM's generation with information retrieved from an external knowledge base:

```
User Query
    |
    v
[Retriever] --> Search knowledge base --> Top-K relevant documents
    |
    v
[Generator (LLM)] <-- Query + Retrieved documents --> Response
```

**Why RAG:**
- Models have a knowledge cutoff — RAG provides current information
- Reduces hallucination by grounding responses in source documents
- Enables attribution and citation
- More cost-effective than fine-tuning for knowledge updates

### 8.2 Embedding Models and Vector Databases

**Embedding models** convert text into dense vectors that capture semantic meaning. These are encoder-only or encoder-decoder Transformers fine-tuned for semantic similarity.

Key properties:
- **Dimension**: Typically 384, 768, or 1024 dimensions
- **Similarity metric**: Cosine similarity or dot product
- **Normalization**: Many models output normalized vectors

Popular embedding models (as of 2025):

| Model | Dimensions | Notes |
|-------|-----------|-------|
| text-embedding-3-large (OpenAI) | 3072 | Commercial API |
| BGE-large-en-v1.5 | 1024 | Open source |
| E5-mistral-7b | 4096 | Large but powerful |
| Cohere embed-v3 | 1024 | Commercial API |
| nomic-embed-text-v1.5 | 768 | Open source, efficient |

**Vector databases** provide efficient approximate nearest neighbor (ANN) search:

| Database | Type | Key Feature |
|----------|------|-------------|
| FAISS | Library | Facebook's library; very fast, no server needed |
| Pinecone | Managed service | Fully managed, easy to start |
| Weaviate | Self-hosted/cloud | Hybrid search built-in |
| Qdrant | Self-hosted/cloud | Rust-based, high performance |
| Milvus | Self-hosted | Scalable, Kubernetes-native |
| pgvector | PostgreSQL extension | Integrates with existing Postgres |

**ANN algorithms**:
- **HNSW** (Hierarchical Navigable Small World): Graph-based. Best recall/speed tradeoff for most cases. Higher memory usage.
- **IVF** (Inverted File Index): Partition-based. Lower memory, slower search. Good for very large datasets.
- **PQ** (Product Quantization): Compresses vectors to reduce memory. Often combined with IVF (IVF-PQ).

### 8.3 Chunking Strategies

Documents must be split into chunks before embedding. The chunking strategy significantly impacts retrieval quality.

**Fixed-size chunking**:
```python
def fixed_chunk(text, chunk_size=512, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
```

**Recursive character splitting** (LangChain default):
Splits on paragraphs first, then sentences, then words, recursively, to stay within chunk size while respecting natural boundaries.

**Semantic chunking**:
Uses embedding similarity between adjacent sentences to identify natural breakpoints.

**Document-structure-aware chunking**:
Uses markdown headers, HTML structure, or PDF layout to chunk along document boundaries.

**Chunk size considerations**:
- **Too small** (< 100 tokens): Lacks context, poor retrieval quality
- **Too large** (> 1000 tokens): Dilutes relevant information with noise
- **Sweet spot**: 256-512 tokens with 50-100 token overlap

### 8.4 Hybrid Search: Dense + Sparse

**Dense retrieval** (embedding-based): Captures semantic similarity. "automobile" matches "car." But can miss exact keyword matches.

**Sparse retrieval** (BM25/TF-IDF): Captures lexical overlap. Excellent for exact terms, names, identifiers. But misses synonyms.

**Hybrid search** combines both:
```
score_hybrid = alpha * score_dense + (1 - alpha) * score_sparse
```

**Reciprocal Rank Fusion (RRF)** is an alternative that doesn't require score normalization:
```
RRF_score(d) = sum( 1 / (k + rank_i(d)) )  for each retrieval method i
```
Where `k` is typically 60.

### 8.5 Reranking and Context Assembly

**Reranking** applies a more powerful (but slower) cross-encoder model to re-score the top-k results:

```
1. Retrieve top-100 chunks via embedding search
2. Rerank with cross-encoder model --> top-10
3. Assemble into prompt context
```

Cross-encoder rerankers (Cohere Rerank, BGE-reranker) process query and document together through a Transformer, allowing full cross-attention.

**Context assembly** with metadata:
```python
context = "\n\n".join([
    f"[Source: {c.metadata['source']}, Page: {c.metadata['page']}]\n{c.text}"
    for c in top_k_chunks
])
```

### 8.6 RAG vs Fine-Tuning

| Dimension | RAG | Fine-Tuning |
|-----------|-----|-------------|
| Knowledge freshness | Current (update index anytime) | Static (frozen at training time) |
| Attribution | Can cite sources | Cannot attribute knowledge |
| Hallucination | Reduced (grounded in text) | Can still hallucinate |
| Cost per query | Higher (retrieval + longer prompt) | Lower (shorter prompt) |
| Setup cost | Moderate (indexing pipeline) | High (data, compute, evaluation) |
| Style/format adaptation | No | Yes |

**Combine them** when you need both current knowledge (RAG) and specific behavioral patterns (fine-tuning).

### 8.7 Production Considerations

1. **Indexing pipeline reliability**: Build monitoring for ingestion failures, embedding drift, and index staleness
2. **Retrieval quality monitoring**: Track MRR, NDCG in production. Log retrieved chunks for debugging
3. **Context window management**: "Lost in the Middle" — place most relevant information first and last
4. **Latency budget**: Retrieval adds 50-200ms; reranking adds 50-100ms
5. **Cost management**: Cache common queries. Batch embedding calls
6. **Security (Week 4)**: Retrieved documents may contain prompt injection. Sanitize retrieved content

---

## 9. Mixture of Experts (MoE)

**Course Connection:** Week 6 (Deployment — model selection), Week 7 (Optimization — compute efficiency)

### 9.1 Sparse Activation Explained

Standard "dense" Transformers activate all parameters for every input token. MoE uses **sparse activation** — only a subset of parameters is activated per token:

```
Dense Model (7B):    Every token uses all 7B parameters.
MoE Model (47B):     Every token uses ~13B parameters (2 of 8 experts + shared).
                     Total parameters: 47B. Active parameters: ~13B.
```

This decouples **model capacity** (total parameters) from **computational cost** (active parameters per token).

### 9.2 Router and Gating Mechanism

In each MoE layer, the standard FFN is replaced by N expert FFNs and a **router**:

```
MoE Layer:

  Input x
     |
     v
  [Router: W_gate * x --> logits for N experts]
     |
     v
  Select top-K experts (typically K=2)
     |
     +-----> Expert 1: FFN_1(x)  * gate_1
     +-----> Expert 4: FFN_4(x)  * gate_4
     |
     v
  Output = gate_1 * FFN_1(x) + gate_4 * FFN_4(x)
```

**Router formulation**:
```
g(x) = softmax(TopK(W_gate * x))
```

### 9.3 Load Balancing

Without intervention, routing tends to collapse — a few experts receive most tokens.

**Auxiliary load balancing loss** (Switch Transformer):
```
L_balance = alpha * N * sum_i(f_i * P_i)

where:
  f_i = fraction of tokens routed to expert i
  P_i = average routing probability for expert i
  alpha = balancing coefficient (typically 0.01)
```

### 9.4 Challenges

- **Memory**: All expert weights must be in memory, even though only K are active per token
- **Expert collapse**: Some experts may never be selected
- **Routing instability**: Training is less reproducible
- **Communication overhead**: Expert parallelism requires all-to-all communication
- **Batching**: Tokens route to different experts, creating irregular computation

### 9.5 Notable MoE Models

| Model | Total Params | Active Params | Experts | Top-K |
|-------|-------------|---------------|---------|-------|
| Switch Transformer | 1.6T | ~1.6B | 128 | 1 |
| Mixtral 8x7B | 47B | ~13B | 8 | 2 |
| Mixtral 8x22B | 176B | ~44B | 8 | 2 |
| DBRX | 132B | ~36B | 16 | 4 |
| DeepSeek-V2 | 236B | ~21B | 160 | 6 |
| DeepSeek-V3 | 671B | ~37B | 256 | 8 |

**Production relevance**: MoE models offer frontier-level quality at reduced per-token compute. However, their large memory footprint means deployment requires high-memory GPU setups.

---

## 10. Diffusion Models

**Course Connection:** Week 6 (Deployment), Week 7 (Optimization — applies to image generation)

### 10.1 Forward and Reverse Diffusion

Diffusion models generate data by learning to reverse a gradual noising process.

**Forward process** (fixed):
Starting from a clean data sample `x_0`, progressively add Gaussian noise over T timesteps:
```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
```

Using reparameterization with `alpha_bar_t = prod_{s=1}^{t} (1 - beta_s)`:
```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
where epsilon ~ N(0, I)
```

At t = T, x_T is approximately pure Gaussian noise.

**Reverse process** (learned):
```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
```

The neural network predicts the noise `epsilon_theta(x_t, t)` that was added.

### 10.2 Denoising Score Matching

Training objective:
```
L = E_{t, x_0, epsilon}[ || epsilon - epsilon_theta(x_t, t) ||^2 ]
```

Simple MSE between actual noise and predicted noise.

**Training procedure**:
```
1. Sample x_0 from training data
2. Sample t uniformly from {1, ..., T}
3. Sample epsilon ~ N(0, I)
4. Compute x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
5. Train network to predict epsilon from x_t and t
6. Loss = || epsilon - epsilon_theta(x_t, t) ||^2
```

### 10.3 U-Net Architecture

The denoising network is typically a **U-Net** — encoder-decoder with skip connections:

```
Input (x_t + t_emb)
  |
  [DownBlock] --> skip 1 --------+
  [DownBlock] --> skip 2 -----+  |
  [DownBlock] --> skip 3 --+  |  |
  |                        |  |  |
  [MiddleBlock + Attn]    |  |  |
  |                        |  |  |
  [UpBlock] <-- skip 3 ---+  |  |
  [UpBlock] <-- skip 2 ------+  |
  [UpBlock] <-- skip 1 ---------+
  |
Output (predicted noise)
```

Newer architectures (DiT) replace U-Net with a Transformer processing image patches as tokens.

### 10.4 Latent Diffusion and Stable Diffusion

Operating in pixel space is prohibitively expensive. **Latent Diffusion Models** operate in compressed space:

```
Image (512x512x3) --> [VAE Encoder] --> Latent (64x64x4) --> [Diffusion] --> [VAE Decoder] --> Image
```

~64x computational reduction. Text conditioning via CLIP/T5 embeddings injected through cross-attention.

### 10.5 Classifier-Free Guidance

Improves text-image alignment by amplifying the conditioning signal:
```
epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)
```
Where `w` is guidance scale (typically 5-15).

### 10.6 Production Considerations

**Differences from autoregressive LLMs:**
- Fixed compute (20-50 denoising steps, not proportional to output)
- No KV cache management
- Near-linear throughput scaling with batching
- Typically 2-10 seconds per image

**Optimization:**
- Better samplers (DPM++, Euler) for fewer steps
- Consistency distillation for 1-4 step generation
- TensorRT optimization for 2-3x speedup
- FP16 standard; INT8 possible with calibration

---

## 11. DeepSeek Architecture

**Course Connection:** Week 6 (Deployment — model selection), Week 7 (Optimization — architectural efficiency)

DeepSeek-V2 and V3 introduced architectural innovations that significantly reduce inference cost while maintaining frontier quality.

### 11.1 Multi-Head Latent Attention (MLA)

Standard attention stores separate K and V projections for each head in the KV cache. MLA compresses these into a low-dimensional latent:

**Standard MHA KV cache** (per token, per layer):
```
KV_size = 2 * n_heads * d_head
```
For 128 heads, d_head=128: `2 * 128 * 128 = 32,768` values per token per layer.

**MLA approach**:
```
# Compression (during generation):
c_KV = W_DKV * h        # d_model --> d_c (compressed)

# Decompression (during attention):
K = W_UK * c_KV          # Reconstruct keys
V = W_UV * c_KV          # Reconstruct values
```

DeepSeek-V2 uses d_c = 512 vs 32,768 for standard MHA — a **93.75% reduction** in KV cache size.

**Position encoding**: MLA separates position-sensitive and position-insensitive components:
```
# Position-insensitive (from compressed latent):
Q_nope, K_nope = from compressed representations

# Position-sensitive (separate, small):
Q_rope = RoPE(W_QR * c_Q)     # d_rope ≈ 64
K_rope = RoPE(W_KR * h)       # d_rope ≈ 64

# Combined:
Attention = [Q_nope; Q_rope] . [K_nope; K_rope]^T / sqrt(d)
```

Total KV cache per token: d_c + d_rope (576) vs 32,768 for standard MHA.

**Algebraic absorption**: During inference, the decompression matrices can be absorbed into Q and output projections, adding zero inference FLOPs.

### 11.2 DeepSeekMoE

Two key innovations over standard MoE:

**Fine-grained expert segmentation**:
```
Mixtral:       8 experts, select top-2    --> C(8,2) = 28 combinations
DeepSeek-V2:  160 experts, select top-6   --> billions of combinations
DeepSeek-V3:  256 experts, select top-8   --> trillions of combinations
```

Finer-grained experts allow more precise specialization and dramatically more routing diversity.

**Shared experts**: Always-active experts that capture common knowledge:
```
Output = SharedExpert(x) + sum(gate_i * RoutedExpert_i(x))
```

Shared experts reduce redundancy across routed experts and mitigate expert collapse.

**DeepSeek-V3**: 256 routed + 1 shared expert per MoE layer. 671B total, ~37B active per token. Uses auxiliary-loss-free load balancing.

### 11.3 Multi-Token Prediction (MTP)

Standard autoregressive models predict one token at a time. DeepSeek-V3 predicts the next k tokens:

```
Position t: predict token t+1 (standard)
            predict token t+2 (additional MTP head)
```

Each MTP head:
```
h_MTP_d = TransformerBlock_d(concat(h_main_t, embed(token_{t+d-1})))
logits_d = LM_head(RMSNorm(h_MTP_d))
```

**Training loss**: `L = L_next_token + lambda * L_future_tokens`

**Benefits**:
1. **Richer training signal**: Denser supervision per example
2. **Better representations**: Forces planning ahead
3. **Speculative decoding**: MTP heads serve as draft models for 1.5-2x inference speedup

### 11.4 Production Relevance

**Cost comparison** (approximate, for equivalent quality):
```
Traditional dense model: 100% cost baseline
Standard MoE (Mixtral): ~40-50% (fewer active params, same memory)
DeepSeek-V3:            ~20-30% (fewer active params + smaller KV cache)
```

**KV cache reduction (MLA)**: Enables dramatically larger batch sizes — fewer GPUs for the same concurrent user count. Especially impactful for long-context RAG applications.

**Compute efficiency (MoE)**: 671B capacity at 37B compute cost. Fine-grained routing provides better quality per FLOP.

**Inference speed (MTP)**: Built-in speculative decoding without a separate draft model.

**Lesson**: When selecting models for production, KV cache efficiency and active parameter count should be primary considerations alongside benchmark scores.

---

## Further Reading

**Foundational Papers:**
- Vaswani et al. (2017). "Attention Is All You Need."
- Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."
- Dao (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning."

**Quantization and Efficiency:**
- Frantar et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers."
- Lin et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration."
- Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized Language Models."
- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models."

**Alignment:**
- Ouyang et al. (2022). "Training language models to follow instructions with human feedback." (InstructGPT / RLHF)
- Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model."

**Architectures:**
- Fedus et al. (2022). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity."
- Jiang et al. (2024). "Mixtral of Experts."
- DeepSeek-AI (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model."
- DeepSeek-AI (2024). "DeepSeek-V3 Technical Report."

**Diffusion Models:**
- Ho et al. (2020). "Denoising Diffusion Probabilistic Models."
- Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models."

**RAG:**
- Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."

---

*This appendix is part of the Building Production AI Systems course. For questions and corrections, refer to the course repository.*
