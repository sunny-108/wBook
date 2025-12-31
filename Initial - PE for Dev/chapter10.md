# Chapter 10: Context Engineering – Supplying the Right Information to LLMs

Context engineering is the disciplined practice of curating, structuring, prioritizing, and delivering the **right external information** to a Large Language Model (LLM) so that its reasoning, generation, or decision-making aligns with real-world constraints and goals. While **prompt engineering** focuses on *how* you ask, **context engineering** focuses on *what the model knows at inference time*. Together they form a dual lever: wording + knowledge.

---
## 10.1 Definition & Relationship to Prompt Engineering

**Prompt Engineering**: Crafting instructions, roles, formatting, and constraints that steer the model's behavior.

**Context Engineering**: Selecting, shaping, and injecting supplemental information (retrieved documents, code, configuration, telemetry, user state, historical turns) that the model will condition on.

**Key Distinction:** Prompt engineering optimizes *instructional clarity*; context engineering optimizes *informational relevance*.

**Overlap:** Both require understanding task intent, constraints, and evaluation criteria. Both influence token efficiency, latency, quality, and safety.

**Synergy Analogy:** Prompt = "question architecture"; Context = "knowledge payload." A great prompt with poor context yields generic answers; rich context with a vague prompt yields misaligned answers. Excellence requires both.

### 10.1.1 Why Context Engineering Matters
- LLMs are **stateless across requests** (unless you supply state). Context carries continuity.
- Reduces hallucinations by anchoring outputs in authoritative sources.
- Enables domain adaptation without fine-tuning.
- Drives compliance: only expose approved, sanitized materials.
- Improves determinism and reproducibility.

### 10.1.2 Context Surfaces
- Inline prompt text (system/user messages)
- Tool outputs (retrieval results, API responses)
- Function call arguments / structured JSON
- Hidden system instructions (governance policies)
- Conversation history pruning strategy

---
## 10.2 Core Components of Context Payloads

1. **Source Selection** – Which repositories (code, docs, KB, DB fragments) are candidates?
2. **Retrieval Strategy** – Keyword, semantic, hybrid, graph traversal, metadata filtering.
3. **Chunking & Windowing** – How content is split (size, overlap, semantic boundaries).
4. **Ranking & Scoring** – BM25, dense similarity, RRF, recency, authority weighting.
5. **Compression** – Summarization, entity extraction, semantic folding, lossy vs lossless.
6. **Structuring & Formatting** – Section headers, provenance metadata, JSON schemas.
7. **Prioritization & Ordering** – High-signal first; critical constraints before examples.
8. **Context Budget Management** – Token allocation per layer (instructions vs facts vs examples).
9. **Freshness & Validity Controls** – TTL, cache invalidation, version pinning.
10. **Safety & Redaction** – PII scrub, policy filters, license compliance.
11. **Observability & Metrics** – Retrieval logs, hit-rate, usage heatmaps.
12. **Governance & Access Control** – Who/what can be retrieved; masking logic.

---
## 10.3 Guiding Principles

1. **Relevance Density** – Maximize useful tokens / total tokens.
2. **Minimal Sufficient Set** – Provide *just enough* to solve; avoid distraction.
3. **Hierarchical Layering** – Core facts → constraints → examples → supplemental detail.
4. **Provenance Transparency** – Always show (or internally track) origins for audit.
5. **Deterministic Assembly** – Idempotent retrieval + ordering rules.
6. **Safety by Design** – Redact before model ingestion, never after generation only.
7. **Adaptive Context** – Tailor slices to task type, user persona, and phase (draft vs refine).
8. **Token Economy** – Track marginal utility of each added chunk.
9. **Latency-Aware Tradeoffs** – Progressive enrichment (fast path + optional deep fetch).
10. **Evaluation Feedback Loop** – Use success/failure signals to tune retrieval and compression.

---
## 10.4 Techniques & Patterns

### 10.4.1 Layered Context Envelope
```
[SYSTEM POLICY]
[HIGH-PRIORITY FACTS]
[USER INPUT / DELTA]
[RELEVANT RETRIEVED SNIPPETS]
[EXAMPLES / TESTS]
[OUTPUT SCHEMA INSTRUCTIONS]
```

### 10.4.2 Hybrid Retrieval Pipeline
1. Lexical (BM25) candidate set.
2. Dense embedding similarity.
3. Metadata filter (language=python, repo=payments, updated<30d).
4. Re-rank with cross-encoder.
5. Deduplicate + compress.

### 10.4.3 Semantic Chunking
- Boundary detection via AST for code, headings for docs.
- Overlap to preserve context windows (e.g., 50 token stride).
- Store embeddings per chunk variant (raw + summary).

### 10.4.4 Context Compression Methods
| Method | Use Case | Risk |
|--------|---------|------|
| Extractive summarization | Preserve key sentences | May omit glue logic |
| Abstractive summarization | Long narrative docs | Hallucinated paraphrase |
| Entity/Schema extraction | Config or logs | Loss of nuance |
| Instruction distillation | SOP manuals | Potential oversimplification |

### 10.4.5 Dynamic Budget Allocation
Pseudo-policy:
```
if task_type in {"bug_fix","refactor"}:
    allocate 50% tokens to code context
elif task_type == "architecture":
    allocate more examples & constraints
reserve 10% tokens for clarifying follow-ups
```

### 10.4.6 Conversational State Pruning
- Strategy: **Recency + Salience + Dependency Graph**.
- Keep turns with unresolved TODOs, referenced IDs, accepted decisions.
- Summarize stale narrative into compact state objects.

### 10.4.7 Retrieval Augmented Critique
Two-pass:
1. Generate draft.
2. Retrieve spec + style guide.
3. Critique & patch for compliance.

### 10.4.8 Tool-Gated Expansion
- Start minimal.
- If uncertainty markers detected ("not sure", low confidence pattern), auto-trigger deeper retrieval.

---
## 10.5 Basic vs Advanced Context Strategies

| Level | Characteristics | Example |
|-------|-----------------|---------|
| Basic | Static paste of docs | Paste README + ask question |
| Intermediate | Keyword retrieval + manual pruning | BM25 top 5 chunks |
| Advanced | Hybrid ranking + compression + budgeting | Dense + lexical + re-rank + summarization |
| Expert | Adaptive pipeline with feedback metrics | Closed-loop reinforcement adjusting weights |
| Enterprise | Governance, redaction, compliance, observability | ABAC filters + audit logs + PII scrub |

### 10.5.1 Zero-Shot Context Omission
When baseline knowledge suffices; omit external context to save cost and latency.

### 10.5.2 Few-Shot Context Exemplars
Select 2–4 curated examples representative of edge + base cases.

### 10.5.3 Instruction + Retrieval Fusion
Embed retrieval hints:
```
Use only the retrieved snippets below; cite snippet_id in reasoning.
```

---
## 10.6 Testing & Metrics for Context Quality

### 10.6.1 Retrieval Evaluation
- **Recall@k**: Relevant chunks present?
- **Precision@k**: Irrelevant noise minimized?
- **MRR / nDCG**: Ranking quality.
- **Overlap Penalty**: Duplicate or redundant chunks.

### 10.6.2 Context Utilization Metrics
Instrumentation:
```
utilized_tokens = sum(tokens_in_chunks_model_cited)
utilization_rate = utilized_tokens / total_context_tokens
```
(Approximate via citation detection, attention proxies, or heuristic pattern usage.)

### 10.6.3 Outcome Metrics
- Task success rate
- Hallucination incidence
- Time-to-resolution
- Edit distance between model output and accepted solution
- Escalation / clarification prompt count

### 10.6.4 Degradation & Drift Monitoring
- Version drift: context version vs production version mismatch count.
- Staleness: average age (days) of provided chunks.
- Security misses: flagged but unredacted secrets incidents.

### 10.6.5 Automated Harness Example
```python
class ContextTestHarness:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def evaluate_task(self, query, gold_answer, docs):
        ctx = self.retriever.retrieve(query)
        response = self.llm.generate(query, ctx)
        return {
            'recall': self.compute_recall(ctx, gold_answer, docs),
            'hallucination': self.detect_hallucination(response, docs),
            'latency_ms': self.measure_latency(query),
            'context_tokens': self.count_tokens(ctx),
            'utilization': self.estimate_utilization(response, ctx)
        }
```

---
## 10.7 Reusable Context Patterns

### 10.7.1 Compliance Envelope Pattern
```
[POLICY SUMMARY]
[ALLOWED APIS LIST]
[DATA CLASSIFICATION RULES]
[USER REQUEST]
``` 
Ensures outputs respect regulatory boundaries.

### 10.7.2 Code Region Focus Pattern
Retrieve only functions touched by a diff + their immediate dependencies.

### 10.7.3 Spec Anchoring Pattern
Place authoritative spec at top; forbid contradiction: "If spec conflicts with examples, spec wins."

### 10.7.4 Multi-View Fusion Pattern
Provide multiple modality views (raw log excerpt, parsed KPIs, summarized incident timeline) to reinforce reasoning.

### 10.7.5 Delta-Oriented Update Pattern
Only supply *changes since last step* plus persistent state summary to control growth.

### 10.7.6 Bidirectional Trace Pattern
Attach upstream requirement IDs + downstream test cases to maintain traceability.

### 10.7.7 Progressive Depth Pattern
Prompt instructs: "Ask for deeper context if current information insufficient." Implements demand-driven expansion.

---
## 10.8 Common Pitfalls & Remedies

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Overstuffing | Long, diluted responses | Apply minimal sufficient heuristic |
| Stale Context | Out-of-date suggestions | Add freshness TTL + invalidation hooks |
| Irrelevant Chatter | Model distracts | Improve ranking + semantic filters |
| Missing Critical Constraint | Violations / bugs | Maintain mandatory constraint preamble |
| Leakage of Sensitive Data | Compliance breach | Pre-ingestion PII/redaction pipeline |
| Order Instability | Non-deterministic answers | Deterministic sort (score, tie-break keys) |
| Duplicate Chunks | Waste tokens | Content hash + deduplicate |
| Unverifiable Claims | Hallucination | Enforce citation requirement with snippet IDs |

### 10.8.1 Anti-Hallucination Guardrails
- Snippet citation enforcement.
- Refusal pattern if confidence < threshold.
- Post-generation fact cross-check retrieval.

### 10.8.2 Latency Spikes
Mitigate via tiered caches (embedding cache, retrieval result cache, summary cache).

---
## 10.9 Integration with Development Workflows

### 10.9.1 IDE Assistance
- Retrieve local codebase symbols + docstrings.
- Diff-aware: limit to impacted modules.

### 10.9.2 CI/CD Gates
- PR description auto-generation uses only modified files + related tests.
- Security scan context: dependencies + advisories + config files.

### 10.9.3 Incident Response
Context bundle = recent logs (compressed), runbook excerpt, last deployment diff, current alerts metrics summary.

### 10.9.4 Documentation Generation
Supply architecture diagrams (serialized), config defaults, public interfaces to produce updated docs.

### 10.9.5 Test Authoring
Gather function signatures, behavior spec, edge case history from issue tracker.

---
## 10.10 Continuous Improvement & Optimization

### 10.10.1 Closed-Loop Tuning
1. Collect outcome + utilization metrics.
2. Identify low-yield chunks.
3. Retrain ranking (learning-to-rank).
4. Adjust chunk size & compression aggressiveness.
5. Re-measure drift and hallucination rate.

### 10.10.2 Adaptive Policies
- Increase compression when token pressure high.
- Elevate authority-weight of spec during regulation-critical periods.

### 10.10.3 Governance Dashboard KPIs
- Sensitive term leakage attempts blocked.
- Average context size vs success rate curve.
- Freshness distribution histogram.

### 10.10.4 Experimentation Framework
A/B: Baseline retrieval vs hybrid + compression; measure delta in acceptance rate.

### 10.10.5 Versioning Strategy
Semantic version context packs: `context-pack@payments-v2.1.3` ensuring reproducibility across audits.

---
## 10.11 Conclusion
Context engineering operationalizes *relevance* and *trustworthiness* in LLM interactions. By systematically selecting, ranking, compressing, and governing information, you transform a generic model into a task-aligned collaborator. The strongest results emerge when context engineering and prompt engineering co-evolve: precise instructions applied to high-signal, policy-compliant knowledge payloads.

**Key Takeaways:**
1. Optimize for relevance density, not raw volume.
2. Treat context assembly as a deterministic, testable pipeline.
3. Instrument everything: retrieval quality, utilization, drift, security.
4. Embrace adaptive, feedback-driven refinement.
5. Codify reusable patterns to accelerate safe scaling.

In the next chapter, we will deepen these foundations with advanced orchestration, governance, and optimization strategies that enable enterprise-grade context systems.
