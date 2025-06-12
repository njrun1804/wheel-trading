# Claude Code Orchestrator — Implementation Brief

---

### 1 · Mission (what success looks like)

* **Input**: One natural-language command, e.g.
  `"Refactor sizing, test at scale, post PR"`
* **Output**: A merged PR created only if:

  * All tests pass and post-deploy checks are clean.
  * End-to-end wall-clock ≤ 90 s on a cold start.
  * Peak RAM ≤ 70 % (≈ 16 GB on Mac M4 Pro, 24 GB).
* **LLM guard-rails**: ≤ 4 k tokens per step, ≤ 4096 output tokens.

---

### 2 · Current assets (always-on ✓)

| Tool tier          | MCP servers                                       |
| ------------------ | ------------------------------------------------- |
| Code / search      | ✓ `filesystem`, ✓ `ripgrep`, ✓ `dependency_graph` |
| Reasoning / memory | ✓ `memory`, ✓ `sequential-thinking`               |
| Domain logic       | ✓ `python_analysis`                               |
| Observability      | ✓ `trace_phoenix`                                 |

On-demand: `duckdb`, `statsource`, `brave`, `puppeteer`, `mlflow`.

Environment flags already exported:

```
CLAUDE_CODE_THINKING_BUDGET_TOKENS=4096
CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096
CLAUDE_CODE_PARALLELISM=8
```

---

### 3 · Execution flow (7 deterministic phases)

1. **Map** `ripgrep` → `dependency_graph` → `memory` → `candidate_slices`
2. **Logic** `dependency_graph.call_graph` → pruned slices
3. **Monte Carlo** `python_analysis.monte_carlo()` 15 s cap
4. **Plan** LLM → `plan.json` (JSON DAG, pools: **scan** / **mutate**)
5. **Optimise** `duckdb` / `pyrepl` → top-k param grid
6. **Execute** refactor → write → `test_runner.run`
7. **Review / Loop** `trace_phoenix` & post-deploy check (≤ 3 retries)

---

### 4 · Immediate tasks for Claude Code (branch `orchestrator_bootstrap`)

1. **`orchestrator.py`** — prompt → result; validate / load `plan.json`.
2. **`slice_cache.py`** — SHA-1-keyed cache table: `slice_cache(hash PK, vector BLOB)`.
3. **`pressure.py`** — background gauge; publish `mem_ratio` (RSS/total) every 250 ms.

*Stage code, run tests; pause for human review once green.*

---

### 5 · Acceptance checklist

* Cold-start end-to-end ≤ 90 s on 3 M LOC repo.
* RSS never > 70 % during Monte-Carlo.
* Failing tests or trace errors trigger auto-retry (≤ 3).
* PR contains code diff, `plan.json`, MC summary, Phoenix trace link.