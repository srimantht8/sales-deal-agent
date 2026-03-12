# Friction Log

Real issues encountered during development with LangChain, LangGraph, LangSmith, and related tools.

### [LANGCHAIN] langchain-community TavilySearchResults deprecated in v0.3.25
- **When:** Setting up web search tool
- **Issue:** `from langchain_community.tools import TavilySearchResults` emits a deprecation warning. The replacement is `langchain_tavily.TavilySearch` from the `langchain-tavily` package. This is a good migration but the deprecation timeline (removal in v1.0) isn't well-publicized.
- **Workaround:** Used `langchain-tavily` package directly with `from langchain_tavily import TavilySearch`.
- **Impact:** Low — caught during research phase. Would be confusing for someone following older tutorials.
- **Suggestion:** LangChain docs search results still prominently show the deprecated import path. Update SEO and code examples.

### [LANGCHAIN] HuggingFaceEmbeddings deprecated in langchain-community, migrated to langchain-huggingface
- **When:** Setting up local embeddings fallback (for running without cloud API keys)
- **Issue:** `from langchain_community.embeddings import HuggingFaceEmbeddings` emits `LangChainDeprecationWarning: The class HuggingFaceEmbeddings was deprecated in LangChain 0.2.2 and will be removed in 1.0.` The replacement is `from langchain_huggingface import HuggingFaceEmbeddings` via a separate `langchain-huggingface` package.
- **Workaround:** Added try/except import: try `langchain_huggingface` first, fall back to `langchain_community`. Also need `sentence-transformers` as a dependency.
- **Impact:** Low. Migration path is clear but requires an additional pip package. The deprecation warning formatting in the terminal is hard to read (includes `:class:` RST directives in the raw warning text).
- **Suggestion:** The deprecation warning text contains unrendered RST markup (`:class:\`~langchain-huggingface`). Should be plain text. Also, consider bundling `langchain-huggingface` as a dependency of `langchain-community` to smooth the migration.

### [LANGCHAIN] duckduckgo-search package renamed to ddgs
- **When:** Testing DuckDuckGo fallback search
- **Issue:** The `duckduckgo-search` package (used by `DuckDuckGoSearchResults` in langchain-community) has been renamed to `ddgs`. Importing from `duckduckgo_search` emits `RuntimeWarning: This package (duckduckgo_search) has been renamed to ddgs! Use pip install ddgs instead.` and returns zero results.
- **Workaround:** Added try/except import: try `from ddgs import DDGS` first, fall back to `from duckduckgo_search import DDGS`. Also added `ddgs` as a dependency.
- **Impact:** ~5 minutes. The deprecation is recent and langchain-community still references the old package name.
- **Suggestion:** LangChain community tools should update the DuckDuckGo integration to use the new `ddgs` package name. The old package appears to return empty results.

### [BEDROCK] Direct model IDs no longer work for on-demand inference
- **When:** Running `make demo` — first LLM call (intake node) via ChatBedrockConverse
- **Issue:** `botocore.errorfactory.ValidationException: Invocation of model ID anthropic.claude-3-5-sonnet-20241022-v2:0 with on-demand throughput isn't supported. Retry your request with the ID or ARN of an inference profile that contains this model.` Bedrock now requires cross-region inference profile IDs (e.g., `us.anthropic.claude-3-5-sonnet-20241022-v2:0`) instead of raw model IDs for on-demand usage.
- **Workaround:** Changed default `bedrock_model_id` from `anthropic.claude-3-5-sonnet-20241022-v2:0` to `us.anthropic.claude-3-5-sonnet-20241022-v2:0` (cross-region inference profile).
- **Impact:** ~3 minutes. The error message is clear about what to do but doesn't tell you the correct inference profile ID format.
- **Suggestion:** `langchain-aws` should document the inference profile requirement prominently, or auto-map model IDs to their inference profile equivalents. Most Bedrock tutorials still show raw model IDs.

### [BEDROCK] Haiku returns list fields as JSON strings in structured output
- **When:** Running synthesis node with `with_structured_output(SynthesisResult)` via Claude Haiku on Bedrock
- **Issue:** Bedrock's Converse API with Haiku returns `list[str]` fields (talking_points, risks, opportunities) as JSON-encoded strings (e.g., `'["point 1", "point 2"];\n'`) instead of actual lists. This causes Pydantic `ValidationError: Input should be a valid list`. Sonnet does not exhibit this behavior. The string includes trailing semicolons and whitespace.
- **Workaround:** Added a `@field_validator` with `mode="before"` on the Pydantic model to detect string inputs, strip trailing semicolons, and `json.loads()` them back into lists.
- **Impact:** ~10 minutes with direct inspection of the raw Converse API response. A developer who doesn't think to check the raw payload — especially someone following a quickstart that only tests with Sonnet — could spend 30+ minutes chasing a Pydantic validation error that looks like a schema bug rather than a model-level type coercion issue.
- **Suggestion:** `langchain-aws` should normalize tool call argument types before passing to Pydantic, or at minimum document this Haiku behavior. Alternatively, Bedrock's Converse API should enforce the tool schema types server-side.

### [LANGCHAIN] TavilySearch.invoke() return type changed from list to dict
- **When:** Testing web search tool and displaying results in Streamlit UI
- **Issue:** `TavilySearch.invoke()` from `langchain-tavily` now returns a dict `{"query": ..., "results": [...], ...}` instead of a flat list of result dicts. Code expecting a list (e.g., iterating or slicing `results[:5]`) fails with `TypeError: unhashable type: 'slice'`. The old `TavilySearchResults` from langchain-community returned a list.
- **Workaround:** Added type checking in `_search_tavily()`: if result is a dict, unwrap the nested `results` key.
- **Impact:** ~5 minutes once I found it. The real risk is for someone migrating from `TavilySearchResults` to `TavilySearch` following the deprecation notice — the migration guide covers the import path change but not the return type change, so code that worked before silently breaks at runtime with a non-obvious `TypeError`.
- **Suggestion:** `langchain-tavily` should document the return type change from the deprecated `TavilySearchResults`. Migration guides should highlight API differences, not just import path changes.

### [BEDROCK] Haiku produces malformed JSON in LLM-as-judge evaluator
- **When:** Running `make eval` — the `output_quality_judge` evaluator asks Haiku to return a JSON object `{"score": 0.X, "feedback": "..."}`
- **Issue:** For longer outputs (e.g., the Meridian Health follow-up email), Haiku embeds unescaped quotes and commas inside the `feedback` string value, producing invalid JSON. `json.loads()` fails with `Expecting ',' delimiter`. Sonnet handles this reliably. The existing markdown-fence stripping (````json ... ```) isn't sufficient — the JSON itself is structurally broken.
- **Workaround:** The evaluator's try/except falls back to a default score of 0.5. Acceptable for a demo but means one metric is unreliable on Haiku.
- **Impact:** ~2 minutes to diagnose from eval logs. Low immediate impact (graceful fallback), but undermines evaluation reliability.
- **Suggestion:** For LLM-as-judge patterns with smaller models, `langchain-aws` or LangSmith docs should recommend using `with_structured_output()` instead of free-form JSON prompting. Alternatively, LangSmith's built-in evaluator SDK could handle this pattern natively.
- **Resolution:** Replaced free-form JSON prompting with `with_structured_output(JudgeResult)`, which uses tool calling to enforce the Pydantic schema. This bypasses the JSON serialization issue entirely — the model never needs to produce raw JSON text. The try/except fallback to 0.5 remains as a safety net for other failure modes (API errors, timeouts).

### [LANGGRAPH] Send() belongs in conditional edges, not nodes — error message doesn't explain the distinction
- **When:** Building the router node. Initial attempt returned `list[Send]` directly from `router_node()` and also passed it to `add_conditional_edges()`.
- **Issue:** Returning `list[Send]` from a node function produces `InvalidUpdateError: Expected dict, got [Send(node='run_web_research', arg={...})]`. The error tells you the return type is wrong but doesn't explain that `Send()` objects must be returned from conditional edge functions, not node functions. A developer unfamiliar with the node-vs-edge distinction will search for "Send() return type" rather than understanding the architectural separation.
- **Workaround:** Split the router into two functions: `router_node()` returns a dict of routing decisions (`router.py:27`), and `route_to_research()` returns `list[Send]` for `add_conditional_edges()` (`router.py:54`). This is the correct pattern but wasn't obvious from the error alone.
- **Impact:** ~5 minutes. The fix is simple once you understand the distinction, but the error message sends you down the wrong debugging path.
- **Suggestion:** The `InvalidUpdateError` when a node returns `Send()` should include: "Did you mean to use Send() in a conditional edge? Node functions must return dict updates. Use `add_conditional_edges()` with a function that returns `list[Send]`." This would save developers from re-reading the full Send() guide to understand the node/edge boundary.

### [LANGGRAPH] Invoking an interrupted graph without Command(resume=) silently re-runs from START
- **When:** Implementing human-in-the-loop review. After the graph paused at `interrupt()` in `human_review.py:39`, my first resume attempt was `graph.invoke(None, config)`.
- **Issue:** Calling `invoke()` with a regular input (or `None`) on a thread with a pending interrupt silently re-runs the entire graph from START. No error, no warning, no hint that `Command(resume=value)` is the correct pattern. With list-based state fields, the re-run *appends* duplicate results (research data doubled). The developer sees the same `__interrupt__` output again and has no signal about what went wrong. Tested on `langgraph==0.2.60`.
- **Workaround:** Use `graph.invoke(Command(resume=value), config)` and check `graph.get_state(config).next` to detect paused state before attempting resume (`demo.py:44-48`, `ui/app.py:153-157`).
- **Impact:** ~15 minutes. The correct pattern is clean once discovered, but the silent re-run behavior is dangerous — in production, this could trigger duplicate API calls, duplicate emails, or duplicate tool executions with no error signal.
- **Suggestion:** When `invoke()` is called on a thread with a pending interrupt, either: (a) raise an error: `"Thread {thread_id} has a pending interrupt at node '{node_name}'. Use Command(resume=value) to continue, or pass force_restart=True to re-run from START."`, or (b) emit a warning log. The current silent behavior violates the principle of least surprise.

### [LANGSMITH] @traceable on LangGraph nodes creates duplicate trace entries
- **When:** Adding `@traceable` decorators to node functions (`intake.py:42`, `synthesis.py:60`, `generate.py:54`) for explicit tracing control.
- **Issue:** LangGraph automatically traces every node execution as a child span of the root "LangGraph" run. Adding `@traceable` to a node function creates a *second*, separate child span for the same execution. The 3 decorated nodes each appear twice in the trace — once from LangGraph's built-in tracing and once from `@traceable`. This isn't documented: the `@traceable` docs show it being used on functions, and the LangGraph docs mention built-in tracing, but neither warns about the interaction.
- **Workaround:** Accepted the duplication for the 3 key nodes where explicit `run_type="chain"` metadata was useful. Did not decorate the other 6 nodes. The ideal pattern would be to use `@traceable` only on sub-functions *called within* nodes, not on the node functions themselves.
- **Impact:** ~5 minutes to notice and understand the duplication in the LangSmith UI. Not blocking, but makes traces harder to read — especially for a 9-node graph where 3 nodes show duplicate entries.
- **Suggestion:** Either: (a) detect when `@traceable` is applied to a function that LangGraph will also trace, and deduplicate automatically, or (b) add a warning in the `@traceable` docs: "Do not use `@traceable` on LangGraph node functions — LangGraph provides built-in tracing. Use `@traceable` on sub-functions called within nodes for additional granularity."

### [LANGSMITH] evaluate() has no built-in support for interrupt/resume graphs
- **When:** Writing the evaluation target function in `eval/run_eval.py` to wrap the agent graph (which uses `interrupt()` for human review).
- **Issue:** `langsmith.evaluation.evaluate()` calls the target function once per dataset example and expects a dict back. When the graph hits `interrupt()`, the target returns incomplete state (missing `final_output`, `generated_output`). The developer must manually: (1) detect the interrupt via `get_state().next`, (2) auto-approve with `Command(resume="approve")`, (3) return the final state. This is ~10 lines of boilerplate per target function — and getting it wrong silently produces bad eval results (e.g., evaluating the interrupted state instead of the final output).
- **Workaround:** Handle the interrupt/resume cycle manually inside the target function. Generate unique thread IDs per eval run (`uuid.uuid4()`) to avoid state leaking across runs via the checkpointer. The `ls_*` wrapper functions in `evaluators.py:117-144` adapt the evaluator signatures.
- **Impact:** ~8 minutes. First attempt used `invoke(None, config)` instead of `Command(resume=)` (see interrupt re-run issue above), which produced duplicate research results in the eval output.
- **Suggestion:** Provide a built-in helper: `from langgraph.evaluation import make_eval_target` that wraps a compiled graph with auto-interrupt-handling and unique thread ID generation. Or add an `auto_resume` parameter to `evaluate()` for graphs with checkpointers: `evaluate(target, data, evaluators, auto_resume="approve")`.
