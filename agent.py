"""
FL Orchestrator Agent — calls Claude API to make per-round decisions.

Decisions:
  - Which clients to include in the next training round
  - Whether to stop training early
  - Optional learning rate adjustment
"""

import json
import os
import httpx

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are an autonomous orchestrator agent for a federated learning system.
Each round, you receive metrics from all clients and the global model.
You must decide:
1. Which client IDs to SELECT for the next training round (pick at least 2).
2. Whether to STOP training early (if accuracy has plateaued or is decreasing).
3. A learning rate adjustment if needed (suggest a float, or "keep" to stay the same).

Respond ONLY with valid JSON, no markdown, no explanation:
{
  "selected_clients": [0, 1, 3],
  "stop_early": false,
  "learning_rate": "keep",
  "reasoning": "short explanation of your decision"
}

Selection guidelines:
- Prefer clients with LOWER loss (they have better local data fit).
- Exclude clients that are very slow (high training time) unless their data is critical.
- If a client's loss is much higher than others, it may have noisy/poor data — consider excluding.
- Include at least 2 clients per round for meaningful aggregation.
- If global accuracy has not improved for 2+ rounds, consider stopping early.
"""


def call_agent(round_num: int, total_rounds: int, client_metrics: list[dict],
               global_metrics: dict, history: list[dict]) -> dict:
    """
    Call the LLM agent with current FL state and get back orchestration decisions.

    Args:
        round_num: Current round number (1-indexed)
        total_rounds: Max rounds configured
        client_metrics: List of dicts with per-client info from last round
            e.g. [{"client_id": 0, "train_loss": 2.1, "num_examples": 4000}, ...]
        global_metrics: Dict with global eval results
            e.g. {"eval_loss": 2.3, "eval_acc": 0.10}
        history: List of past round summaries for context

    Returns:
        Dict with agent decisions
    """
    user_message = f"""
Round {round_num} of {total_rounds} just completed.

## Per-Client Metrics (from this round's training):
{json.dumps(client_metrics, indent=2)}

## Global Model Evaluation:
{json.dumps(global_metrics, indent=2)}

## History of Past Rounds:
{json.dumps(history, indent=2)}

Available client IDs: {list(range(len(client_metrics)))}

What should I do for the next round?
"""

    # If no API key, fall back to a simple heuristic
    if not ANTHROPIC_API_KEY:
        return _fallback_heuristic(round_num, total_rounds, client_metrics, global_metrics, history)

    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": MODEL,
                "max_tokens": 500,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_message}],
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        text = data["content"][0]["text"]
        # Strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        decision = json.loads(text)
        return decision
    except Exception as e:
        print(f"[Agent] LLM call failed ({e}), using fallback heuristic")
        return _fallback_heuristic(round_num, total_rounds, client_metrics, global_metrics, history)


def _fallback_heuristic(round_num, total_rounds, client_metrics, global_metrics, history):
    """Simple rule-based fallback if the LLM is unavailable."""
    # Sort clients by loss (ascending = best first), pick top 60%
    sorted_clients = sorted(client_metrics, key=lambda c: c.get("train_loss") if c.get("train_loss") is not None else 999)
    n_select = max(2, int(len(sorted_clients) * 0.6))
    selected = [c["client_id"] for c in sorted_clients[:n_select]]

    # Check for plateau
    stop = False
    if len(history) >= 3:
        recent_accs = [h.get("eval_acc", 0) for h in history[-3:]]
        if max(recent_accs) - min(recent_accs) < 0.005:
            stop = True

    return {
        "selected_clients": selected,
        "stop_early": stop,
        "learning_rate": "keep",
        "reasoning": f"Fallback heuristic: selected {n_select} clients with lowest loss",
    }
