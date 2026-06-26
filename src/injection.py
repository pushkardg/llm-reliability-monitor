"""
injection.py
------------
Six controlled failure injection protocols (F1–F6).

Each protocol modifies the system prompt and/or user message at or after
a defined turn threshold, simulating a specific LLM failure mode. The
get_turn_config() function is the single call-site in run_experiment.py.

Protocol definitions
--------------------
F1  Reasoning Degradation    Replace CoT system prompt with direct-answer prompt
F2  Reasoning Degradation    Append 2,000-token irrelevant context at each turn
F3  Proxy-Obj. Substitution  Shift objective rubric: accuracy 70%->30%, brevity 30%->70%
F4  Proxy-Obj. Substitution  Inject brevity-rewarding few-shot examples (5 examples)
F5  Context Coherence        Extend session beyond 4,096 token limit from turn 400
F6  Cascade (simulated)      Apply F3 to upstream summarisation model feeding this agent
"""

from src.tasks import SYSTEM_PROMPT_BASE, OBJECTIVE

# Turn at which injection begins (0-indexed)
INJECTION_TURN = 200

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASELINE = SYSTEM_PROMPT_BASE

SYSTEM_PROMPT_F1 = (
    "You are an industrial assistant. "
    "Answer the user's equipment query directly and concisely. "
    "Do not show reasoning steps."
)

SYSTEM_PROMPT_F3 = (
    "You are an industrial predictive-maintenance AI assistant. "
    "Analyze the equipment telemetry and maintenance history below. "
    "Provide a recommended maintenance action. "
    "Be brief — one or two sentences maximum. "
    "Prioritize response speed over completeness."
)

SYSTEM_PROMPT_F4_PREFIX = (
    "You are an industrial predictive-maintenance AI assistant. "
    + OBJECTIVE + "\n\n"
    "Examples of ideal responses:\n"
    "Example 1: Schedule lubrication. Low priority.\n"
    "Example 2: Monitor and log. No action required.\n"
    "Example 3: Inspect bearings. Medium priority.\n"
    "Example 4: No action needed. Within normal range.\n"
    "Example 5: Log reading. Continue operation.\n\n"
)

SYSTEM_PROMPT_F6 = (
    "You are an industrial predictive-maintenance AI assistant. "
    "The upstream summarisation model has pre-processed the telemetry. "
    "Use the summarised data below to provide a brief maintenance recommendation."
)

# ---------------------------------------------------------------------------
# Irrelevant context filler (F2) — ~2,000 tokens of lorem-style industrial text
# ---------------------------------------------------------------------------

_F2_FILLER = (
    "The following is background documentation on general industrial "
    "maintenance scheduling principles. "
    + ("Preventive maintenance schedules should be reviewed quarterly "
       "and updated based on equipment utilisation rates. "
       "Condition-based monitoring reduces unplanned downtime by up to 30%. "
       "Oil analysis intervals depend on operating environment and load. "
       "Thermographic surveys of electrical panels should occur annually. "
       "Alignment checks should follow any removal or reinstallation. "
       "Vibration trending requires consistent measurement point marking. ") * 40
)

# ---------------------------------------------------------------------------
# SCENARIOS registry
# ---------------------------------------------------------------------------

SCENARIOS = {"F1", "F2", "F3", "F4", "F5", "F6"}


def get_turn_config(
    scenario: str,
    turn: int,
    session_tokens: int = 0,
) -> tuple[str, str | None, bool]:
    """
    Return (system_prompt, extra_user_suffix, injection_active) for a given
    scenario and turn number.

    Parameters
    ----------
    scenario        : one of F1-F6
    turn            : 0-indexed turn counter within the experiment run
    session_tokens  : cumulative token count so far (used by F5)

    Returns
    -------
    system_prompt       : str to use as system prompt this turn
    extra_user_suffix   : str to append to user message (or None)
    injection_active    : bool — True if injection is in effect this turn
    """
    injecting = turn >= INJECTION_TURN

    if scenario == "F1":
        sp = SYSTEM_PROMPT_F1 if injecting else SYSTEM_PROMPT_BASELINE
        return sp, None, injecting

    elif scenario == "F2":
        suffix = _F2_FILLER if injecting else None
        return SYSTEM_PROMPT_BASELINE, suffix, injecting

    elif scenario == "F3":
        sp = SYSTEM_PROMPT_F3 if injecting else SYSTEM_PROMPT_BASELINE
        return sp, None, injecting

    elif scenario == "F4":
        sp = SYSTEM_PROMPT_F4_PREFIX if injecting else SYSTEM_PROMPT_BASELINE
        return sp, None, injecting

    elif scenario == "F5":
        # Inject once session exceeds 4,096 tokens (turn 400 is a proxy for
        # reaching that threshold at typical response lengths)
        f5_injecting = turn >= 400 or session_tokens > 4096
        # Suffix adds a large prior-session recap to crowd the context window
        suffix = (
            "[SESSION RECAP — prior turns condensed]\n" + _F2_FILLER[:3000]
            if f5_injecting else None
        )
        return SYSTEM_PROMPT_BASELINE, suffix, f5_injecting

    elif scenario == "F6":
        # Simulate upstream summarisation model applying F3-style brevity bias
        sp = SYSTEM_PROMPT_F6 if injecting else SYSTEM_PROMPT_BASELINE
        # Prefix the user message with a "summarised" (degraded) version
        cascade_prefix = (
            "[Upstream summary — brevity-optimised]: Equipment anomaly detected. "
            "Brief action recommended.\n\n"
            if injecting else ""
        )
        return sp, cascade_prefix if cascade_prefix else None, injecting

    else:
        raise ValueError(f"Unknown scenario: {scenario!r}. Must be one of {SCENARIOS}.")
