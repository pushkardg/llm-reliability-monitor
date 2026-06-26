"""
baselines.py
------------
Four standard monitoring baselines compared against the SDIs in the paper:

  1. Request error rate  (HTTP 5xx / exception rate)
  2. P95 inference latency
  3. BERTScore           (against a fixed reference set)
  4. ADWIN               (adaptive windowing applied to rolling BERTScore)

None of these require the LLM to be running at metric-compute time;
they consume per-turn records written by run_experiment.py.
"""

import statistics
import math
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# 1. Error rate
# ---------------------------------------------------------------------------

def compute_error_rate(records: list[dict], window: int = 100) -> float:
    """
    Fraction of the last `window` records that have error=True.
    Returns 0.0 if there are no records in the window.
    """
    recent = records[-window:]
    if not recent:
        return 0.0
    return sum(1 for r in recent if r.get("error", False)) / len(recent)


# ---------------------------------------------------------------------------
# 2. P95 latency
# ---------------------------------------------------------------------------

def compute_p95_latency(records: list[dict], window: int = 100) -> float:
    """
    95th-percentile latency (seconds) over the last `window` records.
    Records must contain a "latency_s" key.
    """
    recent = records[-window:]
    latencies = [r["latency_s"] for r in recent if "latency_s" in r]
    if not latencies:
        return 0.0
    latencies.sort()
    idx = max(0, int(math.ceil(0.95 * len(latencies))) - 1)
    return latencies[idx]


# ---------------------------------------------------------------------------
# 3. BERTScore
# ---------------------------------------------------------------------------

_bertscore_scorer = None


def _get_bertscore():
    global _bertscore_scorer
    if _bertscore_scorer is None:
        from bert_score import BERTScorer
        _bertscore_scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            rescale_with_baseline=True,
        )
    return _bertscore_scorer


def compute_bertscore_f1(
    current_outputs: list[str],
    reference_outputs: list[str],
) -> float:
    """
    Mean BERTScore F1 between current_outputs and reference_outputs.
    Both lists should be the same length; if lengths differ, the shorter
    list is repeated to match.
    """
    scorer = _get_bertscore()
    # Align lengths
    n = max(len(current_outputs), len(reference_outputs))
    cands = (current_outputs * math.ceil(n / max(len(current_outputs), 1)))[:n]
    refs  = (reference_outputs * math.ceil(n / max(len(reference_outputs), 1)))[:n]

    _, _, F1 = scorer.score(cands, refs)
    return float(F1.mean().item())


# ---------------------------------------------------------------------------
# 4. ADWIN (Adaptive Windowing)
# ---------------------------------------------------------------------------

class ADWIN:
    """
    Simplified ADWIN change-detection algorithm.

    Reference: Bifet & Gavaldà, "Learning from Time-Changing Data with
    Adaptive Windowing," SDM 2007.

    Keeps a growing window of values and fires when a statistically
    significant mean shift is detected between a recent sub-window and
    the rest of the window.

    Usage:
        adwin = ADWIN(delta=0.002)
        for value in stream:
            if adwin.add(value):
                print("Change detected!")
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._window: deque[float] = deque()
        self._total = 0.0
        self._n = 0

    def add(self, value: float) -> bool:
        """
        Add one observation. Returns True if a change is detected.
        """
        self._window.append(value)
        self._total += value
        self._n += 1
        return self._detect()

    def _detect(self) -> bool:
        """
        Slide a cut-point through the window and test whether the two
        sub-windows have significantly different means.
        """
        if self._n < 10:
            return False

        values = list(self._window)
        total = self._total
        n = self._n

        right_sum = 0.0
        for cut in range(1, n):
            right_sum += values[n - cut]
            left_n  = n - cut
            right_n = cut
            left_sum = total - right_sum

            mu_left  = left_sum  / left_n
            mu_right = right_sum / right_n
            mu_total = total / n

            # Variance bound from Hoeffding's inequality (values in [0,1])
            epsilon_cut = math.sqrt(
                (1.0 / (2 * left_n) + 1.0 / (2 * right_n))
                * math.log(4 * n / self.delta)
            )

            if abs(mu_left - mu_right) >= epsilon_cut:
                # Drop everything before the cut
                for _ in range(left_n):
                    dropped = self._window.popleft()
                    self._total -= dropped
                    self._n -= 1
                return True

        return False

    def reset(self):
        self._window.clear()
        self._total = 0.0
        self._n = 0
