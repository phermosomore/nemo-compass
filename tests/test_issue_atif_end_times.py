# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Issue repro: cost-efficiency evidence fills an end time earlier than the recorded start.

NeMo Relay ATIF stamps each agent step with the moment the model call ENDED (microseconds) and
records the call's own window in ``extra.invocation`` as epoch seconds truncated to milliseconds.
The ATIF loader maps the step timestamp to ``Span.start_time``. Any extractor that faithfully reads
``extra.invocation.end_timestamp`` then yields an ``end_time`` up to 1 ms BEFORE ``start_time``.
``enrich_trace`` assigns it without validation; the private snapshot re-validates each trace on
read, so the signal raises ``ValidationError`` and the CLI's TaskGroup aborts the whole run.

The generated extractor is Nooa/LLM-written Python; these tests stand in a fixed, faithful
extractor at the same seam the existing tests use (``extract_evidence`` -> ``_build_snapshot``).
"""

import asyncio
import json
from functools import partial

from nooa.unifiedllm import FakeLLMClient

from compass.compass_signals.builtins import registered_builtin_signals
from compass.compass_signals.cost_efficiency import signal
from compass.compass_signals.cost_efficiency.evidence import CacheEvidence, enrich_trace
from compass.compass_signals.cost_efficiency.signal import CostEfficiencyConfig
from compass.field_extraction import _build_snapshot
from compass.trace_loaders import ATIFTraceConfig, ATIFTraceLoader

# Values copied from the first step of a real NeMo Relay ATIF trajectory: the step timestamp is
# the invocation's end time at microsecond precision; end_timestamp is the same instant in ms.
RELAY_STEP_TIMESTAMP = "2026-11-10T01:30:28.984285+00:00"
RELAY_INVOCATION = {
    "start_timestamp": 1794274227.658,
    "end_timestamp": 1794274228.984,
    "framework": "nemo_relay",
    "status": "completed",
}


def relay_trajectory() -> dict:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "s",
        "trajectory_id": "t",
        "agent": {"name": "cad-agent", "version": "0", "model_name": "m"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "build it"},
            {
                "step_id": 2,
                "source": "agent",
                "timestamp": RELAY_STEP_TIMESTAMP,
                "message": "done",
                "llm_call_count": 1,
                "metrics": {"prompt_tokens": 100, "completion_tokens": 10},
                "extra": {"invocation": RELAY_INVOCATION},
            },
        ],
    }


def faithful_extractor(trace: dict, span: dict) -> dict | None:
    """What a correct generated extractor reports: the recorded invocation window."""
    if span.get("kind") != "LLM":
        return None
    step = next(
        turn
        for root in trace["root_spans"]
        for turn in root.get("turns", [])
        if span["id"] in turn.get("llm_span_ids", [])
    )
    invocation = step["extra"]["invocation"]
    return {
        "kind": "LLM",
        "start_time": invocation["start_timestamp"],
        "end_time": invocation["end_timestamp"],
    }


def load(tmp_path):
    path = tmp_path / "relay.jsonl"
    path.write_text(json.dumps(relay_trajectory()) + "\n")
    return ATIFTraceLoader(ATIFTraceConfig(path=path)).load()


def test_loader_keeps_step_timestamp_as_start_and_no_end(tmp_path):
    """Precondition (passes on main): the loader itself produces a valid span."""
    (trace,) = load(tmp_path)
    step = trace.root_spans[0].children[0]
    assert step.id == "trajectory/step/2"
    assert step.start_time is not None and step.end_time is None


def test_enriched_view_never_has_a_span_ending_before_it_starts(tmp_path):
    """Lowest deterministic layer: _build_snapshot + enrich_trace, then read the view."""
    view = _build_snapshot(load(tmp_path), CacheEvidence, enrich_trace, faithful_extractor)
    (trace,) = list(view)  # main: pydantic ValidationError "span ... ends before it starts"
    step = trace.root_spans[0].children[0]
    assert step.end_time is None or step.end_time >= step.start_time


def test_cost_efficiency_signal_does_not_raise_on_relay_atif(tmp_path, monkeypatch):
    """Public path: ATIF loader -> registry -> cost-efficiency signal (LLM extraction stubbed)."""

    async def extract(snapshot, schema, instructions, llm, project):
        return partial(_build_snapshot, snapshot, schema, project)(faithful_extractor)

    monkeypatch.setattr(signal, "extract_evidence", extract)
    registry = registered_builtin_signals(
        cost_efficiency=CostEfficiencyConfig(), llm_factory=FakeLLMClient
    )
    result = asyncio.run(registry.analyze("cost-efficiency", load(tmp_path)))
    assert result.signal_name == "cost-efficiency"
