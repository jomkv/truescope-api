"""
Concurrent WebSocket load test for TrueScope verify endpoint.

What this test does:
1) Runs N concurrent WS claim verifications.
2) While each WS stream is running, it randomly calls /v1/verify/calculate
   using random subsets of currently received WS results.
3) Prints per-worker and overall metrics.
4) Saves both detailed results and summary metrics to JSON.

Usage:
    pip install websockets httpx
    python prod_test/load_test.py
"""

import asyncio
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# URL config (env-driven so no production domain is committed)
# ---------------------------------------------------------------------------

load_dotenv()

PROD_URL = str(os.getenv("PROD_URL")).strip().rstrip("/")


def _to_ws_base(prod_url: str) -> str:
    if prod_url.startswith("https://"):
        return "wss://" + prod_url[len("https://") :]
    if prod_url.startswith("http://"):
        return "ws://" + prod_url[len("http://") :]
    raise ValueError(
        "PROD_URL must start with http:// or https:// " f"(got: {prod_url!r})"
    )


WS_URL = f"{_to_ws_base(PROD_URL)}/v1/verify/ws"
CALCULATE_URL = f"{PROD_URL}/v1/verify/calculate"

LOG_RESULTS = True
LOAD_TEST_RESULTS_DIR = Path("tests/production/results")

# Random calculate behavior per worker
CALC_CALLS_MIN = 0
CALC_CALLS_MAX = 5
CALC_TRIGGER_PROB = 0.35
MAX_CALC_SAMPLE_SIZE = 8

# Timeouts
WS_RECV_TIMEOUT_S = 180
WS_OPEN_TIMEOUT_S = 30
WS_PING_TIMEOUT_S = 180
HTTP_TIMEOUT_S = 60
LIVE_LOGS = True

# Spread out claims so requests are not identical
TEST_CLAIMS = [
    "Donald Trump is against marriage equality.",  # T
    "Virginia ranks “in the bottom third of states” in administering the COVID-19 vaccine.",  # T
    "“One in 5 Americans has lost a family member to gun violence.”",  # T
    "Donald Trump wants to terminate the Affordable Care Act.",  # T
    "Project 2025 would defund K-12 schools.",  # T
    "Switzerland banned mammograms.",  # F
    "Filipino actor and singer Billy Crawford has died.",  # F
    "President Ferdinand Marcos Jr. stole the country’s gold reserves",  # F
    "The radiation from Bluetooth earbuds, such as AirPods, causes brain cancer.",  # F
    "“Elon Musk launches new pregnancy robot.”",  # F
]

CONCURRENCY = len(TEST_CLAIMS)


def _log(message: str):
    if not LIVE_LOGS:
        return
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")


async def call_calculate(
    client, evidences: list[dict], worker_id: int, call_id: int
) -> dict:
    """Call /calculate and capture latency + status."""
    t0 = time.perf_counter()
    _log(f"[W{worker_id:02d}] recalc #{call_id} started (evidence={len(evidences)})")
    try:
        response = await client.post(CALCULATE_URL, json=evidences)
        response.raise_for_status()
        payload = response.json()
        result = {
            "ok": True,
            "duration_s": round(time.perf_counter() - t0, 3),
            "overall_verdict": payload.get("overall_verdict"),
            "truth_confidence_score": payload.get("truth_confidence_score"),
            "bias_divergence": payload.get("bias_divergence"),
            "bias_consistency": payload.get("bias_consistency"),
        }
        _log(
            f"[W{worker_id:02d}] recalc #{call_id} done "
            f"({result['duration_s']}s, ok)"
        )
        return result
    except Exception as exc:
        result = {
            "ok": False,
            "duration_s": round(time.perf_counter() - t0, 3),
            "error": str(exc),
        }
        _log(
            f"[W{worker_id:02d}] recalc #{call_id} done "
            f"({result['duration_s']}s, error={result['error']})"
        )
        return result


async def ws_worker(worker_id: int, claim: str) -> dict:
    try:
        import websockets
        import httpx
    except ImportError:
        raise RuntimeError("Run: pip install websockets httpx")

    result = {
        "worker_id": worker_id,
        "claim": claim,
        "mode": "websocket",
        "started_at": None,
        "finished_at": None,
        "duration_s": None,
        "error": None,
        "events_received": 0,
        "event_type_counts": {},
        "overall_verdict": None,
        "truth_confidence": None,
        "total_results": None,
        "stats_payload": None,
        "aggregated_doc_ids": [],
        "used": [],
        # calculate metrics
        "calculate_calls_target": 0,
        "calculate_calls_sent": 0,
        "calculate_calls_ok": 0,
        "calculate_calls_failed": 0,
        "calculate_avg_duration_s": None,
        "calculate_samples": [],
    }

    t_start = time.perf_counter()
    result["started_at"] = datetime.now().isoformat()
    _log(f"[W{worker_id:02d}] worker started")

    ws_results: list[dict] = []
    calc_tasks: list[asyncio.Task] = []
    calc_call_id = 0

    # Decide random amount of calculate calls for this worker
    calc_target = random.randint(CALC_CALLS_MIN, CALC_CALLS_MAX)
    result["calculate_calls_target"] = calc_target

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as http_client:
            async with websockets.connect(
                WS_URL,
                ping_timeout=WS_PING_TIMEOUT_S,
                open_timeout=WS_OPEN_TIMEOUT_S,
            ) as ws:
                await ws.send(json.dumps({"claim": claim}))

                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT_S)
                    msg = json.loads(raw)

                    result["events_received"] += 1
                    msg_type = msg.get("type")
                    result["event_type_counts"][msg_type] = (
                        result["event_type_counts"].get(msg_type, 0) + 1
                    )

                    if msg_type == "heartbeat":
                        continue

                    if msg_type == "result":
                        data = msg.get("data", {})
                        ws_results.append(data)

                        # Randomly trigger calculate while stream is still running
                        can_send_more = result["calculate_calls_sent"] < calc_target
                        if can_send_more and random.random() < CALC_TRIGGER_PROB:
                            sample_size = random.randint(
                                1, min(len(ws_results), MAX_CALC_SAMPLE_SIZE)
                            )
                            snapshot = random.sample(ws_results, sample_size)
                            calc_call_id += 1
                            task = asyncio.create_task(
                                call_calculate(
                                    http_client,
                                    snapshot,
                                    worker_id,
                                    calc_call_id,
                                )
                            )
                            calc_tasks.append(task)
                            result["calculate_calls_sent"] += 1

                    if msg_type == "stats":
                        stats = msg.get("stats", {})
                        aggregated_doc_ids = (
                            msg.get("aggregated_doc_ids") or msg.get("doc_ids") or []
                        )
                        result["overall_verdict"] = stats.get("overall_verdict")
                        result["truth_confidence"] = stats.get("truth_confidence_score")
                        result["total_results"] = msg.get("total_results")
                        result["stats_payload"] = stats
                        result["aggregated_doc_ids"] = aggregated_doc_ids
                        result["used"] = [
                            r
                            for r in ws_results
                            if r.get("doc_id") in aggregated_doc_ids
                        ]

                    if msg_type == "complete":
                        break

            # If random trigger didn't reach target, optionally top-up after complete
            while result["calculate_calls_sent"] < calc_target and ws_results:
                sample_size = random.randint(
                    1, min(len(ws_results), MAX_CALC_SAMPLE_SIZE)
                )
                snapshot = random.sample(ws_results, sample_size)
                calc_call_id += 1
                task = asyncio.create_task(
                    call_calculate(
                        http_client,
                        snapshot,
                        worker_id,
                        calc_call_id,
                    )
                )
                calc_tasks.append(task)
                result["calculate_calls_sent"] += 1

            if calc_tasks:
                calc_outcomes = await asyncio.gather(
                    *calc_tasks, return_exceptions=False
                )
                durations = []
                for item in calc_outcomes:
                    durations.append(item["duration_s"])
                    if item["ok"]:
                        result["calculate_calls_ok"] += 1
                        # keep sample metrics lightweight
                        result["calculate_samples"].append(
                            {
                                "overall_verdict": item.get("overall_verdict"),
                                "truth_confidence_score": item.get(
                                    "truth_confidence_score"
                                ),
                                "duration_s": item["duration_s"],
                            }
                        )
                    else:
                        result["calculate_calls_failed"] += 1

                if durations:
                    result["calculate_avg_duration_s"] = round(
                        sum(durations) / len(durations), 3
                    )

    except Exception as exc:
        result["error"] = str(exc)

    t_end = time.perf_counter()
    result["finished_at"] = datetime.now().isoformat()
    result["duration_s"] = round(t_end - t_start, 2)
    status = "ERROR" if result["error"] else "OK"
    _log(
        f"[W{worker_id:02d}] worker done ({status}, {result['duration_s']}s, "
        f"events={result['events_received']}, recalc={result['calculate_calls_sent']})"
    )

    return result


def build_summary(results: list[dict], concurrency: int, wall_time_s: float) -> dict:
    errors = [r for r in results if r.get("error")]
    succeeded = concurrency - len(errors)

    ok_durations = [r["duration_s"] for r in results if not r.get("error")]
    avg_duration = (
        round(sum(ok_durations) / len(ok_durations), 2) if ok_durations else None
    )

    zero_verdict_workers = [
        r["worker_id"]
        for r in results
        if r.get("overall_verdict") is not None and abs(r["overall_verdict"]) < 1e-9
    ]

    total_events = sum(r.get("events_received", 0) for r in results)

    calc_calls_target = sum(r.get("calculate_calls_target", 0) for r in results)
    calc_calls_sent = sum(r.get("calculate_calls_sent", 0) for r in results)
    calc_calls_ok = sum(r.get("calculate_calls_ok", 0) for r in results)
    calc_calls_failed = sum(r.get("calculate_calls_failed", 0) for r in results)

    calc_avg_durations = [
        r["calculate_avg_duration_s"]
        for r in results
        if r.get("calculate_avg_duration_s") is not None
    ]
    calc_global_avg = (
        round(sum(calc_avg_durations) / len(calc_avg_durations), 3)
        if calc_avg_durations
        else None
    )

    return {
        "concurrency": concurrency,
        "total_wall_time_s": wall_time_s,
        "succeeded": succeeded,
        "errors": len(errors),
        "fastest_s": min(ok_durations) if ok_durations else None,
        "slowest_s": max(ok_durations) if ok_durations else None,
        "avg_s": avg_duration,
        "zero_verdict_worker_ids": zero_verdict_workers,
        "total_events_received": total_events,
        "calculate": {
            "calls_target": calc_calls_target,
            "calls_sent": calc_calls_sent,
            "calls_ok": calc_calls_ok,
            "calls_failed": calc_calls_failed,
            "avg_duration_s": calc_global_avg,
        },
    }


async def main():
    mode_label = "WEBSOCKET"
    run_started_at = datetime.now()
    run_id = run_started_at.strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print("  TrueScope Load Test")
    print(f"  Mode           : {mode_label}")
    print(f"  WS Target      : {WS_URL}")
    print(f"  Calc Target    : {CALCULATE_URL}")
    print(f"  Concurrency    : {CONCURRENCY}")
    print(f"  Calc Calls/Req : random [{CALC_CALLS_MIN}, {CALC_CALLS_MAX}]")
    print(f"  Started        : {run_started_at.isoformat()}")
    print(f"{'='*60}\n")

    tasks = [
        ws_worker(i + 1, TEST_CLAIMS[i % len(TEST_CLAIMS)]) for i in range(CONCURRENCY)
    ]

    wall_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall_end = time.perf_counter()
    wall_time = round(wall_end - wall_start, 2)

    print(f"{'─'*60}")
    print("  Per-request results")
    print(f"{'─'*60}")

    for r in sorted(results, key=lambda x: x["worker_id"]):
        status = "ERROR" if r["error"] else "OK"
        if r["overall_verdict"] is not None:
            verdict_str = (
                f"verdict={r['overall_verdict']:.3f}  "
                f"conf={r['truth_confidence']:.3f}  "
                f"results={r['total_results']}"
            )
        else:
            verdict_str = "no verdict"

        print(
            f"  [{r['worker_id']:02d}] {status:<5} {r['duration_s']}s  "
            f"events={r.get('events_received', 0):<4}  "
            f"calc={r.get('calculate_calls_sent', 0)} "
            f"(ok={r.get('calculate_calls_ok', 0)}, fail={r.get('calculate_calls_failed', 0)})  "
            f"{verdict_str}"
        )
        if r["error"]:
            print(f"        ↳ {r['error']}")

    summary = build_summary(results, CONCURRENCY, wall_time)

    print(f"\n{'─'*60}")
    print("  Summary")
    print(f"{'─'*60}")
    print(f"  Total wall time : {summary['total_wall_time_s']}s")
    print(f"  Succeeded       : {summary['succeeded']}/{CONCURRENCY}")
    print(f"  Errors          : {summary['errors']}")
    if summary["avg_s"] is not None:
        print(f"  Fastest         : {summary['fastest_s']}s")
        print(f"  Slowest         : {summary['slowest_s']}s")
        print(f"  Avg             : {summary['avg_s']}s")
    if summary["zero_verdict_worker_ids"]:
        print(f"  Zero verdict IDs: {summary['zero_verdict_worker_ids']}")
    print(
        "  Calculate calls : "
        f"{summary['calculate']['calls_sent']} "
        f"(target={summary['calculate']['calls_target']}, "
        f"ok={summary['calculate']['calls_ok']}, "
        f"failed={summary['calculate']['calls_failed']}, "
        f"avg={summary['calculate']['avg_duration_s']}s)"
    )

    if LOG_RESULTS:
        LOAD_TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = LOAD_TEST_RESULTS_DIR / f"load_test_websocket_{run_id}.json"

        compact_results = []
        for r in sorted(results, key=lambda x: x["worker_id"]):
            scoring_doc_ids = r.get("aggregated_doc_ids", []) or []
            scoring_results = r.get("used", [])

            compact_results.append(
                {
                    "worker_id": r.get("worker_id"),
                    "claim": r.get("claim"),
                    "score": {
                        "overall_verdict": r.get("overall_verdict"),
                        "truth_confidence": r.get("truth_confidence"),
                        "total_results": r.get("total_results"),
                    },
                    "aggregated_doc_ids": scoring_doc_ids,
                    "scoring_results": scoring_results,
                    "metrics": {
                        "duration_s": r.get("duration_s"),
                        "events_received": r.get("events_received"),
                        "event_type_counts": r.get("event_type_counts"),
                        "error": r.get("error"),
                        "calculate_calls_target": r.get("calculate_calls_target"),
                        "calculate_calls_sent": r.get("calculate_calls_sent"),
                        "calculate_calls_ok": r.get("calculate_calls_ok"),
                        "calculate_calls_failed": r.get("calculate_calls_failed"),
                        "calculate_avg_duration_s": r.get("calculate_avg_duration_s"),
                    },
                }
            )

        payload = {
            "run_id": run_id,
            "started_at": run_started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "mode": mode_label,
            "targets": {
                "ws_url": WS_URL,
                "calculate_url": CALCULATE_URL,
            },
            "summary": summary,
            "results": compact_results,
        }

        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  Saved report    : {output_path}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
