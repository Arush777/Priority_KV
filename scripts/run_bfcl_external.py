#!/usr/bin/env python
"""Run one shard of BFCL V3 multi-turn work units under a retention arm.

Loads the model once, iterates the shard, and checkpoints every conversation
atomically so a preempted or timed-out Slurm task resumes without redoing or
duplicating work. Scoring uses the unmodified official ``multi_turn_checker``.

    uv run python scripts/run_bfcl_external.py \
        --config configs/external_bfcl_prajna_v1.yaml --shard-index 0 --shard-size 25
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prioritykv.external import FREEZE_ID  # noqa: E402
from prioritykv.external.arms import (  # noqa: E402
    PressGenerator,
    SnapKVUnavailableError,
    TokenGatherGenerator,
)
from prioritykv.external.bfcl_data import build_system_prompt, load_tasks  # noqa: E402
from prioritykv.external.bfcl_official import (  # noqa: E402
    assert_pinned_revision,
    load_official,
    reset_execution_instances,
)
from prioritykv.external.bfcl_rollout import (  # noqa: E402
    ContextLimitExceeded,
    run_rollout,
    score_rollout,
)
from prioritykv.external.checkpoint import (  # noqa: E402
    ResultStore,
    build_shards,
    completed_work_ids,
    read_jsonl,
    write_failure,
    write_point,
    write_shard_status,
)
from prioritykv.external.config import (  # noqa: E402
    harness_revision,
    keep_policy_config,
    load_config,
    uv_lock_hash,
)

_STOP = {"requested": False, "signal": None}


def _install_signal_handlers() -> None:
    """Flush the current checkpoint and exit cleanly on preemption."""
    def handler(signum, _frame):
        _STOP["requested"] = True
        _STOP["signal"] = int(signum)
        print(f"[run] caught signal {signum}; finishing current unit then stopping",
              flush=True)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/external_bfcl_prajna_v1.yaml"))
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--shard-size", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap units this run")
    ap.add_argument("--arms", default=None, help="restrict to these arms")
    ap.add_argument("--task-ids", default=None, help="restrict to these task ids")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="plan only, no model load")
    return ap.parse_args()


def gpu_info() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        return {
            "cuda": True,
            "device_name": torch.cuda.get_device_name(0),
            "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
        }
    except Exception as exc:  # noqa: BLE001
        return {"cuda": False, "error": str(exc)}


def load_model(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = cfg["model"]["local_dir"]
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    kwargs = {
        "device_map": "cuda:0",
        "trust_remote_code": True,
        "attn_implementation": cfg["model"].get("attn_implementation", "sdpa"),
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, **kwargs
        )
    model.eval()
    return model, tok


def make_generator(arm: str, model, tok, cfg: dict, keep_frac: float, seed: int):
    """FullKV runs unpressed; every other arm is a kvpress press.

    Arms must differ only in which KV entries they keep, never in how those
    entries are removed -- otherwise the table measures mechanism, not policy.
    """
    keep_cfg = keep_policy_config(cfg, keep_frac=keep_frac, seed=seed)
    max_len = int(cfg["model"]["max_model_len"])
    thinking = bool(cfg["model"].get("enable_thinking", False))
    if arm == "full":
        return TokenGatherGenerator(model, tok, arm="full", keep_cfg=keep_cfg,
                                    max_model_len=max_len, enable_thinking=thinking)
    sk = cfg["arms"]["snapkv"]
    if arm == "snapkv" and sk.get("allow_fallback"):
        raise RuntimeError("snapkv.allow_fallback must stay false")
    return PressGenerator(
        model, tok, arm=arm, keep_cfg=keep_cfg,
        window_size=int(sk["window_size"]), kernel_size=int(sk["kernel_size"]),
        max_model_len=max_len, enable_thinking=thinking,
    )


def main() -> int:  # noqa: C901
    args = parse_args()
    _install_signal_handlers()
    cfg = load_config(args.config)

    gorilla_root = cfg["dataset"]["gorilla_root"]
    dataset_revision = assert_pinned_revision(gorilla_root, cfg["dataset"]["gorilla_revision"])
    official = load_official(gorilla_root)
    from bfcl_eval.model_handler.utils import default_decode_execute_prompting

    store = ResultStore(args.out or cfg["paths"]["results_root"]).ensure()
    work_items = read_jsonl(store.manifest / "work_items.jsonl")
    if args.arms:
        allow = set(args.arms.split(","))
        work_items = [w for w in work_items if w["arm"] in allow]
    if args.task_ids:
        allow_t = set(args.task_ids.split(","))
        work_items = [w for w in work_items if w["task_id"] in allow_t]

    shard_size = args.shard_size or int(cfg["cluster"]["shard_size"])
    shard_index = args.shard_index
    if shard_index is not None:
        shards = build_shards(work_items, shard_size)
        if shard_index >= len(shards):
            print(f"[run] shard {shard_index} beyond {len(shards)} shards; nothing to do")
            return 0
        work_items = shards[shard_index].work_items

    done = completed_work_ids(store)
    pending = [w for w in work_items if w["work_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]

    print(f"[run] shard={shard_index} assigned={len(work_items)} "
          f"already_complete={len(work_items) - len(pending)} pending={len(pending)}",
          flush=True)
    if args.dry_run or not pending:
        print("RUN_OK (nothing to do)" if not pending else "DRY_RUN_OK", flush=True)
        return 0

    tasks_by_id = {
        t.task_id: t
        for t in load_tasks(gorilla_root, categories=list(cfg["dataset"]["categories"]),
                            doc_mapping=official["MULTI_TURN_FUNC_DOC_FILE_MAPPING"])
    }

    ginfo = gpu_info()
    print(f"[run] gpu: {json.dumps(ginfo)}", flush=True)
    model, tok = load_model(cfg)

    slurm = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": socket.gethostname(),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    env_meta = {
        "harness_revision": harness_revision(REPO_ROOT),
        "uv_lock_sha256": uv_lock_hash(REPO_ROOT),
        "gpu": ginfo,
        "slurm": slurm,
    }
    max_new = args.max_new_tokens or int(cfg["model"]["max_new_tokens"])
    ceiling = int(cfg["model"]["prompt_token_ceiling"])
    step_limit = int(cfg["protocol"]["max_step_limit"])

    n_ok = n_fail = 0
    t_shard = time.perf_counter()

    for i, item in enumerate(pending):
        if _STOP["requested"]:
            print("[run] stopping before next unit (signal received)", flush=True)
            break

        wid, arm, task_id = item["work_id"], item["arm"], item["task_id"]
        task = tasks_by_id.get(task_id)
        base = {**item, "freeze_id": FREEZE_ID, "dataset_revision": dataset_revision,
                **env_meta}
        t0 = time.perf_counter()
        try:
            import torch

            torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            pass

        try:
            if task is None:
                raise KeyError(f"task {task_id} not in pinned dataset")
            system_prompt = build_system_prompt(task, official["DEFAULT_SYSTEM_PROMPT"])
            generator = make_generator(arm, model, tok, cfg, float(item["keep_frac"]),
                                       int(item["seed"]))
            # Namespace execution state per work unit so arms cannot contaminate
            # one another's stateful API instances.
            reset_execution_instances()
            rollout = run_rollout(
                task, generator,
                system_prompt=system_prompt,
                decode_execute=default_decode_execute_prompting,
                execute_calls=official["execute_multi_turn_func_call"],
                is_empty_response=official["is_empty_execute_response"],
                max_step_limit=step_limit,
                max_new_tokens=max_new,
                prompt_token_ceiling=ceiling,
                execution_model_name=f"pkv_{arm}_{wid[:12]}",
            )
            reset_execution_instances()
            verdict = score_rollout(
                task, rollout,
                multi_turn_checker=official["multi_turn_checker"],
                model_name=f"pkvscore_{arm}_{wid[:12]}",
            )
            reset_execution_instances()

            peak = {}
            try:
                import torch

                peak = {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            except Exception:  # noqa: BLE001
                pass

            payload = {
                **base,
                "terminal_status": rollout.terminal_status,
                "score_valid": bool(verdict.get("valid")),
                "score_verdict": verdict,
                "force_quit": rollout.force_quit,
                "steps_used": rollout.steps_used,
                "n_turns": task.n_turns,
                "prompt_token_counts": rollout.prompt_token_counts,
                "max_prompt_tokens": rollout.max_prompt_tokens,
                "requested_keep": rollout.requested_keep,
                "realized_keep": rollout.realized_keep,
                "model_result_decoded": rollout.model_result_decoded,
                "raw_outputs": rollout.raw_outputs,
                "timings": rollout.timings,
                "rollout_extra": rollout.extra,
                "memory": peak,
                "wall_seconds": time.perf_counter() - t0,
            }
            write_point(store, payload)
            n_ok += 1
            print(f"[run] {i + 1}/{len(pending)} {arm} {task_id} "
                  f"valid={payload['score_valid']} steps={rollout.steps_used} "
                  f"{payload['wall_seconds']:.1f}s", flush=True)

        except ContextLimitExceeded as exc:
            write_failure(store, {**base, "terminal_status": "excluded_context_limit",
                                  "reason": "MODEL_CONTEXT_LIMIT", "error": str(exc),
                                  "prompt_tokens": exc.prompt_tokens, "ceiling": exc.limit})
            n_fail += 1
            print(f"[run] EXCLUDED {arm} {task_id}: {exc}", flush=True)

        except SnapKVUnavailableError as exc:
            # Never degrade into a fake baseline: stop the shard outright.
            write_failure(store, {**base, "terminal_status": "snapkv_unavailable",
                                  "error": str(exc)})
            print(f"[run] FATAL snapkv unavailable: {exc}", flush=True)
            raise

        except Exception as exc:  # noqa: BLE001
            status = "oom" if "out of memory" in str(exc).lower() else "model_failure"
            write_failure(store, {**base, "terminal_status": status,
                                  "error": f"{type(exc).__name__}: {exc}",
                                  "traceback": traceback.format_exc()[-4000:]})
            n_fail += 1
            print(f"[run] FAILED {arm} {task_id}: {type(exc).__name__}: {exc}", flush=True)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    status = {
        "shard_index": shard_index,
        "assigned": len(work_items),
        "attempted": n_ok + n_fail,
        "succeeded": n_ok,
        "failed": n_fail,
        "remaining": len(pending) - (n_ok + n_fail),
        "stopped_by_signal": _STOP["signal"],
        "elapsed_s": time.perf_counter() - t_shard,
        **env_meta,
    }
    write_shard_status(store, shard_index if shard_index is not None else -1, status)
    print(json.dumps(status, indent=2), flush=True)
    print("RUN_INTERRUPTED" if _STOP["requested"] else "RUN_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
