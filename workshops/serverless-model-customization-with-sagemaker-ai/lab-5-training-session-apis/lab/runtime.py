"""Train a Strands agent with GRPO on a SageMaker training session.

The agent runs in this Python process. Its model calls go to a local rollout
gateway (AgentCore RL Toolkit), which renders chat messages to tokens, samples
through the training session's SamplingClient, and records token ids, logprobs
and loss masks per turn. Rewards come from the use case's Python scorer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm.auto import tqdm
from sagemaker.train.training_session import AdamParams, ServiceClient

from agentcore_rl_toolkit.backends.experimental.sagemaker.datum import trace_record_to_datum
from agentcore_rl_toolkit.rollout_gateway import BaseTrace, HfTemplateRenderer
from agentcore_rl_toolkit.rollout_gateway.gateway import RolloutGateway
from agentcore_rl_toolkit.rollout_gateway.sampling_backends.sagemaker_sdk import SageMakerSdkBackend
from agentcore_rl_toolkit.rollout_gateway.server import ThreadedGatewayServer


logger = logging.getLogger("lab")

# run_agent(payload, model_config) -> {"reward": float, ...}
AgentFn = Callable[[dict, dict], dict]


class SamplingBackend(SageMakerSdkBackend):
    """Gateway sampling backend that remembers which rollouts had a failed sampling call."""

    def __init__(self, sampling_client):
        super().__init__(sampling_client)
        self.failed_sessions: set[str] = set()

    async def generate(self, **kw):
        try:
            return await super().generate(**kw)
        except Exception:
            # the rollout is left out of training
            if kw.get("session_id"):
                self.failed_sessions.add(kw["session_id"])
            raise


@dataclass
class RunConfig:
    region: str
    base_model_arn: str
    role_arn: str
    model_package_group_arn: str
    s3_output_path: str
    lora_rank: int = 32
    lora_alpha: int = 64
    learning_rate: float = 4e-5
    loss: str = "ppo"
    groups_per_step: int = 8
    group_size: int = 8
    temperature: float = 1.0
    max_new_tokens: int = 2048
    max_context_tokens: int = 16384
    max_turns: int = 12
    rollout_timeout: float = 600.0
    concurrency: int = 64
    chat_template_kwargs: dict = field(default_factory=dict)
    progress: bool = True  # show a progress bar for the rollouts of each step and each evaluation
    history_mode: str = "linear"
    adam_beta2: float = 0.95
    grad_clip_norm: float | None = 1.0
    # "sync": wait for the new sampler after each update; "overlap": keep sampling with
    # the previous weights while the snapshot is created (rollouts are then at most one step stale)
    refresh: str = "sync"
    train_op_timeout: float = 900.0


@dataclass
class Rollout:
    payload: dict
    reward: float
    result: dict
    records: list
    seconds: float
    error: str | None = None
    # True when a sampling call failed on the service side; such rollouts are not trained on
    failed: bool = False


class TrainingRun:
    """One training session plus the local gateway the agent talks to."""

    def __init__(self, cfg: RunConfig, out_dir: str | Path | None = None):
        self.cfg = cfg
        self.out_dir = Path(out_dir) if out_dir else None
        self.tc = None
        self.gateway = None
        self.server = None
        self.backend = None
        self.step = 0
        self._refresh_task = None
        self.refresh_seconds: list[float] = []
        self._sem = asyncio.Semaphore(cfg.concurrency)

    # -- lifecycle -----------------------------------------------------------

    def open(self, resume_from_model_package: str | None = None, init_from_model_package: str | None = None) -> "TrainingRun":
        cfg = self.cfg
        svc = ServiceClient(
            region=cfg.region,
            role_arn=cfg.role_arn,
            model_package_group_arn=cfg.model_package_group_arn,
            s3_output_path=cfg.s3_output_path,
        )
        t0 = time.monotonic()
        if init_from_model_package:
            # weights only (e.g. an adapter registered with scripts/register_adapter.py)
            self.tc = svc.create_training_client_from_state(
                model_package_arn=init_from_model_package, rank=cfg.lora_rank, alpha=cfg.lora_alpha
            )
        elif resume_from_model_package:
            self.tc = svc.create_training_client_from_state_with_optimizer(
                model_package_arn=resume_from_model_package,
                base_model=cfg.base_model_arn,
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
            )
        else:
            self.tc = svc.create_lora_training_client(
                base_model=cfg.base_model_arn, rank=cfg.lora_rank, alpha=cfg.lora_alpha
            )
        logger.info("training session %s ready in %.0fs", self.tc.training_session_name, time.monotonic() - t0)
        tokenizer = self.tc.get_tokenizer()
        self.backend = SamplingBackend(self.tc.create_sampling_client())
        renderer = HfTemplateRenderer(tokenizer, chat_template_kwargs=cfg.chat_template_kwargs)
        self.gateway = RolloutGateway(
            backend=self.backend,
            renderer=renderer,
            tokenizer=tokenizer,
            adapters=["openai"],
            history_mode=cfg.history_mode,
            max_turns_per_sid=cfg.max_turns,
        )
        # Loopback only: the agent runs in this process.
        self.server = ThreadedGatewayServer(self.gateway, host="127.0.0.1", port=0)
        self.server.start()
        return self

    def close(self) -> None:
        if self.tc is not None:
            try:
                self.tc.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("stopping training session failed: %s", exc)
            self.tc = None
        if self.server is not None:
            self.server.shutdown()
            self.server = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    @property
    def session_name(self) -> str:
        return self.tc.training_session_name

    # -- rollouts ------------------------------------------------------------

    def model_config(self, sid: str) -> dict:
        return {"base_url": self.server.base_url, "model_id": "policy", "api_key": sid, "max_turns": self.cfg.max_turns}

    async def rollout(self, agent_fn: AgentFn, payload: dict, temperature: float | None = None) -> Rollout:
        cfg = self.cfg
        sid = uuid.uuid4().hex
        sampling = {"max_new_tokens": cfg.max_new_tokens, "temperature": cfg.temperature if temperature is None else temperature}
        async with self._sem:
            self.gateway.create_session(sid, sampling_defaults=sampling, max_context_tokens=cfg.max_context_tokens)
            t0 = time.monotonic()
            result: dict = {}
            error = None
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(agent_fn, payload, self.model_config(sid)), cfg.rollout_timeout
                )
            except asyncio.TimeoutError:
                error = "timeout"
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"[:300]
            reward = float(result.get("reward", 0.0)) if isinstance(result, dict) else 0.0
            # finish the session on the gateway's own event loop
            fut = asyncio.run_coroutine_threadsafe(
                self.gateway.finish_session(sid, base_sample=BaseTrace(rollout_id=sid), reward=reward), self.server._loop
            )
            records = await asyncio.wrap_future(fut)
            if sid in self.backend.failed_sessions:
                self.backend.failed_sessions.discard(sid)
                error = error or "sampling failed"
                return Rollout(payload, reward, result, [], time.monotonic() - t0, error, failed=True)
            return Rollout(payload, reward, result, [r for r in records if r.token_ids], time.monotonic() - t0, error)

    async def _gather(self, coros: list, desc: str) -> list:
        """Run coroutines concurrently, in order, with an optional progress bar."""
        bar = tqdm(total=len(coros), desc=desc, leave=False) if self.cfg.progress else None

        async def tick(c):
            r = await c
            if bar is not None:
                bar.update(1)
            return r

        try:
            return list(await asyncio.gather(*[tick(c) for c in coros]))
        finally:
            if bar is not None:
                bar.close()

    async def evaluate(self, agent_fn: AgentFn, payloads: list[dict], temperature: float = 0.0) -> list[Rollout]:
        return await self._gather([self.rollout(agent_fn, p, temperature) for p in payloads], "evaluation")

    # -- training ------------------------------------------------------------

    async def train_step(self, agent_fn: AgentFn, batch: list[dict]) -> tuple[dict, list[list[Rollout]]]:
        cfg = self.cfg
        t0 = time.monotonic()
        flat_rollouts = await self._gather(
            [self.rollout(agent_fn, p) for p in batch for _ in range(cfg.group_size)], f"step {self.step} rollouts")
        groups = [flat_rollouts[i * cfg.group_size:(i + 1) * cfg.group_size] for i in range(len(batch))]
        t_rollout = time.monotonic() - t0

        datums, skipped = [], 0
        for group in groups:
            group = [r for r in group if not r.failed]
            rewards = np.array([r.reward for r in group], dtype=np.float64)
            if len(group) < 2 or rewards.std() < 1e-6:
                skipped += 1
                continue
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            for r, a in zip(group, adv):
                for rec in r.records:
                    d = trace_record_to_datum(rec, float(a))
                    if d is not None:
                        datums.append(d)

        flat = [r for g in groups for r in g]
        m = {
            "step": self.step,
            "reward_mean": float(np.mean([r.reward for r in flat])),
            "reward_std": float(np.std([r.reward for r in flat])),
            "groups_skipped": skipped,
            "groups": len(groups),
            "rollouts": len(flat),
            "errors": sum(r.error is not None for r in flat),
            "failed": sum(r.failed for r in flat),
            "datums": len(datums),
            "turns_mean": float(np.mean([r.result.get("turns", 0) for r in flat if isinstance(r.result, dict)] or [0])),
            "rollout_s_mean": float(np.mean([r.seconds for r in flat])),
            "gen_tokens_mean": float(np.mean([sum(sum(rec.loss_mask) for rec in r.records) for r in flat])),
            "time_rollout": t_rollout,
        }
        for key in sorted({k for r in flat if isinstance(r.result, dict) for k in r.result.get("metrics", {})}):
            vals = [r.result["metrics"][key] for r in flat if isinstance(r.result, dict) and key in r.result.get("metrics", {})]
            m[f"m/{key}"] = float(np.mean(vals))

        t1 = time.monotonic()
        if self._refresh_task is not None:
            # overlap mode: the previous refresh must finish before the next one starts
            await self._refresh_task
            self._refresh_task = None
        if datums:
            fb = self.tc.forward_backward(datums, loss_fn=cfg.loss)
            op = self.tc.optim_step(AdamParams(learning_rate=cfg.learning_rate, beta2=cfg.adam_beta2,
                                               grad_clip_norm=cfg.grad_clip_norm))
            fb_res = await asyncio.to_thread(fb.result, timeout=cfg.train_op_timeout)
            await asyncio.to_thread(op.result, timeout=cfg.train_op_timeout)
            m.update({f"train/{k}": v for k, v in (fb_res.metrics or {}).items() if isinstance(v, (int, float))})
        m["time_train"] = time.monotonic() - t1

        t2 = time.monotonic()
        if datums:
            if cfg.refresh == "overlap":
                # next step starts sampling with the current weights and switches when the snapshot is ready
                self._refresh_task = asyncio.create_task(self._refresh())
            else:
                await self._refresh()
        m["time_refresh"] = time.monotonic() - t2
        m["time_step"] = time.monotonic() - t0
        self.step += 1
        self._log("metrics.jsonl", m)
        return m, groups

    async def refresh(self, attempts: int = 3) -> None:
        """Snapshot the current weights and switch the sampler to them.

        Retries on service errors; if every attempt fails, sampling continues with the previous weights.
        """
        t = time.monotonic()
        for i in range(attempts):
            try:
                sc = await asyncio.to_thread(self.tc.save_weights_and_get_sampling_client)
                self.backend.set_sampling_client(sc)
                self.refresh_seconds.append(time.monotonic() - t)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("refresh attempt %d failed: %s", i + 1, str(exc)[:200])
                await asyncio.sleep(10 * (i + 1))
        logger.warning("refresh failed %d times; keeping the previous sampler", attempts)

    _refresh = refresh

    async def wait_refresh(self) -> None:
        if self._refresh_task is not None:
            await self._refresh_task
            self._refresh_task = None

    def save(self, inference: bool = True) -> str:
        if inference:
            res = self.tc.save_weights_for_sampler(description=f"step {self.step}").result(timeout=self.cfg.train_op_timeout)
        else:
            res = self.tc.save_state().result(timeout=self.cfg.train_op_timeout)
        return res.model_package_arn

    # -- logging -------------------------------------------------------------

    def _log(self, name: str, row: dict) -> None:
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / name, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def log_rollouts(self, name: str, rollouts: list[Rollout], **extra) -> None:
        for r in rollouts:
            res = r.result if isinstance(r.result, dict) else {}
            self._log(name, {**extra, "id": r.payload.get("id"), "reward": r.reward, "error": r.error or res.get("error"),
                             "seconds": round(r.seconds, 1), "turns": res.get("turns"),
                             "metrics": res.get("metrics"), "transcript": res.get("transcript")})


def summarize_eval(rollouts: list[Rollout]) -> dict:
    out = {"n": len(rollouts), "reward_mean": float(np.mean([r.reward for r in rollouts])),
           "errors": sum(r.error is not None for r in rollouts),
           "seconds_mean": float(np.mean([r.seconds for r in rollouts])),
           "gen_tokens_mean": float(np.mean([sum(sum(rec.loss_mask) for rec in r.records) for r in rollouts]))}
    keys = {k for r in rollouts if isinstance(r.result, dict) for k in r.result.get("metrics", {})}
    for k in sorted(keys):
        vals = [r.result["metrics"][k] for r in rollouts if isinstance(r.result, dict) and k in r.result.get("metrics", {})]
        out[k] = float(np.mean(vals))
    return out


def config_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)
