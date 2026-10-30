"""Pass rate of a model package (or the base model) on DeepCoder test problems.

Code runs in AgentCore Code Interpreter.

python coderl/eval_adapter.py --model-package-arn ARN --n 100 --samples 2 --reasoning low \
    [--base-model Qwen/Qwen3.8-27B]
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import boto3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coderl.code_interpreter import CodeInterpreterSandbox  # noqa: E402
from coderl.sandbox_proxy import LocalSandbox  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model-package-arn", default=None)
p.add_argument("--n", type=int, default=100)
p.add_argument("--samples", type=int, default=2)
p.add_argument("--reasoning", default="medium")
p.add_argument("--max-tokens", type=int, default=12288)
p.add_argument("--code-interpreter-name", default="code_rl_sandbox")
p.add_argument("--out", default=None)
p.add_argument("--base-model", default="Qwen/Qwen3.8-27B")
a = p.parse_args()
REGION = os.environ["AWS_REGION"]

ctl = boto3.client("bedrock-agentcore-control", region_name=REGION)
found = [c for c in ctl.list_code_interpreters(maxResults=100)["codeInterpreterSummaries"] if c["name"] == a.code_interpreter_name]
ci_id = found[0]["codeInterpreterId"] if found else ctl.create_code_interpreter(
    name=a.code_interpreter_name, networkConfiguration={"networkMode": "SANDBOX"})["codeInterpreterId"]
while ctl.get_code_interpreter(codeInterpreterId=ci_id)["status"] != "READY":
    time.sleep(5)
ci = CodeInterpreterSandbox(ci_id, REGION, sessions=16)
os.environ["SANDBOX_URL"] = LocalSandbox(ci.run, concurrency=16).start()
os.environ["SANDBOX_MAX_CONCURRENCY"] = "16"

from sagemaker.train.training_session import SamplingParams, ServiceClient, StopReason  # noqa: E402
from training_sessions_cookbook.rl.envs.deepcoder import load_deepcoder_tasks  # noqa: E402
from training_sessions_cookbook.rl.envs.lcb_utils import extract_code_from_model  # noqa: E402
from training_sessions_cookbook.rl.envs.sandbox import sandbox_check_correctness  # noqa: E402
from training_sessions_cookbook.rl.prompts import apply_gpt_oss, apply_hf  # noqa: E402


async def main():
    svc = ServiceClient(region=REGION, role_arn=os.environ["ROLE_ARN"],
                        model_package_group_arn=os.environ["MODEL_PACKAGE_GROUP_ARN"], s3_output_path=os.environ["S3_OUTPUT_PATH"])
    t0 = time.time()
    if a.model_package_arn:
        tc = svc.create_training_client_from_state(model_package_arn=a.model_package_arn, rank=32, alpha=64)
        sc = tc.save_weights_and_get_sampling_client()
    else:
        tc = svc.create_lora_training_client(a.base_model, rank=32, alpha=64)
        sc = tc.create_sampling_client()
    try:
        tok = tc.get_tokenizer()
        tasks = load_deepcoder_tasks("test")[: a.n]
        sp = SamplingParams(max_tokens=a.max_tokens, temperature=1.0)

        def prompt(problem):
            if "gpt-oss" in a.base_model:
                return tok.encode(apply_gpt_oss(problem, reasoning=a.reasoning), add_special_tokens=False)
            return apply_hf(problem, tok, reasoning=a.reasoning)

        t1 = time.time()
        futs = [sc.sample(prompt(t.problem), num_samples=a.samples, sampling_params=sp) for t in tasks]
        res = await asyncio.gather(*[asyncio.to_thread(f.result) for f in futs])
        sample_s = time.time() - t1
        seqs = [s for r in res for s in r.sequences]

        async def grade(task, text):
            code = extract_code_from_model(text)
            ok = False
            if code is not None:
                ok, _ = await sandbox_check_correctness(task.tests, code, 6)
            return ok

        t2 = time.time()
        oks = await asyncio.gather(*[grade(t, tok.decode(s.tokens)) for t, r in zip(tasks, res) for s in r.sequences])
        out = {"model_package_arn": a.model_package_arn, "n": a.n, "samples": a.samples, "reasoning": a.reasoning,
               "base_model": a.base_model, "pass_rate": float(np.mean(oks)),
               "cut": float(np.mean([s.stop_reason == StopReason.LENGTH for s in seqs])),
               "tokens_mean": float(np.mean([len(s.tokens) for s in seqs])),
               "sample_s": sample_s, "grade_s": time.time() - t2, "seconds": time.time() - t0}
        print("EVAL", json.dumps(out), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(out) + "\n")
    finally:
        tc.stop()
        if ci is not None:
            ci.close()


asyncio.run(main())
