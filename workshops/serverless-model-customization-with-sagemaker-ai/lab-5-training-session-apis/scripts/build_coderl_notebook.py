"""Build 5c_code_rl.ipynb (Qwen3.8-27B)."""
import textwrap
from pathlib import Path

import nbformat as nbf

V = dict(MODEL="Qwen/Qwen3.8-27B", MODEL_SHORT="Qwen3.8-27B", OUT="5c_code_rl.ipynb",
         REASONING="medium", EVAL_N="50", GRPO_STEPS="2",
         BASE_MODEL_SPEC='{"HubContentName": "huggingface-vlm-qwen3-8-27b", "HubContentVersion": "1.3.0"}',
         RESULTS=("At medium reasoning effort, many of the base model's responses reach the 12,288-token limit before they\n"
                  "contain code. In the reference runs (100 problems x 2 attempts), the base model passed 12.0-12.5% and the\n"
                  "pre-trained adapter 36.0-38.5%. This notebook uses 50 problems to save time, so expect some variation."))

cells = []


def fill(s):
    for k, v in V.items():
        s = s.replace("{{" + k + "}}", v)
    return textwrap.dedent(s).strip("\n")


def md(s):
    cells.append(nbf.v4.new_markdown_cell(fill(s)))


def code(s):
    cells.append(nbf.v4.new_code_cell(fill(s)))


md("""
# Code RL with SageMaker AI training sessions

This notebook trains a model with reinforcement learning (RL) on code. The model gets a competitive-programming
problem and writes a Python solution. The solution runs against the problem's test cases in a sandbox, and the
reward is 1.0 when all tests pass.

The same setup works for any output that you can run and check automatically: SQL queries checked against a
database, code checked by unit tests, configuration files checked by a validator, data transformations checked
against expected output. To use it for your own task, replace the problems and the `grade` function and keep the
sandbox and the loop.

### How the pieces fit together

![Code RL loop](images/code-rl-loop.svg)

- **Training loop** (your code, in this kernel): samples attempts from the training session, scores them and sends
  the update.
- **Training session** (SageMaker AI): holds `{{MODEL_SHORT}}` with a LoRA (low-rank adaptation) adapter, generates
  responses and runs `forward_backward` and `optim_step`.
- **Grading** ([training sessions cookbook](https://github.com/aws/sagemaker-training-sessions-cookbook)): extracts
  the code block from a response and runs the problem's tests with the LiveCodeBench harness.
- **Sandbox**: the model's code is untrusted, so it runs in Amazon Bedrock AgentCore Code Interpreter, without
  network access or AWS credentials. The cookbook's grading code expects a SandboxFusion-style HTTP endpoint, so a
  small local endpoint on `127.0.0.1` in this kernel forwards each program to Code Interpreter.

### What you do

Responses are up to 12,288 tokens, so a training step that moves the pass rate (64 problems x 8 attempts) takes 13
to 22 minutes. Training for hours does not fit in a workshop, so you:

1. measure the base model on held-out problems,
2. run {{GRPO_STEPS}} GRPO (Group Relative Policy Optimization) steps yourself to see the loop work,
3. load an adapter that was trained beforehand with the cookbook's Code RL recipe (20 steps, about 6 hours) and
   measure it on the same problems.

{{RESULTS}}
""")

md("## 1. Setup")
md("""
Install the requirements, then restart the kernel (**Kernel > Restart Kernel**) so that the new package versions
are loaded.
""")
code("""
%pip install -r requirements.txt
""")

code("""
import asyncio, json, os, random, time
import boto3
import numpy as np

session = boto3.Session()
REGION = session.region_name
ACCOUNT = session.client("sts").get_caller_identity()["Account"]
ROLE_ARN = session.client("iam").get_role(RoleName="SageMakerTrainingSessionRole")["Role"]["Arn"]
S3_OUTPUT = f"s3://sagemaker-{REGION}-{ACCOUNT}/training-sessions/code-rl/"
PRETRAINED_ADAPTER = os.environ.get("PRETRAINED_ADAPTER_S3", "s3://<bucket>/code-rl/adapter/model/")

sm = session.client("sagemaker")
try:
    MPG_ARN = sm.create_model_package_group(ModelPackageGroupName="code-rl")["ModelPackageGroupArn"]
except sm.exceptions.ClientError:
    MPG_ARN = sm.describe_model_package_group(ModelPackageGroupName="code-rl")["ModelPackageGroupArn"]
print(REGION, ROLE_ARN, MPG_ARN, sep="\\n")
""")
md("""
### The sandbox

Generated code runs in Amazon Bedrock AgentCore Code Interpreter. The code interpreter created below has network
mode `SANDBOX` (no network access) and no execution role, so the code it runs has no AWS credentials.
`CodeInterpreterSandbox` (in `coderl/code_interpreter.py`) keeps 16 sessions open and runs each program in its own
directory of one of them.

The cookbook's grading code sends programs to a SandboxFusion-style HTTP endpoint, so `LocalSandbox` serves
`/run_code` on `127.0.0.1` in this kernel and passes each request to `CodeInterpreterSandbox.run`.
""")
code("""
from coderl.code_interpreter import CodeInterpreterSandbox
from coderl.sandbox_proxy import LocalSandbox

agentcore = session.client("bedrock-agentcore-control")
found = [c for c in agentcore.list_code_interpreters(maxResults=100)["codeInterpreterSummaries"]
         if c["name"] == "code_rl_sandbox"]
if found:
    CODE_INTERPRETER_ID = found[0]["codeInterpreterId"]
else:
    CODE_INTERPRETER_ID = agentcore.create_code_interpreter(
        name="code_rl_sandbox", networkConfiguration={"networkMode": "SANDBOX"})["codeInterpreterId"]
while agentcore.get_code_interpreter(codeInterpreterId=CODE_INTERPRETER_ID)["status"] != "READY":
    time.sleep(5)

sandbox = CodeInterpreterSandbox(CODE_INTERPRETER_ID, REGION, sessions=16)
os.environ["SANDBOX_URL"] = LocalSandbox(sandbox.run, concurrency=16).start()
os.environ["SANDBOX_MAX_CONCURRENCY"] = "16"
print(sandbox.run({"code": "print(sum(range(10)))", "run_timeout": 10})["run_result"]["stdout"])
""")

md("""
## 2. Problems and reward

`load_deepcoder_tasks` returns problems with their test cases. The prompt asks for a solution in a fenced Python
block. `grade` extracts the last code block, runs the LiveCodeBench test harness in the sandbox, and computes the
cookbook's reward: 1.0 if all tests pass, minus 0.1 if the answer has no code block.
""")
code("""
from training_sessions_cookbook.rl.envs.code_rl import deepcoder_reward
from training_sessions_cookbook.rl.envs.deepcoder import load_deepcoder_tasks
from training_sessions_cookbook.rl.envs.lcb_utils import extract_code_from_model
from training_sessions_cookbook.rl.envs.sandbox import sandbox_check_correctness
from training_sessions_cookbook.rl.prompts import apply_hf

train_tasks = load_deepcoder_tasks("train", seed=0, max_prompts=512)
test_tasks = load_deepcoder_tasks("test")[:{{EVAL_N}}]
print(len(train_tasks), "train |", len(test_tasks), "test")
print(test_tasks[0].problem[:800])


async def grade(task, response_text):
    code = extract_code_from_model(response_text)
    passed = False
    if code is not None:
        passed, _ = await sandbox_check_correctness(task.tests, code, 6)
    reward, _ = deepcoder_reward(response_text, passed)
    return reward, passed
""")

md("""
## 3. Base model on held-out problems

The prompt goes through the model's chat template with {{REASONING}} reasoning effort. Responses have up to 12,288 tokens at
temperature 1.0. Each problem is sampled twice; the progress bars show sampling and then grading.
""")
code("""
from sagemaker.train.training_session import AdamParams, SamplingParams, ServiceClient

from coderl.progress import gather

svc = ServiceClient(region=REGION, role_arn=ROLE_ARN, model_package_group_arn=MPG_ARN, s3_output_path=S3_OUTPUT)
training_client = svc.create_lora_training_client(
    "{{MODEL}}", training_session_name=f"code-rl-{int(time.time())}", rank=32, alpha=64)
tokenizer = training_client.get_tokenizer()
sampling_client = training_client.create_sampling_client()
SAMPLING = SamplingParams(max_tokens=12288, temperature=1.0)


def prompt_tokens(task):
    return apply_hf(task.problem, tokenizer, reasoning="{{REASONING}}")


async def evaluate(client, tasks, samples_per_problem=2):
    futures = [client.sample(prompt_tokens(t), num_samples=samples_per_problem, sampling_params=SAMPLING) for t in tasks]
    results = await gather([asyncio.to_thread(f.result) for f in futures], "sampling")
    pairs = [(t, tokenizer.decode(s.tokens)) for t, r in zip(tasks, results) for s in r.sequences]
    graded = await gather([grade(t, x) for t, x in pairs], "grading")
    return float(np.mean([p for _, p in graded])), pairs


t0 = time.time()
base_pass, base_texts = await evaluate(sampling_client, test_tasks)
print(f"base model: pass rate {base_pass:.1%} on {len(test_tasks)} held-out problems, 2 attempts each ({time.time() - t0:.0f}s)")
""")

md("""
## 4. A few GRPO steps

![One GRPO step](images/grpo.svg)

Each step samples 8 attempts for each of 8 training problems and grades them in the sandbox. Within each group of 8,
the advantage of an attempt is its reward minus the group mean, divided by the group standard deviation. Groups
where all 8 attempts got the same reward are skipped. One training datum per attempt holds the prompt and response
tokens, the sampling logprobs, the advantage, and weights (0 for prompt tokens, 1 for response tokens).
`forward_backward` with the `ppo` loss (proximal policy optimization) and `optim_step` update the adapter, and
`save_weights_and_get_sampling_client` gives a sampler for the new weights.

{{GRPO_STEPS}} steps of 8 problems are not enough to change the held-out pass rate; section 5 measures an adapter trained
for 20 steps of 64 problems.
""")
code("""
def build_datum(prompt, response_tokens, logprobs, advantage):
    tokens = prompt + list(response_tokens)
    inputs, targets = tokens[:-1], tokens[1:]
    n_prompt = len(prompt) - 1
    n_resp = len(inputs) - n_prompt
    seq = len(inputs)
    return {
        "modelInput": {"chunks": [{"tokens": inputs, "type": "encoded_text"}]},
        "lossFnInputs": {
            "target_tokens": {"data": targets, "dtype": "int64", "shape": [seq]},
            "logprobs": {"data": [0.0] * n_prompt + list(logprobs), "dtype": "float32", "shape": [seq]},
            "advantages": {"data": [0.0] * n_prompt + [advantage] * n_resp, "dtype": "float32", "shape": [seq]},
            "weights": {"data": [0.0] * n_prompt + [1.0] * n_resp, "dtype": "float32", "shape": [seq]},
        },
    }


STEPS, PROBLEMS_PER_STEP, K = {{GRPO_STEPS}}, 8, 8
ADAM = AdamParams(learning_rate=1e-5, beta2=0.95, grad_clip_norm=1.0)
rng = random.Random(0)

for step in range(STEPS):
    t0 = time.time()
    batch = rng.sample(train_tasks, PROBLEMS_PER_STEP)
    prompts = [prompt_tokens(t) for t in batch]
    futures = [sampling_client.sample(p, num_samples=K, sampling_params=SAMPLING) for p in prompts]
    results = await gather([asyncio.to_thread(f.result) for f in futures], f"step {step} sampling")
    graded = await gather([grade(t, tokenizer.decode(s.tokens)) for t, r in zip(batch, results) for s in r.sequences],
                          f"step {step} grading")

    datums, rewards_all, skipped = [], [], 0
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        seqs = result.sequences
        rewards = np.array([r for r, _ in graded[i * K:(i + 1) * K]])
        rewards_all += list(rewards)
        if rewards.std() < 1e-6:
            skipped += 1
            continue
        advantages = (rewards - rewards.mean()) / rewards.std()
        datums += [build_datum(prompt, s.tokens, s.logprobs, float(a)) for s, a in zip(seqs, advantages)]

    if datums:
        fb = training_client.forward_backward(datums, loss_fn="ppo")
        opt = training_client.optim_step(ADAM)
        await asyncio.to_thread(fb.result)
        await asyncio.to_thread(opt.result)
        sampling_client = await asyncio.to_thread(training_client.save_weights_and_get_sampling_client)
    print(f"step {step}  train reward {np.mean(rewards_all):.3f}  skipped groups {skipped}/{PROBLEMS_PER_STEP}  "
          f"datums {len(datums)}  {time.time() - t0:.0f}s")
""")
md("Stop this session before opening the next one.")
code("""
training_client.stop()
""")

md("""
## 5. Evaluate the pre-trained adapter

The adapter was trained with the cookbook's Code RL recipe: 20 steps of 64 problems x 8 attempts on
`{{MODEL_SHORT}}`, LoRA rank 32, about 6 hours. Its files (`adapter_config.json`, `adapter_model.safetensors`) are in
S3. To use them, copy them to your bucket and register them as a model package in your account with the layout that
`save_weights_for_sampler` produces. A new session then starts from these weights with
`create_training_client_from_state`.

The same steps move an adapter between accounts or Regions, and let you continue training from an adapter that a
long-running script produced.
""")
code("""
from urllib.parse import urlparse

s3 = session.client("s3")
src = urlparse(PRETRAINED_ADAPTER)
dest_prefix = "training-sessions/code-rl/pretrained/model/"
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=src.netloc, Prefix=src.path.lstrip("/")):
    for obj in page.get("Contents", []):
        rel = obj["Key"][len(src.path.lstrip("/")):]
        s3.copy({"Bucket": src.netloc, "Key": obj["Key"]}, f"sagemaker-{REGION}-{ACCOUNT}", dest_prefix + rel)

package_arn = sm.create_model_package(
    ModelPackageGroupName="code-rl",
    ModelPackageDescription="pre-trained Code RL adapter",
    ModelApprovalStatus="Approved",
    InferenceSpecification={"Containers": [{
        "ModelDataSource": {"S3DataSource": {"S3Uri": f"s3://sagemaker-{REGION}-{ACCOUNT}/{dest_prefix}",
                                             "S3DataType": "S3Prefix", "CompressionType": "None"}},
        "IsCheckpoint": False,
        "BaseModel": {{BASE_MODEL_SPEC}},
    }]},
)["ModelPackageArn"]
print(package_arn)
""")
md("""
`create_sampling_client()` on a client created from a model package samples the base weights. Call
`save_weights_and_get_sampling_client()` once to get a sampler for the loaded adapter.
""")
code("""
pretrained_client = svc.create_training_client_from_state(
    model_package_arn=package_arn, training_session_name=f"code-rl-pretrained-{int(time.time())}", rank=32, alpha=64)
pretrained_sampler = pretrained_client.save_weights_and_get_sampling_client()

t0 = time.time()
pretrained_pass, pretrained_texts = await evaluate(pretrained_sampler, test_tasks)
print(f"pass rate: base {base_pass:.1%} -> pre-trained adapter {pretrained_pass:.1%} "
      f"({len(test_tasks)} held-out problems, 2 attempts each, {time.time() - t0:.0f}s)")
""")
code("""
pretrained_client.stop()
sandbox.close()
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / V["OUT"]
nbf.write(nb, out)
print("wrote", out)
