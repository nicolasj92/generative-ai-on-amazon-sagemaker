"""Build 5b_scheduling_agent.ipynb (Qwen3.8-27B, thinking off)."""
from pathlib import Path

import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md("""
# Train a scheduling agent with SageMaker AI training sessions

In this notebook you train the model behind a [Strands](https://strandsagents.com) agent with reinforcement
learning (RL). The agent books meetings. A user asks for a meeting with several people on a given day; the agent
reads each attendee's calendar with a tool, finds the earliest start time that fits everyone's working hours across
time zones, and books it. It is a multi-turn agent: one request takes several model calls and tool calls.

The task stands for a common type of business agent: one that collects data through tools and has to combine it
correctly under rules. Examples are scheduling, resource allocation, eligibility checks and order changes. The
same notebook trains such an agent if you replace the tools, the requests and the reward.

### How the pieces fit together

![Agent training loop](images/agent-loop.svg)

- **Strands agent** (your code): a normal Strands agent with two tools, `get_calendar` and `book_meeting`.
- **Rollout gateway** ([AgentCore RL Toolkit](https://github.com/awslabs/agentcore-rl-toolkit)): runs on
  `127.0.0.1` in this kernel and serves an OpenAI-compatible chat API. The agent sends its model calls there. The
  gateway turns the chat messages into tokens with the model's chat template, samples from the training session,
  parses tool calls, and records the token ids and logprobs of every call.
- **Training session** (SageMaker AI): holds `Qwen3.8-27B` with a LoRA (low-rank adaptation) adapter, generates
  tokens, and runs `forward_backward` and `optim_step`.
- **GRPO step** (Group Relative Policy Optimization, your code): scores each episode from the booking the agent
  made and turns the recorded model calls into training data.

The model runs with thinking turned off (`enable_thinking=False` in the chat template), so each model call answers
directly with text or a tool call.

| Step | Time |
|---|---|
| 1. Setup | 5 min |
| 2. The calendars and the agent | 10 min |
| 3. Open a training session, measure the base model | 5 min |
| 4. Train with GRPO | 30 min |
| 5. Evaluate, save the adapter | 5 min |
""")

md("## 1. Setup")
md("""
Install the requirements, then restart the kernel (**Kernel > Restart Kernel**) so that the new package versions
are loaded.
""")
code("""
%pip install -r requirements.txt
""")

md("""
The next cell connects to your AWS account:

- **Region and account** come from the default boto3 session, so the training session runs in the Region your
  notebook uses.
- **`ROLE_ARN`** is the IAM role `SageMakerTrainingSessionRole`. The training session service assumes it to read the
  base model and write adapters to S3. The workshop stack creates it.
- **`S3_OUTPUT`** is the S3 prefix where the service writes the adapter files, in your
  `sagemaker-<region>-<account>` bucket.
- **`MPG_ARN`** is a model package group, `scheduling-agent`. Every adapter you save is registered there as a model package
  version. The cell creates the group, or reuses it if it already exists.
- **`BASE_MODEL_ARN`** identifies `Qwen3.8-27B` in the SageMaker Public Hub. `TrainingRun` opens the session with it
  in section 3.
- **Logging:** the `lab` helpers report at `INFO` level when the session is ready or a weight refresh fails.
""")
code("""
import asyncio, json, logging, random, time
import boto3
import numpy as np

logging.basicConfig(level=logging.WARNING)
logging.getLogger("lab").setLevel(logging.INFO)

session = boto3.Session()
REGION = session.region_name
ACCOUNT = session.client("sts").get_caller_identity()["Account"]
BASE_MODEL_ARN = f"arn:aws:sagemaker:{REGION}:aws:hub-content/SageMakerPublicHub/Model/huggingface-vlm-qwen3-8-27b/1.3.0"
ROLE_ARN = session.client("iam").get_role(RoleName="SageMakerTrainingSessionRole")["Role"]["Arn"]
S3_OUTPUT = f"s3://sagemaker-{REGION}-{ACCOUNT}/training-sessions/scheduling-agent/"

sm = session.client("sagemaker")
try:
    MPG_ARN = sm.create_model_package_group(ModelPackageGroupName="scheduling-agent")["ModelPackageGroupArn"]
except sm.exceptions.ClientError:
    MPG_ARN = sm.describe_model_package_group(ModelPackageGroupName="scheduling-agent")["ModelPackageGroupArn"]
print(REGION, ROLE_ARN, MPG_ARN, sep="\\n")
""")

md("""
## 2. The calendars and the agent

The environment is generated in Python, so it is fast, deterministic and free to call many times:

- **People:** 20 people in nine cities, including the half-hour offsets of Mumbai and Adelaide. Each has their
  own working hours (for example 08:00-16:00 or 12:00-20:00 local time) and one to four busy blocks on the
  requested day. Some calendars list busy blocks in local time, others in UTC.
- **Requests:** a meeting with 3 to 5 attendees and a duration of 30 to 90 minutes, for example
  "Please book a 60-minute meeting with Hugo, Greta and Elif on Thursday 2026-11-12, as early as possible."
- **Correct answer:** the earliest start in UTC that fits every attendee, found by checking every 15-minute step
  of the day. About 6% of the requests have no possible slot; the agent should then book `none`.
- **Train and test** requests use different people and calendars, so the test measures whether the agent applies
  the rules, not whether it remembers the data.

The next cell loads the requests and prints one with its correct answer.
""")
code("""
from usecases.scheduling_hard import load, people_for, valid_for, _fmt

train_tasks = load("train", 512)
test_tasks = load("test", 128)
task = test_tasks[0]
print(task["message"])
print("earliest slot (UTC):", _fmt(task["expected"]) if task["expected"] is not None else "none")
""")

md("""
### What the agent does

The agent is multi-turn: in one episode it calls the model several times. Each tool result goes back to the model
as the next turn, until the model answers without a tool call.

![One episode of the scheduling agent](images/agent-episode.svg)

The figure shows a typical episode for a request from the test set; in the reference runs, an episode took about three
model calls and four tool calls on average. The model first asks for the three calendars in one call. With the results, it
converts everyone's working hours and busy blocks to UTC (one calendar is already in UTC), finds the earliest
90-minute gap, and books 20:00 UTC (14:00 in Chicago). A third model call confirms the booking in text. The reward
is computed after the episode from the booking that `book_meeting` recorded. During training, each of the three
model calls becomes one training datum.

The agent is a plain Strands agent with two tools. `run_agent` builds it for one request, runs it, and scores the
single booking it made: 1.0 for the earliest valid slot, 0.5 for a valid but later slot, 0 for an invalid slot or
when it books zero or several times. The model comes from `model_config`: during training that is the rollout
gateway, but the same function runs against Amazon Bedrock.
""")
code("""
from datetime import datetime

from strands import Agent, tool
from strands.agent.conversation_manager import NullConversationManager
from strands.models.openai import OpenAIModel
from strands.models import BedrockModel

SYSTEM_PROMPT = (
    "You schedule meetings. Each person has their own working hours in their local time zone. Calendars list busy "
    "blocks either in the person's local time or in UTC, as stated in the calendar. Find the earliest start time on "
    "the requested date (in UTC, a multiple of 15 minutes) where the meeting fits inside every attendee's working "
    "hours and overlaps none of their busy blocks. Then call book_meeting once with the start time in UTC. If no "
    "slot exists that day, call book_meeting with start_utc set to 'none'."
)


def make_tools(people, bookings):
    @tool
    def get_calendar(person: str) -> str:
        \"\"\"Get a person's city, UTC offset, working hours and busy blocks for the requested day.

        Args:
            person: First name.
        \"\"\"
        p = people.get(person.strip().title())
        if p is None:
            return "unknown person"
        off = int(p["utc_offset"] * 60)
        shift, tz = (off, "UTC") if p["calendar_in_utc"] else (0, "local")
        return json.dumps({"person": person.strip().title(), "city": p["city"], "utc_offset_hours": p["utc_offset"],
                           "working_hours_local": f"{_fmt(p['hours'][0])}-{_fmt(p['hours'][1])}",
                           "busy_times_in": tz, "busy": [f"{_fmt(a - shift)}-{_fmt(b - shift)}" for a, b in p["busy"]]})

    @tool
    def book_meeting(attendees: list[str], start_utc: str, duration_minutes: int) -> str:
        \"\"\"Book the meeting. Call this exactly once.

        Args:
            attendees: First names of all attendees.
            start_utc: Start time in UTC as HH:MM on the requested date, or 'none' if no slot exists.
            duration_minutes: Meeting length in minutes.
        \"\"\"
        bookings.append({"start_utc": str(start_utc).strip(), "duration": int(duration_minutes)})
        return "booked"

    return [get_calendar, book_meeting]


def reward_fn(task, bookings):
    if len(bookings) != 1:
        return 0.0
    b = bookings[0]
    if b["start_utc"].lower() == "none":
        return float(task["expected"] is None)
    try:
        t = datetime.strptime(b["start_utc"].replace("Z", "").strip()[-5:], "%H:%M")
    except ValueError:
        return 0.0
    start = t.hour * 60 + t.minute
    if b["duration"] != task["duration"]:
        return 0.0
    people = people_for(task["seed"])
    if not all(valid_for(people[a], start, start + task["duration"]) for a in task["attendees"]):
        return 0.0
    return 1.0 if start == task["expected"] else 0.5


def make_model(model_config):
    if "bedrock_model_id" in model_config:
        return BedrockModel(model_id=model_config["bedrock_model_id"], temperature=0.0)
    return OpenAIModel(client_args={"base_url": model_config["base_url"], "api_key": model_config["api_key"],
                                    "max_retries": 0, "timeout": 600},
                       model_id="policy")


def run_agent(task, model_config):
    bookings = []
    agent = Agent(model=make_model(model_config), tools=make_tools(people_for(task["seed"]), bookings),
                  system_prompt=SYSTEM_PROMPT, conversation_manager=NullConversationManager(), callback_handler=None)
    try:
        agent(task["message"])
        error = None
    except Exception as exc:  # turn limit, context limit: the episode ends here
        error = str(exc)[:200]
    return {"reward": reward_fn(task, bookings), "messages": agent.messages, "error": error,
            "turns": sum(m["role"] == "assistant" for m in agent.messages)}
""")

md("Run one request with a model on Amazon Bedrock to see the agent work. `show` prints the tool calls and results.")
code("""
def show(messages, width=300):
    for m in messages:
        for b in m.get("content") or []:
            if "toolUse" in b:
                print(f"  -> {b['toolUse']['name']}({json.dumps(b['toolUse']['input'])[:width]})")
            elif "toolResult" in b:
                print("  <-", " ".join(c.get("text", "") for c in b["toolResult"]["content"])[:width])
            elif b.get("text", "").strip():
                print(f"  {m['role']}: {b['text'][:width]}")


out = run_agent(task, {"bedrock_model_id": "openai.gpt-oss-120b-1:0"})
show(out["messages"])
print("reward:", out["reward"], "| earliest slot:", _fmt(task["expected"]) if task["expected"] is not None else "none")
""")

md("""
## 3. Open a training session and measure the base model

`TrainingRun` (in `lab/runtime.py`) creates a LoRA training client on `Qwen3.8-27B` and starts the rollout gateway
on `127.0.0.1`. The gateway renders prompts with the model's chat template; `chat_template_kwargs` turns thinking
off. `run.rollout(run_agent, task)` registers a gateway session, calls `run_agent` with a `model_config` pointing
to the gateway, and returns the reward together with the token-level records of every model call.
""")
code("""
from lab.runtime import RunConfig, TrainingRun

cfg = RunConfig(region=REGION, base_model_arn=BASE_MODEL_ARN, role_arn=ROLE_ARN,
                model_package_group_arn=MPG_ARN, s3_output_path=S3_OUTPUT,
                lora_rank=32, learning_rate=4e-5, loss="ppo",
                chat_template_kwargs={"enable_thinking": False},
                max_new_tokens=1536, max_context_tokens=16384, max_turns=10, rollout_timeout=300, refresh="sync")
run = TrainingRun(cfg).open()
print("training session:", run.session_name)
""")
code("""
async def evaluate(tasks):
    rollouts = await run.evaluate(run_agent, tasks, temperature=0.0)
    return float(np.mean([r.reward for r in rollouts])), rollouts

t0 = time.time()
base_score, base_rollouts = await evaluate(test_tasks)
print(f"base model: mean reward {base_score:.3f} on {len(test_tasks)} held-out requests ({time.time() - t0:.0f}s)")
""")

md("""
## 4. Train with GRPO

![One GRPO step](images/grpo.svg)

Each step takes 8 requests and runs the agent 8 times on each (temperature 1.0). Within each group of 8, the
advantage of an attempt is its reward minus the group mean, divided by the group standard deviation. Groups where
all 8 attempts scored the same are skipped. Every model call of an attempt becomes one training datum with that
attempt's advantage, so an episode with four model calls gives four datums.

`run.train_step` (in `lab/runtime.py`) does this in one call. The core of it:

```python
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
for r, adv in zip(group, advantages):
    datums += [trace_record_to_datum(rec, float(adv)) for rec in r.records]
fb = run.tc.forward_backward(datums, loss_fn="ppo")
opt = run.tc.optim_step(AdamParams(learning_rate=cfg.learning_rate, beta2=0.95, grad_clip_norm=1.0))
await run.refresh()   # save_weights_and_get_sampling_client
```

`ppo` is the proximal policy optimization loss. After the update, `run.refresh()` snapshots the new weights, so the
next step samples from the updated model. A step takes about 2 to 2.5 minutes; the progress bar shows the 64
episodes of the current step.
""")
code("""
GROUPS, STEPS = 8, 12
history = []
rng = random.Random(0)

for step in range(STEPS):
    metrics, groups = await run.train_step(run_agent, rng.sample(train_tasks, GROUPS))
    history.append(metrics["reward_mean"])
    print(f"step {step:2d}  train reward {metrics['reward_mean']:.3f}  skipped groups {metrics['groups_skipped']}/{GROUPS}  "
          f"datums {metrics['datums']:3d}  {metrics['time_step']:.0f}s")
""")
code("""
import matplotlib.pyplot as plt

plt.plot(history, marker="o")
plt.xlabel("step"); plt.ylabel("mean train reward (temperature 1.0)"); plt.grid(alpha=0.3)
plt.show()
""")

md("## 5. Evaluate and save")
code("""
t0 = time.time()
trained_score, trained_rollouts = await evaluate(test_tasks)
print(f"base {base_score:.3f} -> trained {trained_score:.3f} on {len(test_tasks)} held-out requests ({time.time() - t0:.0f}s)")
""")
md("Requests the base model got wrong and the trained model got right:")
code("""
fixed = [(b, t) for b, t in zip(base_rollouts, trained_rollouts) if b.reward < 1 and t.reward == 1]
print(len(fixed), "requests fixed")
for b, t in fixed[:2]:
    p = b.payload
    print("\\nREQUEST:", p["message"], "| earliest slot:", _fmt(p["expected"]) if p["expected"] is not None else "none")
    print("base model:"); show(b.result["messages"])
    print("trained model:"); show(t.result["messages"])
""")
md("""
Save the adapter as a model package. It can be deployed to a SageMaker endpoint like any other model package; the
[training sessions cookbook](https://github.com/aws/sagemaker-training-sessions-cookbook) has a deployment notebook
that serves several adapters on one endpoint.
""")
code("""
model_package_arn = run.save(inference=True)
print(model_package_arn)
""")
md("## Clean up\\n\\nStop the training session. It is billed while it is open.")
code("""
run.close()
""")
md("""
## Use this for your own agent

Replace three things and keep the training loop:

| Part | In this notebook | For your agent |
|---|---|---|
| Tools | `get_calendar`, `book_meeting` | Your APIs, or mocks of them that return realistic data |
| Requests | Generated, each with its correct answer | Real or generated requests, each with a way to check the outcome |
| Reward | Check the booking against every calendar | Compare the agent's final action or tool calls with the correct ones |

Measure the base model first. Training needs requests where the base model succeeds sometimes and fails
sometimes; here it scored about 0.72 before training.
""")
md("""
## From this notebook to production

- **The agent code stays the same.** `run_agent` takes a `model_config`. In this notebook it points to the rollout
  gateway or to Amazon Bedrock; in production it points to the endpoint that serves the trained adapter. The agent
  itself can run on Amazon Bedrock AgentCore Runtime.
- **Tools become your APIs.** In production, the tools call your systems, for example as MCP tools behind Amazon
  Bedrock AgentCore Gateway. For training, use mocks or a sandbox copy of those systems that return realistic data,
  as this notebook does: training runs thousands of episodes, and every tool call must be safe to repeat.
- **Longer training runs** belong in a script rather than a notebook. The
  [AgentCore RL Toolkit](https://github.com/awslabs/agentcore-rl-toolkit) used here for the rollout gateway also
  supports rollouts from agents hosted on AgentCore Runtime.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "5b_scheduling_agent.ipynb"
nbf.write(nb, out)
print("wrote", out)
