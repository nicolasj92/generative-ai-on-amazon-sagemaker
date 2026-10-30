"""Run generated code in Amazon Bedrock AgentCore Code Interpreter.

`CodeInterpreterSandbox.run(request)` takes a SandboxFusion-style request
({"code": ..., "files": {name: base64}, "run_timeout": seconds}) and returns a
SandboxFusion-style response, so the cookbook's grading code can use it through
`sandbox_proxy.LocalSandbox`.

Each request runs in its own directory of one of a few long-lived sessions.
Create the code interpreter with network mode SANDBOX (no network access) and
no execution role (no AWS credentials inside).
"""

from __future__ import annotations

import base64
import queue
import shlex
import uuid

import boto3
from botocore.config import Config


class CodeInterpreterSandbox:
    def __init__(self, code_interpreter_id: str, region: str, sessions: int = 16):
        self.id = code_interpreter_id
        self.client = boto3.client("bedrock-agentcore", region_name=region,
                                   config=Config(read_timeout=120, max_pool_connections=sessions,
                                                 retries={"max_attempts": 8, "mode": "adaptive"}))
        self.sessions = queue.Queue()
        for _ in range(sessions):
            self.sessions.put(self._start())

    def _start(self) -> str:
        return self.client.start_code_interpreter_session(
            codeInterpreterIdentifier=self.id, sessionTimeoutSeconds=3 * 3600)["sessionId"]

    def _call(self, session_id: str, name: str, arguments: dict) -> dict:
        out = self.client.invoke_code_interpreter(codeInterpreterIdentifier=self.id, sessionId=session_id,
                                                  name=name, arguments=arguments)
        for event in out["stream"]:
            if "result" in event:
                return event["result"]
        return {"isError": True, "content": [{"text": "no result"}]}

    def run(self, request: dict) -> dict:
        work = f"job-{uuid.uuid4().hex}"
        timeout = float(request.get("run_timeout", 30))
        files = [{"path": f"{work}/main.py", "text": request["code"]}]
        files += [{"path": f"{work}/{name}", "blob": base64.b64decode(data)} for name, data in (request.get("files") or {}).items()]
        session_id = self.sessions.get()
        try:
            self._call(session_id, "writeFiles", {"content": files})
            result = self._call(session_id, "executeCommand", {
                "command": f"cd {work} && timeout {timeout:g} python3 main.py < /dev/null; "
                           f"code=$?; cd .. && rm -rf {shlex.quote(work)}; exit $code"})
        except self.client.exceptions.ResourceNotFoundException:
            self.sessions.put(self._start())  # session expired: replace it
            return {"status": "SandboxError", "message": "session expired", "run_result": None}
        except Exception as exc:  # noqa: BLE001
            self.sessions.put(session_id)
            return {"status": "SandboxError", "message": str(exc)[:500], "run_result": None}
        self.sessions.put(session_id)
        out = result.get("structuredContent") or {}
        code = out.get("exitCode")
        timed_out = code == 124  # exit code of `timeout`
        run = {"status": "TimeLimitExceeded" if timed_out else "Finished", "return_code": None if timed_out else code,
               "stdout": (out.get("stdout") or "").replace("\r\n", "\n"),
               "stderr": (out.get("stderr") or "").replace("\r\n", "\n"),
               "execution_time": out.get("executionTime")}
        return {"status": "Success" if code == 0 else "Failed", "message": "", "compile_result": None, "run_result": run}

    def close(self):
        while not self.sessions.empty():
            try:
                self.client.stop_code_interpreter_session(codeInterpreterIdentifier=self.id,
                                                          sessionId=self.sessions.get_nowait())
            except Exception:  # noqa: BLE001
                pass
