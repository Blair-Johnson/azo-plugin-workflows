"""Opt-in Luna live test: real controller handoff, resizing, recovery and cleanup."""
import argparse
import asyncio
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import workflows as wf
from agent_zoo.runtime_launch import RuntimeLauncher, RuntimeLaunch, RuntimeSettings, InstanceRef


async def wait_for(label, predicate, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            print("PASS", label, flush=True)
            return result
        await asyncio.sleep(0.2)
    raise AssertionError("Timed out: " + label)


async def connect(handle):
    observation = await wait_for("controller ready", lambda: (
        result if (result := handle.poll()).status == "ready" else None))
    client = wf.AzoWs.attach(observation.ready["websocket_url"], terminate_on_close=False)
    await client.connect()
    hello = client.observation().hello
    assert hello["session_id"] == handle.ref.session_id
    assert hello["instance_id"] == handle.ref.instance_id
    return client


async def command(client, text, predicate):
    await client.slash(text, client_id="workflow-control-smoke")
    return await wait_for(text, predicate)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true", required=True,
                        help="Launch real Luna sessions (incurs model usage)")
    parser.parse_args()
    name = "fixes_" + uuid.uuid4().hex[:8]
    folder = Path("/tmp") / name
    folder.mkdir()
    store = wf.WorkflowStore(name)
    handles, clients = [], []
    launcher = RuntimeLauncher()
    settings = RuntimeSettings(project="workflow-plugin-tests", workdir=str(folder),
                               state_root=str(wf.resolve_state_root()), model="gpt-5.6-luna-max")
    try:
        await exercise(name, folder, store, handles, clients, launcher, settings)
    finally:
        # Only exact workers created by this test, never pre-existing workflows.
        if store.exists():
            with store.lock():
                if store.control().get("state") != "archived":
                    store.update_control(state="paused")
            for agent in store.agents():
                if wf.process_evidence(agent.get("process_identity")) == "alive" and agent.get("websocket_url"):
                    try:
                        await wf.send_to_agent(agent, "shutdown")
                    except Exception as exc:
                        print("CLEANUP WARNING", agent["id"], str(exc), flush=True)
        for client in clients:
            await client.close()
        for handle in handles:
            if handle.poll().status != "exited":
                target = handle.poll().ready
                if target:
                    await wf.send_to_agent(target, "shutdown")
            try:
                await asyncio.to_thread(handle.process.wait, timeout=15)
            except subprocess.TimeoutExpired:
                handle.process.terminate()
                await asyncio.to_thread(handle.process.wait, timeout=10)
        print("Evidence:", store.dir, flush=True)


# Exercise is defined below so cleanup remains centralized.
async def exercise(name, folder, store, handles, clients, launcher, settings):
    definition = f"""folder: {folder}
model: gpt-5.6-luna-max
queues:
  q_input: {{}}
  q_output: {{}}
  q_crash: {{}}
agents:
  grp_work:
    count: 1
    takes_from: q_input
    sends_to: [q_output]
    instructions: |
      This is a bounded smoke test. Run sleep 20 in a headless terminal once.
      Wait for it to finish, then put the item title in a buffer and finish_task
      with that buffer sent to q_output. No other work or subagents.
      {{{{item}}}}
  grp_crash:
    count: 0
    takes_from: q_crash
    sends_to: []
    instructions: |
      This is a crash-recovery test. Call await_interrupts with timeout 20 before
      doing anything else, then call finish_task with no outputs. No other work.
"""
    store.create(definition, by="live-control-smoke")
    for title in ["alpha", "beta", "gamma"]:
        store.add_item("q_input", title, by="live-control-smoke")
    for number in [1, 2]:
        request = RuntimeLaunch(
            target=InstanceRef.new_session(), settings=settings, startup_mode="paused",
            title=f"workflow-test-controller-{number}", lifetime="independent",
            startup_system_message="You are a bounded workflow test controller. "
            "Workflow command results are already done. Acknowledge notices without tools. "
            "Do not change workflows unless an explicit later user request asks you to archive one.")
        handle = launcher.start(launcher.prepare(request))
        handles.append(handle)
        clients.append(await connect(handle))
    a, b = clients
    await command(a, f"/workflow run {name}", lambda: len(store.agents("grp_work")) == 1)
    await wait_for("first worker ready", lambda: store.agents("grp_work")[0]["state"] == "running")
    await command(a, f"/workflow set {name}:grp_work count 2",
                  lambda: len(store.agents("grp_work")) == 2)
    await wait_for("two workers ready", lambda: all(
        agent["state"] == "running" for agent in store.agents("grp_work")))
    identities = {agent["id"]: agent["instance_id"] for agent in store.agents("grp_work")}
    await command(b, f"/workflow run {name}", lambda: (
        store.control()["controller"]["session_id"] == handles[1].ref.session_id))
    assert {agent["id"]: agent["instance_id"] for agent in store.agents("grp_work")} == identities
    print("PASS handoff preserved both live worker instances", flush=True)
    await command(b, f"/workflow set {name}:grp_work count 1", lambda: (
        store.definition()["agents"]["grp_work"]["count"] == 1))
    assert all(agent["state"] == "running" for agent in store.agents("grp_work"))
    print("PASS pool shrink kept both current workers running", flush=True)
    await wait_for("all three items completed", lambda: len(store.items("q_output")) == 3, timeout=240)
    assert len(store.agents("grp_work")) == 3
    assert all(agent["model"] == "gpt-5.6-luna-max" for agent in store.agents())

    def workers_reaped():
        for agent in store.agents("grp_work"):
            result = subprocess.run(["ps", "-o", "stat=", "-p", str(agent["pid"])],
                                    capture_output=True, text=True)
            if result.stdout.strip():
                return False
        return True
    await wait_for("workers reaped after completion and handoff", workers_reaped, timeout=30)
    await check_crash(name, store, b)
    await b.send_user(f"Please call command_workflow to archive only workflow {name}. "
                      "Pass agent_group as null. This is the final explicit test request.")
    await wait_for("archive through model-facing tool with null group", lambda: (
        store.control()["state"] == "archived"), timeout=120)
    assert store.items("q_crash")[0]["state"] == "done"
    print("PASS model-facing archive accepted null group; all live checks complete", flush=True)

async def check_crash(name, store, client):
    store.add_item("q_crash", "safe checkpoint interruption", by="live-control-smoke")
    await command(client, f"/workflow set {name}:grp_crash count 1", lambda: (
        len(store.agents("grp_crash")) == 1))

    def complete_checkpoint():
        agent = store.agents("grp_crash")[0]
        if agent["state"] != "running":
            return None
        try:
            saved = wf.saved_checkpoint(agent["project"], agent["session_id"])
        except (FileNotFoundError, wf.WorkflowError):
            return None
        calls = [call for entry in saved.document.get("entries", [])
                 for call in entry.get("tool_calls", [])]
        action, _ = wf.checkpoint_recovery(saved.document)
        return agent if action == "continue" and any(
            call.get("name") == "await_interrupts" for call in calls) else None

    agent = await wait_for("complete saved tool result before crash", complete_checkpoint, timeout=60)
    identity = wf.capture_process_identity(agent["pid"])
    assert identity and identity.to_dict() == agent["process_identity"]
    os.kill(agent["pid"], signal.SIGKILL)
    await wait_for("safe checkpoint relaunched", lambda: (
        store.agent(agent["id"])["resumes"] == 1), timeout=30)
    await wait_for("resumed worker finished", lambda: (
        store.agent(agent["id"])["state"] == "finished"), timeout=120)
    after = store.agent(agent["id"])
    assert after["instance_id"] != agent["instance_id"]
    saved = wf.saved_checkpoint(after["project"], after["session_id"])
    calls = [call for entry in saved.document.get("entries", []) for call in entry.get("tool_calls", [])]
    assert sum(call.get("name") == "await_interrupts" for call in calls) == 1
    assert store.item(after["item"])["state"] == "done"
    print("PASS safe checkpoint resumed without repeating the tool call", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
