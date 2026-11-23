"""Recovery decisions must be based on durable results, not missing output."""
from types import SimpleNamespace

import pytest
import workflows as wf
from agent_zoo.events import ExternalSessionExited, ExternalSessionReady
from test_plugin import create, runtime


def checkpoint(*, complete=True):
    call = {"id": "call_a", "name": "terminal_run", "result": "done" if complete else None}
    messages = [{"role": "assistant", "tool_calls": [{"id": "call_a"}]}]
    if complete:
        messages.append({"role": "tool", "tool_call_id": "call_a", "content": "done"})
    return {"core": {"done": False}, "entries": [{"messages": messages, "tool_calls": [call]}]}


def test_checkpoint_classifier():
    assert wf.checkpoint_recovery(checkpoint())[0] == "continue"
    assert wf.checkpoint_recovery(checkpoint(complete=False))[0] == "uncertain"
    document = checkpoint()
    document["entries"][0]["messages"].pop()
    assert wf.checkpoint_recovery(document)[0] == "uncertain"
    document = checkpoint()
    document["entries"][0]["tool_calls"].append({"id": "call_b", "result": None})
    assert "call_b" in wf.checkpoint_recovery(document)[1]
    for bad in [None, {}, {"core": {}, "entries": []}, {"core": {}, "entries": [None]}]:
        assert wf.checkpoint_recovery(bad)[0] == "uncertain"
    document = checkpoint()
    document["core"]["done"] = True
    assert wf.checkpoint_recovery(document)[0] == "done"
    for role, expected in [("user", "continue"), ("assistant", "message")]:
        document = {"core": {}, "entries": [{"messages": [{"role": role, "content": "hi"}]}]}
        assert wf.checkpoint_recovery(document)[0] == expected


@pytest.mark.parametrize("complete", [True, False])
def test_crash_uses_durable_results(runtime, monkeypatch, complete):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    monkeypatch.setattr(wf, "saved_checkpoint", lambda *a: SimpleNamespace(
        commit_id="saved-revision", document=checkpoint(complete=complete)))
    runtime.crash("agt_reverse_1")
    runtime.drive(runtime.controller.tick("miner"))
    agent = store.agent("agt_reverse_1")
    if complete:
        assert len(runtime.spawns) == 4
        assert runtime.spawns[-1].source.kind == "resume"
        assert runtime.spawns[-1].source.revision_id == "saved-revision"
        runtime.drive(runtime.controller.tick("miner"))
        assert runtime.delivered[-1]["channel"] == "slash_command"
        assert runtime.delivered[-1]["payload"] == {"raw": "/continue"}
        assert not any(d["channel"] == "user_text" for d in runtime.delivered)
    else:
        assert agent["state"] == "attention"
        assert "call_a" in agent["problem"] and "saved-revision" in agent["problem"]
        assert store.item("item_1")["taken_by"] == agent["id"]
        assert store.item("item_1")["state"] == "taken"
        assert not any(r.source.kind == "resume" for r in runtime.spawns)


@pytest.mark.parametrize("handoff", [False, True])
def test_pause_during_resume_does_not_wake_worker(runtime, monkeypatch, handoff):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    monkeypatch.setattr(wf, "saved_checkpoint", lambda *a: SimpleNamespace(
        commit_id="c1", document=checkpoint()))
    runtime.crash("agt_reverse_1")
    runtime.drive(runtime.controller.tick("miner"))
    runtime.drive(runtime.controller.pause("miner"))
    runtime.delivered.clear()
    if handoff:
        # Readiness rediscovery after the owner process has changed.
        runtime.controller.instances_by_id.clear()
    runtime.drive(runtime.controller.tick("miner"))
    assert store.agent("agt_reverse_1")["state"] == "paused"
    assert runtime.delivered == []
    runtime.drive(runtime.controller.run("miner"))
    assert any(d["channel"] == "slash_command" and d["payload"] == {"raw": "/continue"} for d in runtime.delivered)
    assert not any(d["channel"] == "user_text" for d in runtime.delivered)


@pytest.mark.parametrize("role,expected_channel", [("user", "slash_command"), ("assistant", "user_text")])
def test_resume_release_disposition(runtime, monkeypatch, role, expected_channel):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    document = {"core": {}, "entries": [{"messages": [{"role": role, "content": "hello"}]}]}
    monkeypatch.setattr(wf, "saved_checkpoint", lambda *a: SimpleNamespace(commit_id="c2", document=document))
    runtime.crash("agt_reverse_1")
    runtime.drive(runtime.controller.tick("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    assert runtime.delivered[-1]["channel"] == expected_channel
    if expected_channel == "slash_command":
        assert runtime.delivered[-1]["payload"] == {"raw": "/continue"}
    else:
        assert "restarted from a checkpoint" in runtime.delivered[-1]["payload"]["text"]
    assert store.agent("agt_reverse_1")["release"] is None


def test_release_failure_needs_attention(runtime, monkeypatch):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    monkeypatch.setattr(wf, "saved_checkpoint", lambda *a: SimpleNamespace(commit_id="c3", document=checkpoint()))
    runtime.crash("agt_reverse_1")
    runtime.drive(runtime.controller.tick("miner"))
    agent = store.agent("agt_reverse_1")
    runtime.sim_driver.inject_delivery_outcome(agent["session_id"], "delivery_unknown", error="transport failed")
    runtime.drive(runtime.controller.tick("miner"))
    agent = store.agent("agt_reverse_1")
    assert agent["state"] == "attention" and "could not confirm recovery release" in agent["problem"]
    assert store.item("item_1")["state"] == "taken"


def test_paused_worker_without_websocket_is_continued_after_rediscovery(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    runtime.drive(runtime.controller.pause("miner"))
    store.update_agent("agt_reverse_1", websocket_url=None)
    runtime.controller.instances_by_id.clear()
    runtime.delivered.clear()
    runtime.drive(runtime.controller.run("miner"))
    agent = store.agent("agt_reverse_1")
    assert agent["state"] == "running" and agent["release"] is None
    assert any(
        d["session_id"] == agent["session_id"]
        and d["channel"] == "slash_command"
        and d["payload"] == {"raw": "/continue"}
        for d in runtime.delivered
    )


def test_lost_ownership_before_release_keeps_starting(runtime, monkeypatch):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    monkeypatch.setattr(wf, "saved_checkpoint", lambda *a: SimpleNamespace(commit_id="c4", document=checkpoint()))
    runtime.crash("agt_reverse_1")
    runtime.drive(runtime.controller.tick("miner"))
    agent = store.agent("agt_reverse_1")
    store.update_control(controller={"session_id": "new", "instance_id": "new-instance"}, epoch=99)
    runtime.drive(runtime.controller.observe(store, agent, store.definition(), True))
    assert store.agent(agent["id"])["state"] == "starting"
    assert runtime.delivered == []


def test_max_quick_resumes_stops_loop_and_marks_attention(runtime, monkeypatch):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    monkeypatch.setattr(wf, "saved_checkpoint", lambda *a: SimpleNamespace(commit_id="c5", document=checkpoint()))
    for attempt in range(wf.MAX_QUICK_RESUMES):
        agent = store.agent("agt_reverse_1")
        runtime.drive(runtime.controller.on_session_exited(
            ExternalSessionExited(session_id=agent["session_id"], instance_id=agent["instance_id"], returncode=1, confirmed=True)
        ))
        assert store.agent("agt_reverse_1")["resumes"] == attempt + 1
        restarted = store.agent("agt_reverse_1")
        sess = runtime.sim_driver.sessions[restarted["instance_id"]]
        runtime.drive(runtime.controller.on_session_ready(
            ExternalSessionReady(
                session_id=restarted["session_id"],
                instance_id=restarted["instance_id"],
                websocket_url=sess.websocket_url,
                process_identity=sess.process_identity.to_dict(),
            )
        ))
        assert store.agent("agt_reverse_1")["state"] == "running"
    # The (MAX_QUICK_RESUMES + 1)-th crash within 600s refuses further auto-resume.
    agent = store.agent("agt_reverse_1")
    spawns_before = len(runtime.spawns)
    runtime.drive(runtime.controller.on_session_exited(
        ExternalSessionExited(session_id=agent["session_id"], instance_id=agent["instance_id"], returncode=1, confirmed=True)
    ))
    assert len(runtime.spawns) == spawns_before
    stuck = store.agent("agt_reverse_1")
    assert stuck["state"] == "attention" and f"{wf.MAX_QUICK_RESUMES} times in a row" in stuck["problem"]
