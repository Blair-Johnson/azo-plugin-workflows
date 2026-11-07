"""Recovery decisions must be based on durable results, not missing output."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import workflows as wf
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
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    monkeypatch.setattr(wf, "saved_checkpoint", lambda *a: SimpleNamespace(
        commit_id="saved-revision", document=checkpoint(complete=complete)))
    runtime.launcher.handles[1].status = "exited"
    runtime.controller.tick("miner")
    agent = store.agent("agt_reverse_1")
    if complete:
        assert len(runtime.launcher.requests) == 4
        assert runtime.launcher.requests[-1].source.checkpoint.revision_id == "saved-revision"
        runtime.controller.tick("miner")
        assert runtime.sent[-1][1] == "continue"
        assert not any(action == "message" for _, action, _ in runtime.sent)
    else:
        assert agent["state"] == "attention"
        assert "call_a" in agent["problem"] and "saved-revision" in agent["problem"]
        assert store.item("item_1")["taken_by"] == agent["id"]
        assert store.item("item_1")["state"] == "taken"
        assert not any(r.source.__class__.__name__ == "Resume" for r in runtime.launcher.requests)


@pytest.mark.parametrize("handoff", [False, True])
def test_pause_during_resume_does_not_wake_worker(runtime, monkeypatch, handoff):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    monkeypatch.setattr(wf, "saved_checkpoint", lambda *a: SimpleNamespace(
        commit_id="c1", document=checkpoint()))
    runtime.launcher.handles[1].status = "exited"
    runtime.controller.tick("miner")
    runtime.controller.pause("miner")
    runtime.sent.clear()
    if handoff:
        # Readiness rediscovery after the owner process has changed.
        runtime.controller.handles.clear()
        runtime.controller.target = lambda store, agent: store.update_agent(
            agent["id"], websocket_url="ws://ready")
    runtime.controller.tick("miner")
    assert store.agent("agt_reverse_1")["state"] == "paused"
    assert runtime.sent == []
    runtime.controller.run("miner")
    assert any(action == "continue" for _, action, _ in runtime.sent)
    assert not any(action == "message" for _, action, _ in runtime.sent)


@pytest.mark.parametrize("role,action", [("user", "continue"), ("assistant", "message")])
def test_resume_release_disposition(runtime, monkeypatch, role, action):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    document = {"core": {}, "entries": [{"messages": [{"role": role, "content": "hello"}]}]}
    monkeypatch.setattr(wf, "saved_checkpoint", lambda *a: SimpleNamespace(commit_id="c2", document=document))
    runtime.launcher.handles[1].status = "exited"
    runtime.controller.tick("miner")
    runtime.controller.tick("miner")
    assert runtime.sent[-1][1] == action
    assert store.agent("agt_reverse_1")["release"] is None


def test_release_failure_needs_attention(runtime, monkeypatch):
    from unittest.mock import Mock
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    monkeypatch.setattr(wf, "saved_checkpoint", lambda *a: SimpleNamespace(commit_id="c3", document=checkpoint()))
    runtime.launcher.handles[1].status = "exited"
    runtime.controller.tick("miner")
    runtime.controller.send = Mock(side_effect=OSError("transport failed"))
    runtime.controller.tick("miner")
    agent = store.agent("agt_reverse_1")
    assert agent["state"] == "attention" and "could not confirm recovery release" in agent["problem"]
    assert store.item("item_1")["state"] == "taken"


def test_paused_worker_without_websocket_is_continued_after_rediscovery(runtime):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    runtime.controller.pause("miner")
    store.update_agent("agt_reverse_1", websocket_url=None)
    runtime.controller.handles.clear()
    runtime.controller.target = lambda store, agent: store.update_agent(agent["id"], websocket_url="ws://ready")
    runtime.sent.clear()
    runtime.controller.run("miner")
    agent = store.agent("agt_reverse_1")
    assert agent["state"] == "running" and agent["release"] is None
    assert (agent["session_id"], "continue", "") in runtime.sent


def test_lost_ownership_before_release_keeps_starting(runtime, monkeypatch):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    monkeypatch.setattr(wf, "saved_checkpoint", lambda *a: SimpleNamespace(commit_id="c4", document=checkpoint()))
    runtime.launcher.handles[1].status = "exited"
    runtime.controller.tick("miner")
    agent = store.agent("agt_reverse_1")
    store.update_control(controller={"session_id": "new", "instance_id": "new-instance"}, epoch=99)
    runtime.controller.observe(store, agent, store.definition(), True)
    assert store.agent(agent["id"])["state"] == "starting"
    assert runtime.sent == []
