"""Regressions from the live Luna smoke tests."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import workflows as wf
from test_plugin import create, runtime, worker_state


@pytest.mark.parametrize("selection", [{}, {"agent_group": None}, {"agent_group": ""}])
def test_optional_group_omitted_null_or_empty(runtime, selection):
    store = create(runtime)
    runtime.plugin.command_workflow(runtime.state, "miner", "run", **selection)
    runtime.controller.tick("miner")
    result = runtime.plugin.message_agents(runtime.state, "miner", "hello", **selection)
    assert "agt_reverse_1" in result and "agt_discovery_1" in result
    runtime.plugin.command_workflow(runtime.state, "miner", "pause", **selection)
    assert store.control()["state"] == "paused"


def test_optional_tool_fields_are_nullable(runtime):
    create(runtime)
    runtime.controller.run("miner")
    tools = wf.build_tools(runtime.plugin)
    for state in [runtime.state, worker_state(runtime, "agt_reverse_1")]:
        state.tool_schemas = {}
        for tool in tools:
            tool._publish(state)
        for published in state.tool_schemas.values():
            if not published["available"]:
                continue
            schema = published["schema"]["function"]["parameters"]
            for key, prop in schema["properties"].items():
                if key not in schema["required"]:
                    assert "null" in prop["type"], key
                else:
                    assert "null" not in prop["type"], key


def test_finished_worker_retained_until_exit(runtime):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    agent = store.agent("agt_reverse_1")
    key = runtime.controller.handle_key(store, agent)
    handle = runtime.controller.handles[key]
    store.finish_agent(agent["id"], summary="done", outputs=[])
    runtime.controller.tick("miner")
    assert runtime.controller.handles[key] is handle
    handle.status = "exited"
    runtime.controller.tick("miner")
    assert key not in runtime.controller.handles


@pytest.mark.parametrize("end", ["archive", "handoff"])
def test_reap_after_scheduling_ends(runtime, end):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    if end == "archive":
        runtime.controller.archive("miner")
    else:
        store.update_control(controller={"session_id": "other", "instance_id": "o"}, epoch=99)
        runtime.controller.tick("miner")
    assert "miner" not in runtime.controller.managed
    assert len(runtime.controller.handles) == 3
    for handle in runtime.launcher.handles:
        handle.status = "exited"
    runtime.controller.reap_children()
    assert runtime.controller.handles == {}


def test_handles_are_isolated_between_workflows(runtime):
    create(runtime)
    runtime.plugin.create_workflow(runtime.state, "other", "buf_def")
    runtime.controller.run("miner")
    runtime.controller.run("other")
    assert len(runtime.controller.handles) == 4
    runtime.controller.tick("other")
    assert len(runtime.controller.handles) == 4
    assert {key[0] for key in runtime.controller.handles} == {"miner", "other"}


def test_unknown_poll_keeps_process_handle(runtime):
    create(runtime)
    runtime.controller.run("miner")
    for handle in runtime.launcher.handles:
        handle.poll = Mock(side_effect=OSError("unknown"))
    runtime.controller.reap_children()
    assert len(runtime.controller.handles) == 3


def test_pool_shrinks_without_stopping_current_work(runtime):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    runtime.plugin.set_agent_count(runtime.state, "miner", "grp_reverse", 1)
    assert len(runtime.launcher.requests) == 3
    assert not runtime.sent
    store.finish_agent("agt_reverse_1", summary="done", outputs=[])
    assert runtime.controller.tick("miner") == []
    assert store.item("item_3")["state"] == "waiting"
    store.finish_agent("agt_reverse_2", summary="done", outputs=[])
    assert len(runtime.controller.tick("miner")) == 1


def test_second_controller_adopts_live_pool_without_duplicates(runtime):
    store = create(runtime)
    old = runtime.controller
    old.run("miner")
    old.tick("miner")
    state = SimpleNamespace(**vars(runtime.state))
    state._session_id, state._instance_id = "other", "other-instance"
    new = wf.WorkflowController(old.config)
    new.bind(state)
    assert "Took control" in new.run("miner")
    assert len(runtime.launcher.requests) == 3
    assert old.tick("miner") == []
    new.change_setting("miner", "group", "grp_reverse", "count", "3")
    assert len(runtime.launcher.requests) == 4
    assert old.start_agents(store, store.definition(), set()) == []


def test_bundled_skill_precedes_preserved_config(runtime):
    from pathlib import Path
    registry = SimpleNamespace(add_sources=Mock())
    wf.WorkflowSkillRegistration()(SimpleNamespace(skill_registry=registry, _session_kind="default"))
    source = registry.add_sources.call_args.args[0]
    assert Path(source.path).resolve() == Path(wf.__file__).resolve().parents[1] / "skills/workflows/SKILL.md"


def test_reaps_a_real_child_process(runtime):
    import subprocess
    import sys
    import time
    from agent_zoo.runtime_launch import RuntimeHandle
    store = create(runtime)
    runtime.controller.run("miner")
    request = runtime.launcher.requests[0]
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    key = (store.name, "agt_discovery_1", request.target.instance_id)
    handle = RuntimeHandle(request, process, str(runtime.tmp / "ready"), "", "")
    runtime.controller.handles = {key: handle}
    try:
        deadline = time.monotonic() + 5
        while runtime.controller.handles and time.monotonic() < deadline:
            runtime.controller.reap_children()
            time.sleep(0.01)
        assert process.returncode == 0
        assert not runtime.controller.handles
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


def test_archive_returns_attention_items(runtime):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    store.update_agent("agt_reverse_1", state="attention", problem="unresolved result")
    runtime.launcher.handles[1].status = "exited"
    runtime.controller.archive("miner")
    assert store.agent("agt_reverse_1")["state"] == "stopped"
    assert store.item("item_1")["state"] == "waiting"
