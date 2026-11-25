"""Regressions from the live Luna smoke tests, verified on AgentZooSessionDriver & SimSessionDriver."""
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import workflows as wf
from agent_zoo.drivers.sessions import AgentZooSessionDriver
from agent_zoo.events import (
    ExternalSessionExited,
    SessionObservation,
    SessionReady,
    SessionSource,
    SessionSpec,
    SpawnSession,
)
from agent_zoo.host.launch import RuntimeHandle
from test_plugin import create, runtime, worker_state


@pytest.mark.parametrize("selection", [{}, {"agent_group": None}, {"agent_group": ""}])
def test_optional_group_omitted_null_or_empty(runtime, selection):
    store = create(runtime)
    runtime.drive(runtime.plugin.command_workflow(runtime.state, "miner", "run", **selection))
    runtime.drive(runtime.controller.tick("miner"))
    result = runtime.drive(runtime.plugin.message_agents(runtime.state, "miner", "hello", **selection))
    assert "agt_reverse_1" in result and "agt_discovery_1" in result
    runtime.drive(runtime.plugin.command_workflow(runtime.state, "miner", "pause", **selection))
    assert store.control()["state"] == "paused"


def test_optional_tool_fields_are_nullable(runtime):
    create(runtime)
    runtime.drive(runtime.controller.run("miner"))
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


def test_finished_worker_exit_does_not_relaunch(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    agent = store.agent("agt_reverse_1")
    store.finish_agent(agent["id"], summary="done", outputs=[])
    spawns_before = len(runtime.spawns)
    runtime.drive(runtime.controller.on_session_exited(
        ExternalSessionExited(session_id=agent["session_id"], instance_id=agent["instance_id"], returncode=0, confirmed=True)
    ))
    assert store.agent("agt_reverse_1")["state"] == "finished"
    assert len(runtime.spawns) == spawns_before
    assert agent["instance_id"] not in runtime.controller.instances_by_id


@pytest.mark.parametrize("end", ["archive", "handoff"])
def test_exit_events_ignored_after_scheduling_ends(runtime, end):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    agent = store.agent("agt_reverse_1")
    if end == "archive":
        runtime.drive(runtime.controller.archive("miner"))
    else:
        store.update_control(controller={"session_id": "other", "instance_id": "o"}, epoch=99)
        runtime.drive(runtime.controller.tick("miner"))
    assert "miner" not in runtime.controller.managed
    spawns_before = len(runtime.spawns)
    runtime.drive(runtime.controller.on_session_exited(
        ExternalSessionExited(session_id=agent["session_id"], instance_id=agent["instance_id"], returncode=1, confirmed=True)
    ))
    assert len(runtime.spawns) == spawns_before


def test_instances_are_isolated_between_workflows(runtime):
    create(runtime)
    runtime.drive(runtime.plugin.create_workflow(runtime.state, "other", "buf_def"))
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.run("other"))
    assert len(runtime.controller.instances_by_id) == 4
    runtime.drive(runtime.controller.tick("other"))
    assert len(runtime.controller.instances_by_id) == 4
    assert {pair[0] for pair in runtime.controller.instances_by_id.values()} == {"miner", "other"}


def test_unknown_observation_never_treats_worker_as_dead(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    runtime.hooks["observe_override"] = lambda req: SessionObservation(
        session_id=req.session_id, instance_id=req.instance_id, status="unknown", alive=None
    )
    spawns_before = len(runtime.spawns)
    runtime.drive(runtime.controller.tick("miner"))
    assert len(runtime.spawns) == spawns_before
    assert store.agent("agt_reverse_1")["state"] == "running"


def test_pool_shrinks_without_stopping_current_work(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    runtime.drive(runtime.plugin.set_agent_count(runtime.state, "miner", "grp_reverse", 1))
    assert len(runtime.spawns) == 3
    assert not runtime.delivered
    store.finish_agent("agt_reverse_1", summary="done", outputs=[])
    assert runtime.drive(runtime.controller.tick("miner")) == []
    assert store.item("item_3")["state"] == "waiting"
    store.finish_agent("agt_reverse_2", summary="done", outputs=[])
    assert len(runtime.drive(runtime.controller.tick("miner"))) == 1


def test_second_controller_adopts_live_pool_without_duplicates(runtime):
    store = create(runtime)
    old = runtime.controller
    runtime.drive(old.run("miner"))
    runtime.drive(old.tick("miner"))
    state = SimpleNamespace(**vars(runtime.state))
    state._session_id, state._instance_id = "other", "other-instance"
    new = wf.WorkflowController(old.config)
    new.bind(state)
    assert "Took control" in runtime.drive(new.run("miner"))
    assert len(runtime.spawns) == 3
    assert runtime.drive(old.tick("miner")) == []
    runtime.drive(new.change_setting("miner", "group", "grp_reverse", "count", "3"))
    assert len(runtime.spawns) == 4
    assert runtime.drive(old.start_agents(store, store.definition(), set())) == []


def test_bundled_skill_precedes_preserved_config(runtime):
    registry = SimpleNamespace(add_sources=Mock())
    wf.WorkflowSkillRegistration().handle(
        SessionReady(),
        SimpleNamespace(skill_registry=registry, _session_kind="default"),
        lambda _ev: None,
    )
    sources = registry.add_sources.call_args.args[0]
    assert Path(sources[0].path).resolve() == Path(wf.__file__).resolve().parents[1] / "skills/workflows/SKILL.md"


def test_session_driver_reaps_a_real_child_process(tmp_path):
    """AgentZooSessionDriver owns subprocess reaping and emits ExternalSessionExited."""
    exited = threading.Event()
    events = []
    procs = []

    class QuickLauncher:
        def prepare(self, request, cwd=None, env=None):
            return SimpleNamespace(request=request, cwd=cwd, env=env)

        def start(self, plan):
            proc = subprocess.Popen([sys.executable, "-c", "pass"])
            procs.append(proc)
            return RuntimeHandle(plan.request, proc, str(tmp_path / "ready.json"), "", "")

    def on_event(ev):
        events.append(ev)
        if isinstance(ev, ExternalSessionExited):
            exited.set()

    driver = AgentZooSessionDriver(launcher=QuickLauncher(), poll_interval_s=0.01)
    driver.start(on_event)
    try:
        driver.submit(
            "op_spawn_1",
            SpawnSession(
                session_id="wf-worker-1",
                instance_id="wf-inst-1",
                source=SessionSource(kind="fresh"),
                spec=SessionSpec(project="test", workdir=str(tmp_path), kind="wf:miner:agt_discovery_1"),
                wait_ready=False,
                watch=False,
            ),
        )
        assert exited.wait(timeout=5.0)
        assert procs and procs[0].returncode == 0
        exit_ev = next(ev for ev in events if isinstance(ev, ExternalSessionExited))
        assert exit_ev.session_id == "wf-worker-1" and exit_ev.instance_id == "wf-inst-1"
        assert exit_ev.returncode == 0 and exit_ev.confirmed is True
    finally:
        driver.close()
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)


def test_archive_returns_attention_items(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    store.update_agent("agt_reverse_1", state="attention", problem="unresolved result")
    runtime.crash("agt_reverse_1")
    runtime.drive(runtime.controller.archive("miner"))
    assert store.agent("agt_reverse_1")["state"] == "stopped"
    assert store.item("item_1")["state"] == "waiting"
