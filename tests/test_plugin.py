"""Plugin surface, scheduling, control and recovery via unified SessionRequest / SimSessionDriver."""
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import warnings

import pytest
from agent_utils.core.events import IoCompleted, IoFailed, RenderRequested, RunToolCall, TimerFired, TurnSettled
from agent_utils.core.io import IoTable, StartTimer, SubmitOp
from agent_utils.core.session import Session
from agent_utils.core.state import Entry, ToolCall
from agent_utils.core.tool import Tool
from agent_utils.runtime.simulation import SimulationRuntime
from agent_zoo.drivers.sessions import SimSessionDriver
from agent_zoo.events import (
    AdoptSession,
    DiscoverSessions,
    EmitNotice,
    ExternalSessionExited,
    ExternalSessionReady,
    ObserveSession,
    SaveCheckpoint,
    SendSessionData,
    SendSessionInput,
    SessionArtifact,
    SessionDataReceived,
    SessionObservation,
    SessionReady,
    SessionRequest,
    SlashCommand,
    SpawnSession,
    StopSession,
    SuspendRequested,
)
from agent_zoo.features.commands import CommandError

import workflows as wf

DEFINITION = """\
queues:
  q_targets: {}
  q_catalog: { notify_on_new: true }
agents:
  grp_discovery:
    count: 1
    every: 30m
    sends_to: [q_targets]
    instructions: Find checkpoints for {{workflow_name}}.
  grp_reverse:
    count: 2
    takes_from: q_targets
    sends_to: [q_catalog]
    instructions: |
      Reverse engineer this.
      {{item}}
"""


class Buffers:
    def __init__(self, **texts):
        self.texts = texts

    def resolve_for_read(self, state, buffer_id):
        if buffer_id not in self.texts:
            raise KeyError(buffer_id)
        return SimpleNamespace(text=self.texts[buffer_id])

    def register_special_buffer_namespace(self, prefix, provider, replace=False):
        self.namespace = (prefix, provider)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(wf, "saved_model_config", lambda config: None)
    sim_session = Session(system_prompt="base", token_budget=120000)
    sim = SimulationRuntime(sim_session)
    sim_driver = SimSessionDriver(default_delay=0.0)
    sim_driver.register(sim)

    spawns: list[SpawnSession] = []
    requests_log: list[SessionRequest] = []
    hooks = {"spawn_error": None, "observe_override": None}

    def execute_session_request(req: SessionRequest):
        requests_log.append(req)
        if isinstance(req, SpawnSession):
            spawns.append(req)
            if hooks["spawn_error"] is not None:
                return SimpleNamespace(error=hooks["spawn_error"], value=None)
        if isinstance(req, ObserveSession) and hooks["observe_override"] is not None:
            res = hooks["observe_override"](req)
            if res is not None:
                return SimpleNamespace(error=None, value=res)
        return sim_driver.handle("sim_op", req, sim)

    def drive(gen):
        if not inspect.isgenerator(gen):
            return gen
        send_val, throw_exc = None, None
        while True:
            try:
                req = gen.throw(throw_exc) if throw_exc is not None else gen.send(send_val)
            except StopIteration as stop:
                return stop.value
            send_val, throw_exc = None, None
            if isinstance(req, SessionRequest):
                outcome = execute_session_request(req)
                if outcome.error is not None:
                    throw_exc = outcome.error
                else:
                    send_val = outcome.value
            else:
                raise AssertionError(f"Unexpected IoRequest: {req!r}")

    def crash(agent_id: str, workflow_name: str = "miner", returncode: int = 1):
        agent = wf.WorkflowStore(workflow_name).agent(agent_id)
        sim_driver.exit_session(sim, agent["instance_id"], returncode=returncode)

    state = SimpleNamespace(
        _session_id="parent", _instance_id="parent-a", _session_kind="default",
        _agent_zoo_context={"project_name": "test", "project_store_root": str(tmp_path / "registry")},
        session_cwd=str(tmp_path / "work"), _persistence_local_root=str(tmp_path / "local"),
        buffer_manager=Buffers(buf_def=DEFINITION, buf_a="# ResNet50\nhttps://a", buf_b="BERT", buf_c="GPT-2"),
        pending_interrupts=[],
        io=IoTable(),
    )
    controller = wf.WorkflowController({"llm": {"default": "offline", "models": {
        "offline": {"model": "a"}, "big": {"model": "b"}, "retired": {"model": "c", "enabled": False}}}})
    plugin = wf.WorkflowsPlugin(controller)
    controller.bind(state)
    return SimpleNamespace(
        state=state,
        controller=controller,
        plugin=plugin,
        sim=sim,
        sim_driver=sim_driver,
        spawns=spawns,
        requests=requests_log,
        delivered=sim_driver.delivered_inputs,
        hooks=hooks,
        drive=drive,
        execute_session_request=execute_session_request,
        crash=crash,
        tmp=tmp_path,
    )


def create(r):
    r.plugin.create_workflow(r.state, "miner", "buf_def")
    r.plugin.add_items(r.state, "miner", "q_targets", ["buf_a", "buf_b", "buf_c"])
    return wf.WorkflowStore("miner")


def test_create_reports_and_does_not_run(runtime):
    text = runtime.plugin.create_workflow(runtime.state, "miner", "buf_def")
    assert "Nothing is running yet" in text and "/workflow run miner" in text
    assert runtime.spawns == []
    with pytest.raises(wf.WorkflowError, match="no buffer"):
        runtime.plugin.create_workflow(runtime.state, "other", "buf_missing")


def test_run_starts_agents_with_rendered_instructions(runtime):
    store = create(runtime)
    text = runtime.drive(runtime.controller.run("miner"))
    assert "Started agt_discovery_1." in text
    assert "Started agt_reverse_1 on item_1 from q_targets." in text
    assert "Started agt_reverse_2 on item_2 from q_targets." in text
    requests = {r.spec.kind: r for r in runtime.spawns}
    reverse = requests["wf:miner:agt_reverse_1"]
    assert reverse.source.kind == "fresh"
    assert reverse.spec.lifetime == "independent" and reverse.spec.startup_mode == "normal"
    assert "Reverse engineer this." in reverse.spec.first_user_message
    assert "item_1 from q_targets: ResNet50" in reverse.spec.first_user_message
    assert requests["wf:miner:agt_discovery_1"].spec.first_user_message == "Find checkpoints for miner."
    assert reverse.spec.model == "offline" and reverse.spec.project == "test"
    assert ("AZO_WORKFLOW", "miner") in reverse.spec.extra_env
    assert store.item("item_1")["taken_by"] == "agt_reverse_1"
    assert store.control()["controller"]["session_id"] == "parent"
    # Independent workflow workers opt into leave-on-parent-close lifecycle policy.
    assert runtime.sim_driver.session_policies[store.agent("agt_reverse_1")["session_id"]] == "leave"
    # Ready agents become running; the standing group respects `every`.
    runtime.drive(runtime.controller.tick("miner"))
    assert store.agent("agt_reverse_1")["state"] == "running"
    assert len(runtime.spawns) == 3


def test_finished_agent_frees_a_slot(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    store.finish_agent("agt_reverse_1", summary="done", outputs=[("q_catalog", "graph")])
    started = runtime.drive(runtime.controller.tick("miner"))
    assert started == ["Started agt_reverse_3 on item_3 from q_targets."]
    assert any("new items in q_catalog: item_4 (graph)" in n for n in runtime.controller.notices)


PAIRS = """\
queues:
  q_a: {}
  q_b: {}
  q_out: {}
agents:
  grp_pair:
    count: 2
    takes_from: { q_a: 2, q_b: 1 }
    sends_to: [q_out]
    instructions: Compare these.
"""


def create_pairs(r):
    r.state.buffer_manager.texts.update(buf_pairs=PAIRS, buf_a1="A one", buf_a2="A two", buf_a3="A three",
                                        buf_b1="B one")
    r.plugin.create_workflow(r.state, "pairs", "buf_pairs")
    r.plugin.add_items(r.state, "pairs", "q_a", ["buf_a1", "buf_a2", "buf_a3"])
    return wf.WorkflowStore("pairs")


def test_batch_group_waits_for_a_full_batch(runtime):
    store = create_pairs(runtime)
    text = runtime.drive(runtime.controller.run("pairs"))
    assert "No agents started yet: q_b is empty, so grp_pair has nothing to take." in text
    assert "grp_pair will take them." in runtime.plugin.add_items(runtime.state, "pairs", "q_b", ["buf_b1"])
    assert runtime.drive(runtime.controller.tick("pairs")) == [
        "Started agt_pair_1 on item_1, item_2 from q_a and item_4 from q_b."
    ]
    message = runtime.spawns[-1].spec.first_user_message
    assert all(f"{i} from q_" in message for i in ("item_1", "item_2", "item_4")) and "item_3" not in message
    assert store.agent("agt_pair_1")["items"] == ["item_1", "item_2", "item_4"]
    assert runtime.controller.idle_reason(store, None) == (
        "grp_pair takes 2 items from q_a at a time and q_a has 1 waiting; q_b is empty, so grp_pair has nothing "
        "to take.")
    text = runtime.drive(runtime.controller.restart_agent("pairs", "agt_pair_1"))
    assert "item_1, item_2, item_4 are back in their queues" in text
    assert all(m["state"] == "waiting" for m in store.items())
    stop_reqs = [r for r in runtime.requests if isinstance(r, StopSession)]
    assert stop_reqs and stop_reqs[-1].mode == "graceful_ws" and stop_reqs[-1].save is True


def test_batch_returns_on_archive_and_launch_failure(runtime):
    store = create_pairs(runtime)
    runtime.plugin.add_items(runtime.state, "pairs", "q_b", ["buf_b1"])
    runtime.drive(runtime.controller.run("pairs"))
    assert store.agent("agt_pair_1")["items"] == ["item_1", "item_2", "item_4"]
    runtime.drive(runtime.controller.archive("pairs"))
    assert all(m["state"] == "waiting" and m["taken_by"] is None for m in store.items())
    runtime.controller.unarchive("pairs")
    runtime.hooks["spawn_error"] = OSError("boom")
    runtime.drive(runtime.controller.run("pairs"))
    assert store.agent("agt_pair_2")["state"] == "stopped" and "grp_pair" in store.control()["paused_groups"]
    assert all(m["state"] == "waiting" and m["taken_by"] is None for m in store.items())


def test_batch_worker_outputs_record_every_origin(runtime):
    store = create_pairs(runtime)
    runtime.plugin.add_items(runtime.state, "pairs", "q_b", ["buf_b1"])
    runtime.drive(runtime.controller.run("pairs"))
    worker = SimpleNamespace(_session_kind="wf:pairs:agt_pair_1", _session_id="w", _instance_id="w1",
                             buffer_manager=Buffers(buf_early="early", buf_late="late"), pending_interrupts=[],
                             entries=[])
    plugin = wf.WorkflowsPlugin(wf.WorkflowController({}))
    plugin.worker_add_items(worker, "q_out", ["buf_early"])
    plugin.finish_task(worker, "Compared", [{"buffer": "buf_late", "queue": "q_out"}])
    batch = ["item_1", "item_2", "item_4"]
    assert [store.item(i)["from_items"] for i in ("item_5", "item_6")] == [batch, batch]
    assert [store.item(i)["state"] for i in batch] == ["done"] * 3 and store.item("item_3")["state"] == "waiting"


def test_legacy_single_item_records(runtime):
    """Agent and item records written before batches use ``item`` and ``from_item``."""
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    for agent_id in ("agt_reverse_1", "agt_reverse_2"):
        record = store.agent(agent_id)
        record["item"] = record.pop("items")[0]
        wf.write_json(store.agent_path(agent_id), record)
    meta = store.add_item("q_catalog", "old result", by="agt_reverse_1")
    meta.pop("from_items")
    meta["from_item"] = "item_1"
    wf.write_json(store.item_paths(meta["id"])[1], meta)
    assert "agt_reverse_1 on item_1" in wf.render_group(store, "grp_reverse")
    assert "while working on item_1" in wf.render_item(store, meta["id"])
    assert "(from `item_1`)" in wf.render_queue_md(store, "q_catalog")[1]
    out = store.finish_agent("agt_reverse_1", summary="done", outputs=[("q_catalog", "new result")])
    assert store.item("item_1")["state"] == "done" and out["created"][0]["from_items"] == ["item_1"]
    assert "item_2 is back in q_targets" in runtime.drive(runtime.controller.restart_agent("miner", "agt_reverse_2"))
    assert store.item("item_2")["state"] == "waiting"


def test_pause_then_run_resumes(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    text = runtime.drive(runtime.controller.pause("miner", "grp_reverse"))
    assert "Interrupted agt_reverse_1, agt_reverse_2" in text
    assert store.agent("agt_reverse_1")["state"] == "paused"
    assert [(d["channel"], d["payload"]) for d in runtime.delivered] == [
        ("slash_command", {"raw": "/interrupt"}),
        ("slash_command", {"raw": "/interrupt"}),
    ]
    assert runtime.drive(runtime.controller.tick("miner")) == []  # group paused: no replacements
    text = runtime.drive(runtime.controller.run("miner", "grp_reverse"))
    assert "Resumed agt_reverse_1, agt_reverse_2" in text
    assert store.agent("agt_reverse_2")["state"] == "running"
    assert [(d["channel"], d["payload"]) for d in runtime.delivered[-2:]] == [
        ("slash_command", {"raw": "/continue"}),
        ("slash_command", {"raw": "/continue"}),
    ]


def test_crashed_agent_resumes_from_checkpoint(runtime, monkeypatch):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    monkeypatch.setattr(wf, "saved_checkpoint", lambda project, sid, revision=None: SimpleNamespace(
        commit_id="c1", document={"core": {"done": False}, "entries": [
            {"messages": [{"role": "user", "content": "Continue the task"}], "tool_calls": []}]}))
    runtime.crash("agt_reverse_1")
    runtime.drive(runtime.controller.tick("miner"))
    resumed = runtime.spawns[-1]
    assert resumed.spec.kind == "wf:miner:agt_reverse_1" and resumed.spec.startup_mode == "paused"
    assert resumed.source.kind == "resume" and resumed.source.revision_id == "c1"
    assert store.agent("agt_reverse_1")["resumes"] == 1
    runtime.drive(runtime.controller.tick("miner"))
    assert runtime.delivered[-1]["channel"] == "slash_command"
    assert runtime.delivered[-1]["payload"] == {"raw": "/continue"}
    assert not any(d["channel"] == "user_text" for d in runtime.delivered)
    assert store.agent("agt_reverse_1")["state"] == "running"


def test_missing_checkpoint_needs_attention(runtime, monkeypatch):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))

    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(wf, "saved_checkpoint", missing)
    runtime.crash("agt_reverse_1")
    runtime.drive(runtime.controller.tick("miner"))
    agent = store.agent("agt_reverse_1")
    assert agent["state"] == "attention" and "restart_agent" in agent["problem"]
    assert any("agt_reverse_1 needs attention" in n for n in runtime.controller.notices)
    text = runtime.drive(runtime.controller.restart_agent("miner", "agt_reverse_1"))
    assert "item_1 is back in q_targets" in text
    assert store.item("item_1")["state"] == "waiting"


def test_launch_failure_pauses_group(runtime):
    store = create(runtime)
    runtime.hooks["spawn_error"] = OSError("boom")
    runtime.drive(runtime.controller.run("miner"))
    assert "grp_reverse" in store.control()["paused_groups"]
    assert store.item("item_1")["state"] == "waiting"
    assert any("could not start an agent in grp_reverse: boom" in n for n in runtime.controller.notices)


def test_losing_control_stops_scheduling(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    store.update_control(controller={"session_id": "other", "instance_id": "o"}, epoch=99)
    assert runtime.drive(runtime.controller.tick("miner")) == []
    assert "miner" not in runtime.controller.managed
    assert any("controlled by session other" in n for n in runtime.controller.notices)


def test_adopt_after_restart_and_discover_rediscovery(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    # Simulate worker reload onto a new instance_id while parent was restarting.
    old_agent = store.agent("agt_reverse_1")
    succ = runtime.sim_driver.add_lineage_successor(old_agent["instance_id"], kind="reload", now=2.0)

    runtime.state._instance_id = "parent-b"
    fresh = wf.WorkflowController(runtime.controller.config)
    adopted = fresh.adopt(runtime.state)
    assert adopted == ["miner"]
    assert "miner" in fresh.managed
    assert store.control()["controller"]["instance_id"] == "parent-b"
    assert any("running again" in n for n in fresh.notices)

    runtime.drive(fresh.rediscover_workflows(adopted))
    assert any(isinstance(r, DiscoverSessions) for r in runtime.requests)
    assert any(isinstance(r, AdoptSession) for r in runtime.requests)
    refreshed = store.agent("agt_reverse_1")
    assert refreshed["instance_id"] == succ.instance_id
    assert refreshed["websocket_url"] == succ.websocket_url
    assert fresh.instances_by_id[succ.instance_id] == ("miner", "agt_reverse_1")


def test_set_agent_count_starts_more(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    text = runtime.drive(runtime.plugin.set_agent_count(runtime.state, "miner", "grp_reverse", 3))
    assert "def_2" in text and "Started agt_reverse_3 on item_3" in text
    assert store.definition()["agents"]["grp_reverse"]["count"] == 3


def test_update_preview_and_apply(runtime):
    create(runtime)
    runtime.state.buffer_manager.texts["buf_new"] = DEFINITION.replace("count: 2", "count: 1")
    preview = runtime.plugin.update_workflow(runtime.state, "miner", "buf_new", "slower", preview_only=True)
    assert preview.startswith("Preview only") and "+    count: 1" in preview
    applied = runtime.plugin.update_workflow(runtime.state, "miner", "buf_new", "slower")
    assert "from def_1 to def_2" in applied


def test_move_items_hints(runtime):
    create(runtime)
    text = runtime.plugin.move_items(runtime.state, "miner", ["item_3"], "q_trash", note="dup")
    assert "Move them back out of q_trash" in text
    with pytest.raises(wf.WorkflowError, match="did you mean q_targets"):
        runtime.plugin.move_items(runtime.state, "miner", ["item_3"], "targets")


def test_message_agents_and_send_data(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    text = runtime.drive(runtime.plugin.message_agents(runtime.state, "miner", "pull the toolkit", "grp_reverse"))
    assert text == "Sent to agt_reverse_1, agt_reverse_2."
    assert runtime.delivered[-1]["channel"] == "user_text"
    assert runtime.delivered[-1]["payload"] == {"text": "pull the toolkit"}

    # Verify SendSessionData pushes typed SessionArtifact payloads over verified WS.
    agent = store.agent("agt_reverse_1")
    art = SessionArtifact(name="hint", kind="text", data="check layer 4")
    runtime.drive(runtime.controller.send_data(store, agent, [art], message="extra context"))
    assert any(isinstance(r, SendSessionData) for r in runtime.requests)
    assert len(runtime.sim_driver.delivered_data) == 1
    assert runtime.sim_driver.delivered_data[0].artifacts[0].data == "check layer 4"


def test_archive(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    text = runtime.drive(runtime.controller.archive("miner"))
    assert "Archived miner" in text
    assert store.control()["state"] == "archived"
    assert store.item("item_1")["state"] == "waiting"
    stop_reqs = [r for r in runtime.requests if isinstance(r, StopSession)]
    assert len(stop_reqs) == 3 and all(r.mode == "graceful_ws" for r in stop_reqs)
    with pytest.raises(wf.WorkflowError, match="archived"):
        runtime.drive(runtime.controller.run("miner"))


def test_unarchive_and_rename(runtime):
    store = create(runtime)
    with pytest.raises(wf.WorkflowError, match="is not archived"):
        runtime.controller.unarchive("miner")
    runtime.drive(runtime.controller.archive("miner"))
    with pytest.raises(wf.WorkflowError, match="already archived"):
        runtime.drive(runtime.controller.archive("miner"))
    assert "Restored miner" in runtime.controller.unarchive("miner")
    assert store.control()["state"] == "paused"
    assert "Renamed miner to catalog" in runtime.controller.rename("miner", "catalog")
    assert not store.exists()
    renamed = wf.WorkflowStore("catalog")
    assert renamed.item("item_1")["queue"] == "q_targets" and renamed.version == 1
    assert renamed.log_records()[-1]["event"] == "renamed"
    with pytest.raises(wf.WorkflowError, match="no workflow named"):
        runtime.controller.rename("miner", "other")


def test_rename_refuses_while_agents_are_tied_to_the_name(runtime):
    store = create(runtime)
    runtime.plugin.create_workflow(runtime.state, "other", "buf_def")
    with pytest.raises(wf.WorkflowError, match="already exists"):
        runtime.controller.rename("miner", "other")
    runtime.drive(runtime.controller.run("miner"))
    with pytest.raises(wf.WorkflowError, match="pause it first"):
        runtime.controller.rename("miner", "catalog")
    runtime.drive(runtime.controller.pause("miner"))
    with pytest.raises(wf.WorkflowError, match="not finished"):
        runtime.controller.rename("miner", "catalog")
    assert store.exists() and "miner" in runtime.controller.managed
    runtime.drive(runtime.controller.archive("miner"))  # Stops every agent, so the name is free to change.
    runtime.controller.rename("miner", "catalog")
    assert wf.WorkflowStore("catalog").control()["state"] == "archived"


# -- worker side ------------------------------------------------------------------

def worker_state(runtime, agent_id):
    return SimpleNamespace(_session_kind=f"wf:miner:{agent_id}", _session_id="w", _instance_id="w1",
                           buffer_manager=Buffers(buf_out="# Graph\nmlir", buf_bad="x"), pending_interrupts=[],
                           entries=[])


def test_worker_finish_task(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    worker = worker_state(runtime, "agt_reverse_1")
    plugin = wf.WorkflowsPlugin(wf.WorkflowController({}))
    with pytest.raises(wf.WorkflowError, match="only send to"):
        plugin.finish_task(worker, "s", [{"buffer": "buf_bad", "queue": "q_targets"}])
    with pytest.raises(wf.WorkflowError, match='"buffer"'):
        plugin.finish_task(worker, "s", [{"buf": "buf_out"}])
    text = plugin.finish_task(worker, "Recovered it", [{"buffer": "buf_out", "queue": "q_catalog"}])
    assert "created item_4 in q_catalog" in text
    assert worker.shutdown and worker.shutdown_reason == "suspend" and worker.done
    assert worker.shutdown_registry_status == "completed"
    assert store.agent("agt_reverse_1")["state"] == "finished"
    assert store.item("item_1")["state"] == "done"
    assert store.item("item_4")["from_items"] == ["item_1"]


def test_worker_reminder_then_attention(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    worker = worker_state(runtime, "agt_reverse_1")
    plugin = wf.WorkflowsPlugin(wf.WorkflowController({}))
    check = wf.WorkflowNotices(plugin)
    emit = lambda _ev: None
    worker.entries = [SimpleNamespace(role="system"), SimpleNamespace(role="user")]
    check.handle(TurnSettled(), worker, emit)  # baseline
    worker.entries.append(SimpleNamespace(role="assistant", tool_calls=[]))
    check.handle(TurnSettled(), worker, emit)
    assert "without calling finish_task" in worker.pending_interrupts[-1]
    check.handle(TurnSettled(), worker, emit)  # same entry: no second reminder
    assert len(worker.pending_interrupts) == 1
    worker.entries += [SimpleNamespace(role="user"), SimpleNamespace(role="assistant", tool_calls=None)]
    check.handle(TurnSettled(), worker, emit)
    assert store.agent("agt_reverse_1")["state"] == "attention"


def test_worker_receives_session_data_artifact(runtime):
    create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    worker = worker_state(runtime, "agt_reverse_1")
    worker.startup_paused = True
    plugin = wf.WorkflowsPlugin(wf.WorkflowController({}))
    check = wf.WorkflowNotices(plugin)
    emitted = []
    check.handle(
        SessionDataReceived(
            sender_session_id="parent",
            channel="workflow",
            message="Priority update",
            artifacts=(SessionArtifact(name="note", kind="text", data="focus on QKV"),),
        ),
        worker,
        emitted.append,
    )
    assert any("Priority update" in p and "focus on QKV" in p for p in worker.pending_interrupts)


def test_worker_schema_lists_allowed_queues(runtime):
    create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    worker = worker_state(runtime, "agt_reverse_1")
    plugin = wf.WorkflowsPlugin(wf.WorkflowController({}))
    tools = {t.name: t for t in wf.build_tools(plugin)}
    worker.tool_schemas = {}
    for tool in tools.values():
        tool._publish(worker)
    published = worker.tool_schemas
    assert published["finish_task"]["available"] and not published["create_workflow"]["available"]
    assert "You can send to: q_catalog, q_trash." in published["finish_task"]["schema"]["function"]["description"]
    assert "workflow_name" not in published["add_items"]["schema"]["function"]["parameters"]["properties"]
    runtime.state.tool_schemas = {}
    for tool in tools.values():
        tool._publish(runtime.state)
    assert not runtime.state.tool_schemas["finish_task"]["available"]
    assert "workflow_name" in runtime.state.tool_schemas["add_items"]["schema"]["function"]["parameters"]["properties"]


# -- command ---------------------------------------------------------------------

def command_context(runtime):
    emitted = []
    return SimpleNamespace(
        state=runtime.state,
        emit=emitted.append,
        emitted=emitted,
    )


def drain_command_io(runtime, ctx):
    """Drive all queued ``SessionRequest`` ops from ``ctx.state.io`` via ``SimSessionDriver``."""
    notices = wf.WorkflowNotices(runtime.plugin)
    while True:
        effects = ctx.state.io.take_outbox()
        session_ops = [
            eff for eff in effects
            if isinstance(eff, SubmitOp) and isinstance(eff.request, SessionRequest)
        ]
        if not session_ops:
            break
        for eff in session_ops:
            outcome = runtime.execute_session_request(eff.request)
            if outcome.error is not None:
                notices.handle(IoFailed(op_id=eff.op_id, error=outcome.error), ctx.state, ctx.emit)
            else:
                notices.handle(IoCompleted(op_id=eff.op_id, value=outcome.value), ctx.state, ctx.emit)
    return ctx.state.pending_interrupts[-1] if ctx.state.pending_interrupts else ""


def test_command_parsing_and_dispatch(runtime):
    create(runtime)
    command = wf.WorkflowCommand(runtime.plugin)
    ctx = command_context(runtime)
    listing = command(ctx, raw_args="")
    assert listing.outputs[0].body.startswith("| Workflow | Description | Status | Waiting | Done | Updated |")
    with pytest.raises(CommandError, match="usage"):
        command(ctx, raw_args=" run")
    with pytest.raises(CommandError, match="name the group with a colon: /workflow run miner:grp_reverse"):
        command(ctx, raw_args=" run miner grp_reverse")
    with pytest.raises(CommandError, match="did you mean miner:grp_reverse"):
        command(ctx, raw_args=" run miner:reverse")
    with pytest.raises(CommandError, match="no workflow named"):
        command(ctx, raw_args=" pause nope")
    assert ctx.state.io.take_outbox() == []
    receipt = command(ctx, raw_args=" run miner")
    assert receipt.outputs[0].text == "Running miner..."
    assert receipt.operation_id is not None
    text = drain_command_io(runtime, ctx)
    assert "Started agt_reverse_1" in text and "do not repeat" in text
    notice_events = [e for e in ctx.emitted if isinstance(e, EmitNotice)]
    assert notice_events[-1].text.startswith("Running miner from this session")


def test_command_list_archive_and_rename(runtime):
    create(runtime)
    runtime.plugin.create_workflow(runtime.state, "scratch", "buf_def")
    command = wf.WorkflowCommand(runtime.plugin)
    ctx = command_context(runtime)
    listing = command(ctx, raw_args="").outputs[0]
    assert listing.format == "markdown" and listing.subtitle == "2 paused"
    now = wf.unbroken("just now")
    assert f"| `miner` | — | paused | 3 | 0 | {now} |" in listing.body
    assert f"| `scratch` | — | paused | 0 | 0 | {now} |" in listing.body
    assert "discovery" not in listing.body
    assert '`/workflow set NAME description "..."`' in listing.body
    command(ctx, raw_args=' set scratch description "Throwaway test of archiving"')
    text = drain_command_io(runtime, ctx)
    assert "It shows in /workflow list" in text
    assert "| `scratch` | Throwaway test of archiving | paused |" in command(ctx, raw_args="").outputs[0].body

    res = command(ctx, raw_args=" archive scratch")
    assert "Archived scratch" in res.outputs[0].text or res.outputs[0].text == "Archiving scratch..."
    drain_command_io(runtime, ctx)
    listing = command(ctx, raw_args="").outputs[0]
    assert "`scratch`" not in listing.body and "1 archived workflow hidden" in listing.body
    assert listing.subtitle == "1 paused · 1 archived"
    archived = command(ctx, raw_args=" list archived").outputs[0]
    assert archived.title == "Archived workflows"
    assert archived.body.startswith("| Workflow | Description | Waiting | Done | Updated |")
    assert f"| `scratch` | Throwaway test of archiving | 0 | 0 | {now} |" in archived.body
    with pytest.raises(CommandError, match="already archived"):
        command(ctx, raw_args=" archive scratch")
    with pytest.raises(CommandError, match="not archived"):
        command(ctx, raw_args=" unarchive miner")
    with pytest.raises(CommandError, match="usage: /workflow archive NAME"):
        command(ctx, raw_args=" archive miner:grp_reverse")

    with pytest.raises(CommandError, match="usage: /workflow rename"):
        command(ctx, raw_args=" rename miner")
    with pytest.raises(CommandError, match="already exists"):
        command(ctx, raw_args=" rename miner scratch")
    with pytest.raises(CommandError, match="reserved"):
        command(ctx, raw_args=" rename miner archived")
    command(ctx, raw_args=" rename miner test_failures")
    drain_command_io(runtime, ctx)
    assert "Renamed miner to test_failures" in ctx.state.pending_interrupts[-1]
    assert "| `test_failures` | — | paused |" in command(ctx, raw_args="").outputs[0].body
    command(ctx, raw_args=" unarchive scratch")
    drain_command_io(runtime, ctx)
    assert command(ctx, raw_args="").outputs[0].subtitle == "2 paused"


def test_list_when_empty_or_all_archived(runtime):
    command = wf.WorkflowCommand(runtime.plugin)
    ctx = command_context(runtime)
    empty = command(ctx, raw_args="").outputs[0]
    assert empty.body.startswith("No workflows yet.") and empty.subtitle == ""
    create(runtime)
    runtime.drive(runtime.controller.archive("miner"))
    listing = command(ctx, raw_args="").outputs[0]
    assert listing.body.startswith("No active workflows.")
    assert "1 archived workflow hidden: `/workflow list archived`" in listing.body
    assert "miner" in wf.render_index() and "Archived" in wf.render_index()


def test_command_show_and_settings(runtime):
    create(runtime)
    command = wf.WorkflowCommand(runtime.plugin)
    ctx = command_context(runtime)
    assert command(ctx, raw_args=" list").outputs[0].title == "Workflows"
    overview = command(ctx, raw_args=" view miner").outputs[0]
    assert overview.format == "markdown" and overview.title == "miner" and overview.subtitle == "paused · def_1"
    assert "**Model** `offline`" in overview.body and "| `grp_reverse` | q_targets | q_catalog |" in overview.body
    group = command(ctx, raw_args=" view miner:grp_reverse").outputs[0]
    assert "**Model** `offline` (Agent Zoo default)" in group.body and group.subtitle.startswith("takes from")
    queue = command(ctx, raw_args=" view miner:q_targets").outputs[0]
    assert queue.subtitle == "3 waiting · 0 in progress · 0 done" and "| `item_1` | ResNet50 |" in queue.body
    settings = command(ctx, raw_args=" set miner:grp_reverse").outputs[0]
    assert settings.title == "Settings: miner:grp_reverse" and "| `count` | 2 |" in settings.body
    assert settings.format == "markdown"
    with pytest.raises(CommandError, match="to open a workflow, use /workflow view miner:grp_reverse"):
        command(ctx, raw_args=" miner:grp_reverse")
    with pytest.raises(CommandError, match="use one of: list, view"):
        command(ctx, raw_args=" frobnicate")
    with pytest.raises(CommandError, match="usage: /workflow view"):
        command(ctx, raw_args=" view")
    with pytest.raises(CommandError, match="usage: /workflow list"):
        command(ctx, raw_args=" list everything")
    with pytest.raises(wf.WorkflowError, match="reserved"):
        runtime.plugin.create_workflow(runtime.state, "view", "buf_def")
    assert ctx.state.io.take_outbox() == []


def test_command_set_validates_before_submitting(runtime):
    create(runtime)
    command = wf.WorkflowCommand(runtime.plugin)
    ctx = command_context(runtime)
    with pytest.raises(CommandError, match='unknown model profile "tiny"; enabled profiles: offline, big'):
        command(ctx, raw_args=" set miner:grp_reverse model tiny")
    with pytest.raises(CommandError, match='"retired" is disabled'):
        command(ctx, raw_args=" set miner model retired")
    with pytest.raises(CommandError, match="count is set on an agent group"):
        command(ctx, raw_args=" set miner count 4")
    with pytest.raises(CommandError, match="give a value"):
        command(ctx, raw_args=" set miner:grp_reverse count")
    assert command(ctx, raw_args=" set miner:grp_reverse count 2").outputs[0].text.endswith("nothing changed.")
    assert ctx.state.io.take_outbox() == []


def test_command_set_model_and_count(runtime):
    store = create(runtime)
    runtime.drive(runtime.controller.run("miner"))
    runtime.drive(runtime.controller.tick("miner"))
    command = wf.WorkflowCommand(runtime.plugin)
    ctx = command_context(runtime)
    command(ctx, raw_args=" set miner:grp_reverse model big")
    text = drain_command_io(runtime, ctx)
    assert "Set miner:grp_reverse model from default to big (def_2)." in text
    assert "agt_reverse_1 (offline), agt_reverse_2 (offline) keep their current model" in text
    assert "do not repeat" in text
    receipt = command(ctx, raw_args=" set miner:grp_reverse count 3")
    assert receipt.outputs[0].text == "Setting miner:grp_reverse count to 3..."
    text = drain_command_io(runtime, ctx)
    assert "Started agt_reverse_3 on item_3 from q_targets." in text
    assert runtime.spawns[-1].spec.model == "big"
    command(ctx, raw_args=" set miner:grp_reverse model default")
    drain_command_io(runtime, ctx)
    assert store.definition()["agents"]["grp_reverse"]["model"] is None
    command(ctx, raw_args=" set miner:q_catalog notify off")
    assert "no longer notify" in drain_command_io(runtime, ctx)
    command(ctx, raw_args=' set miner folder "~/work/my miner"')
    drain_command_io(runtime, ctx)
    assert store.definition()["folder"] == "~/work/my miner"
    assert [store.version_meta(v)["reason"] for v in range(2, store.version + 1)] == [
        "User set miner:grp_reverse model from default to big",
        "User set miner:grp_reverse count from 2 to 3",
        "User set miner:grp_reverse model from big to default",
        "User set miner:q_catalog notify from on to off",
        "User set miner folder from default to ~/work/my miner",
    ]


def test_create_and_update_check_models(runtime):
    runtime.state.buffer_manager.texts["buf_bad"] = "model: tiny\n" + DEFINITION
    with pytest.raises(wf.WorkflowError, match='definition: unknown model profile "tiny"'):
        runtime.plugin.create_workflow(runtime.state, "miner", "buf_bad")
    create(runtime)
    runtime.state.buffer_manager.texts["buf_bad2"] = DEFINITION.replace(
        "    count: 2\n", "    count: 2\n    model: retired\n")
    with pytest.raises(wf.WorkflowError, match='grp_reverse: model profile "retired" is disabled'):
        runtime.plugin.update_workflow(runtime.state, "miner", "buf_bad2", "x", preview_only=True)


# -- real pipelines & SimulationRuntime --------------------------------------------

@pytest.mark.parametrize("kind,is_worker", [("default", False), ("wf:miner:agt_reverse_1", True),
                                            ("wf:miner:agt_reverse_1x", False), ("rlm", False)])
def test_plugin_builds_real_default_pipeline(tmp_path, monkeypatch, kind, is_worker):
    import agent_zoo.pipelines as pipelines
    import agent_zoo.plugins.loader as plugins
    from agent_zoo.features.modes import default_modes
    root = tmp_path / "plugins"
    (root / "default").mkdir(parents=True)
    (root / "default" / "workflows.py").write_text(Path(wf.__file__).read_text())
    skill = Path(wf.resolve_state_root()) / "plugin-configs" / wf.PLUGIN_NAME / "skills/workflows/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text((Path(wf.__file__).resolve().parent.parent / "skills/workflows/SKILL.md").read_text())
    monkeypatch.setattr(plugins, "_base_dir", lambda: root)
    monkeypatch.setattr(plugins.metadata, "entry_points", lambda **kwargs: [])
    session = Session(system_prompt="offline", token_budget=120000)
    session.state._session_kind = kind
    pipeline = pipelines.build_default_pipeline(
        session, config={}, skill_paths=[], terminal_backend="headless",
        mode_defs=default_modes(), include_llm_drivers=False)
    session.pipeline = pipeline
    session.step(SessionReady())
    session.render()
    tools = {c.name for c in pipeline if isinstance(c, Tool)}
    assert {"create_workflow", "update_workflow", "command_workflow", "rename_workflow", "set_agent_count",
            "message_agents",
            "restart_agent", "add_items", "move_items", "finish_task"} <= tools
    assert bool(tools & {"submit_rlm", "rlm_status", "cancel_rlm"}) == (not is_worker)
    assert any(type(c).__name__ == "WorkflowNotices" for c in pipeline)
    rendered = session.messages[0]["content"]
    assert ("# Workflow agent" in rendered) == is_worker
    assert bool(session.state.skill_registry.get("Workflows")) == (not is_worker)
    assert "There are no workflows yet" in session.state.buffer_manager.resolve_for_read(session.state, "wf:index").text
    assert session.state.tool_schemas["finish_task"]["available"] is is_worker
    assert session.state.tool_schemas["create_workflow"]["available"] is (not is_worker)
    add_props = session.state.tool_schemas["add_items"]["schema"]["function"]["parameters"]["properties"]
    assert ("workflow_name" in add_props) is (not is_worker)


def test_register_features_composes_cleanly_on_default_builder():
    from agent_zoo.pipelines import build_default_builder
    session = Session(system_prompt="base", token_budget=120000)
    builder = build_default_builder(
        session, config={}, include_plugins=False, include_llm_drivers=False,
    )
    wf.register_features(builder, session=session, config={})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline = builder.build()
    assert not caught, [str(w.message) for w in caught]
    # Pure Sans-I/O: Workflows contributes zero custom drivers (delegates to AgentZooSessionDriver + TimerDriver).
    assert not hasattr(wf, "WorkflowDriver")
    assert builder.build_drivers({}) == []
    session.pipeline = pipeline
    session.step(SessionReady())
    session.render()
    assert "There are no workflows yet" in session.state.buffer_manager.resolve_for_read(session.state, "wf:index").text


def test_event_driven_tool_and_slash_command_pipeline_with_sim_session_driver(tmp_path, monkeypatch):
    from agent_zoo.pipelines import build_default_builder

    monkeypatch.setattr(wf, "saved_model_config", lambda config: None)
    cfg = {"llm": {"default": "offline", "models": {"offline": {"litellm": {"request": {"model": "a"}}}}}}
    session = Session(system_prompt="base", token_budget=120000, cwd=str(tmp_path / "work"))
    session.state._session_id = "parent"
    session.state._instance_id = "parent-a"
    session.state._agent_zoo_context = {
        "project_name": "test",
        "project_store_root": str(tmp_path / "registry"),
    }
    session.state._persistence_local_root = str(tmp_path / "local")
    session.state.startup_paused = True

    builder = build_default_builder(
        session, config=cfg, include_plugins=False, include_llm_drivers=False,
    )
    wf.register_features(builder, session=session, config=cfg)
    session.pipeline = builder.build()

    sim = SimulationRuntime(session)
    sim_driver = SimSessionDriver(default_delay=0.01)
    sim_driver.register(sim)

    def advance(until: float) -> None:
        while sim.pending_events:
            next_time = sim._queue[0][0]
            if next_time > until:
                break
            sim.step_next()

    sim.inject(SessionReady(session_id="parent", instance_id="parent-a", paused=True))
    advance(0.05)
    bid, _ = session.state.buffer_manager.create_scratch_buffer(DEFINITION)

    # Dispatch create_workflow through RunToolCall.
    tc = ToolCall(
        id="tc_1",
        name="create_workflow",
        raw_args="{}",
        parsed_args={"workflow_name": "miner", "definition_buffer": bid},
    )
    sim.inject(RunToolCall(tc))
    advance(0.1)
    assert not tc.error and "Created workflow miner (def_1)" in tc.result

    # Dispatch /workflow run miner through SlashCommand -> pure SpawnSession via SimSessionDriver!
    sim.inject(SlashCommand(raw="/workflow run miner"))
    advance(0.5)
    assert any(
        e.system_generated and "do not repeat" in e.messages[0]["content"] and "Started agt_discovery_1." in e.messages[0]["content"]
        for e in session.state.entries
    )
    store = wf.WorkflowStore("miner")
    agent = store.agent("agt_discovery_1")
    assert agent["state"] == "starting"

    # Unsolicited ExternalSessionReady transitions worker to running immediately!
    sim.inject(
        ExternalSessionReady(
            session_id=agent["session_id"],
            instance_id=agent["instance_id"],
            websocket_url=sim_driver.sessions[agent["instance_id"]].websocket_url,
            process_identity=sim_driver.sessions[agent["instance_id"]].process_identity.to_dict(),
        )
    )
    advance(0.6)
    assert store.agent("agt_discovery_1")["state"] == "running"

    # Periodic StartTimer / TimerFired delivers new-item notices without any background thread!
    session.state.entries.append(
        Entry(messages=[{"role": "assistant", "content": "ok"}], index=len(session.state.entries), step=session.state.step)
    )
    store.add_item("q_catalog", "found model", by="agt_discovery_1")
    advance(3.0)
    assert any(
        e.system_generated and "new items in q_catalog" in e.messages[0]["content"]
        for e in session.state.entries
    )

    # Verify worker finish_task emits SuspendRequested(registry_status="completed") -> SaveCheckpoint -> state.done.
    worker_session = Session(system_prompt="base", token_budget=120000)
    worker_session.state._session_kind = "wf:miner:agt_discovery_1"
    worker_builder = build_default_builder(
        worker_session, config=cfg, include_plugins=False, include_llm_drivers=False,
    )
    wf.register_features(worker_builder, session=worker_session, config=cfg)
    worker_session.pipeline = worker_builder.build()
    worker_session.step(SessionReady())
    finish_tc = ToolCall(
        id="tc_finish",
        name="finish_task",
        raw_args="{}",
        parsed_args={"summary": "All done"},
    )
    finish_effects = worker_session.step(RunToolCall(finish_tc))
    assert not finish_tc.error and "Task finished" in finish_tc.result
    assert worker_session.state.shutdown
    assert worker_session.state.shutdown_registry_status == "completed"
    save_effs = [eff for eff in finish_effects if isinstance(getattr(eff, "request", None), SaveCheckpoint)]
    assert len(save_effs) == 1
    worker_session.step(IoCompleted(op_id=save_effs[0].op_id, value={"revision_id": "rev-1"}))
    assert worker_session.state.done
