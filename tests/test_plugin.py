"""Plugin surface, scheduling, control and recovery with a fake launcher and websocket."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agent_utils import Session, Tool
from agent_zoo.runtime.commands import CommandError

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


class FakeHandle:
    def __init__(self, request):
        self.request, self.ref = request, request.target
        self.process = SimpleNamespace(pid=4242)
        self.status = "ready"

    def poll(self):
        ready = {"websocket_url": f"ws://127.0.0.1:1/{self.ref.instance_id}",
                 "instance_id": self.ref.instance_id, "process_identity": {"pid": 4242}}
        return SimpleNamespace(status=self.status, ready=ready if self.status == "ready" else None)


class FakeLauncher:
    def __init__(self):
        self.requests, self.handles = [], []

    def prepare(self, request, cwd=None, env=None):
        return SimpleNamespace(request=request, env=env, cwd=cwd)

    def start(self, plan):
        self.requests.append(plan.request)
        handle = FakeHandle(plan.request)
        self.handles.append(handle)
        return handle


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    sent = []

    async def fake_send(target, action, message=""):
        sent.append((target["session_id"], action, message))

    liveness = {}
    monkeypatch.setattr(wf, "send_to_agent", fake_send)
    monkeypatch.setattr(wf, "capture_process_identity", lambda pid: None)
    monkeypatch.setattr(wf, "process_evidence", lambda identity: liveness.get("default", "alive"))
    monkeypatch.setattr(wf, "list_live_sessions", lambda *a, **k: [])
    monkeypatch.setattr(wf.WorkflowController, "ensure_thread", lambda self: None)
    launcher = FakeLauncher()
    state = SimpleNamespace(
        _session_id="parent", _instance_id="parent-a", _session_kind="default",
        _agent_zoo_context={"project_name": "test", "project_store_root": str(tmp_path / "registry")},
        session_launcher=launcher, session_cwd=str(tmp_path / "work"), _persistence_local_root=str(tmp_path / "local"),
        buffer_manager=Buffers(buf_def=DEFINITION, buf_a="# ResNet50\nhttps://a", buf_b="BERT", buf_c="GPT-2"),
        pending_interrupts=[],
    )
    controller = wf.WorkflowController({"llm": {"default": "offline", "models": {"offline": {}}}})
    plugin = wf.WorkflowsPlugin(controller)
    controller.bind(state)
    return SimpleNamespace(state=state, controller=controller, plugin=plugin, launcher=launcher, sent=sent,
                           liveness=liveness, tmp=tmp_path)


def create(r):
    r.plugin.create_workflow(r.state, "miner", "buf_def")
    r.plugin.add_items(r.state, "miner", "q_targets", ["buf_a", "buf_b", "buf_c"])
    return wf.WorkflowStore("miner")


def test_create_reports_and_does_not_run(runtime):
    text = runtime.plugin.create_workflow(runtime.state, "miner", "buf_def")
    assert "Nothing is running yet" in text and "/workflow run miner" in text
    assert runtime.launcher.requests == []
    with pytest.raises(wf.WorkflowError, match="no buffer"):
        runtime.plugin.create_workflow(runtime.state, "other", "buf_missing")


def test_run_starts_agents_with_rendered_instructions(runtime):
    store = create(runtime)
    text = runtime.controller.run("miner")
    assert "Started agt_discovery_1." in text
    assert "Started agt_reverse_1 on item_1 from q_targets." in text
    assert "Started agt_reverse_2 on item_2 from q_targets." in text
    requests = {r.kind: r for r in runtime.launcher.requests}
    reverse = requests["wf:miner:agt_reverse_1"]
    assert reverse.lifetime == "independent" and reverse.startup_mode == "normal"
    assert "Reverse engineer this." in reverse.first_user_message
    assert "item_1 from q_targets: ResNet50" in reverse.first_user_message
    assert requests["wf:miner:agt_discovery_1"].first_user_message == "Find checkpoints for miner."
    assert reverse.settings.model == "offline" and reverse.settings.project == "test"
    assert store.item("item_1")["taken_by"] == "agt_reverse_1"
    assert store.control()["controller"]["session_id"] == "parent"
    # Ready agents become running; the standing group respects `every`.
    runtime.controller.tick("miner")
    assert store.agent("agt_reverse_1")["state"] == "running"
    assert len(runtime.launcher.requests) == 3


def test_finished_agent_frees_a_slot(runtime):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    store.finish_agent("agt_reverse_1", summary="done", outputs=[("q_catalog", "graph")])
    started = runtime.controller.tick("miner")
    assert started == ["Started agt_reverse_3 on item_3 from q_targets."]
    assert any("new items in q_catalog: item_4 (graph)" in n for n in runtime.controller.notices)


def test_pause_then_run_resumes(runtime):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    text = runtime.controller.pause("miner", "grp_reverse")
    assert "Interrupted agt_reverse_1, agt_reverse_2" in text
    assert store.agent("agt_reverse_1")["state"] == "paused"
    assert [a for _, a, _ in runtime.sent] == ["interrupt", "interrupt"]
    assert runtime.controller.tick("miner") == []  # group paused: no replacements
    text = runtime.controller.run("miner", "grp_reverse")
    assert "Resumed agt_reverse_1, agt_reverse_2" in text
    assert store.agent("agt_reverse_2")["state"] == "running"
    assert [a for _, a, _ in runtime.sent][-2:] == ["continue", "continue"]


def test_crashed_agent_resumes_from_checkpoint(runtime, monkeypatch):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    monkeypatch.setattr(wf, "saved_checkpoint", lambda project, sid, revision=None: SimpleNamespace(commit_id="c1"))
    runtime.launcher.handles[1].status = "exited"
    runtime.controller.tick("miner")
    resumed = runtime.launcher.requests[-1]
    assert resumed.kind == "wf:miner:agt_reverse_1" and resumed.startup_mode == "paused"
    assert resumed.source.checkpoint.revision_id == "c1"
    assert store.agent("agt_reverse_1")["resumes"] == 1
    runtime.controller.tick("miner")
    assert runtime.sent[-1][1] == "message" and "restarted" in runtime.sent[-1][2]
    assert store.agent("agt_reverse_1")["state"] == "running"


def test_missing_checkpoint_needs_attention(runtime, monkeypatch):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")

    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(wf, "saved_checkpoint", missing)
    runtime.launcher.handles[1].status = "exited"
    runtime.controller.tick("miner")
    agent = store.agent("agt_reverse_1")
    assert agent["state"] == "attention" and "restart_agent" in agent["problem"]
    assert any("agt_reverse_1 needs attention" in n for n in runtime.controller.notices)
    text = runtime.controller.restart_agent("miner", "agt_reverse_1")
    assert "item_1 is back in q_targets" in text
    assert store.item("item_1")["state"] == "waiting"


def test_launch_failure_pauses_group(runtime):
    store = create(runtime)
    runtime.launcher.start = Mock(side_effect=OSError("boom"))
    runtime.controller.run("miner")
    assert "grp_reverse" in store.control()["paused_groups"]
    assert store.item("item_1")["state"] == "waiting"
    assert any("could not start an agent in grp_reverse: boom" in n for n in runtime.controller.notices)


def test_losing_control_stops_scheduling(runtime):
    store = create(runtime)
    runtime.controller.run("miner")
    store.update_control(controller={"session_id": "other", "instance_id": "o"}, epoch=99)
    assert runtime.controller.tick("miner") == []
    assert "miner" not in runtime.controller.managed
    assert any("controlled by session other" in n for n in runtime.controller.notices)


def test_adopt_after_restart(runtime):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.state._instance_id = "parent-b"
    fresh = wf.WorkflowController(runtime.controller.config)
    fresh.adopt(runtime.state)
    assert "miner" in fresh.managed
    assert store.control()["controller"]["instance_id"] == "parent-b"
    assert any("running again" in n for n in fresh.notices)


def test_set_agent_count_starts_more(runtime):
    store = create(runtime)
    runtime.controller.run("miner")
    text = runtime.plugin.set_agent_count(runtime.state, "miner", "grp_reverse", 3)
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


def test_message_agents(runtime):
    create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    text = runtime.plugin.message_agents(runtime.state, "miner", "pull the toolkit", "grp_reverse")
    assert text == "Sent to agt_reverse_1, agt_reverse_2."
    assert runtime.sent[-1][1:] == ("message", "pull the toolkit")


def test_archive(runtime):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    text = runtime.controller.archive("miner")
    assert "Archived miner" in text
    assert store.control()["state"] == "archived"
    assert store.item("item_1")["state"] == "waiting"
    with pytest.raises(wf.WorkflowError, match="archived"):
        runtime.controller.run("miner")


# -- worker side ------------------------------------------------------------------

def worker_state(runtime, agent_id):
    return SimpleNamespace(_session_kind=f"wf:miner:{agent_id}", _session_id="w", _instance_id="w1",
                           buffer_manager=Buffers(buf_out="# Graph\nmlir", buf_bad="x"), pending_interrupts=[],
                           entries=[])


def test_worker_finish_task(runtime):
    store = create(runtime)
    runtime.controller.run("miner")
    worker = worker_state(runtime, "agt_reverse_1")
    plugin = wf.WorkflowsPlugin(wf.WorkflowController({}))
    with pytest.raises(wf.WorkflowError, match="only send to"):
        plugin.finish_task(worker, "s", [{"buffer": "buf_bad", "queue": "q_targets"}])
    with pytest.raises(wf.WorkflowError, match='"buffer"'):
        plugin.finish_task(worker, "s", [{"buf": "buf_out"}])
    text = plugin.finish_task(worker, "Recovered it", [{"buffer": "buf_out", "queue": "q_catalog"}])
    assert "created item_4 in q_catalog" in text
    assert worker.shutdown and worker.shutdown_reason == "suspend" and worker.done
    assert store.agent("agt_reverse_1")["state"] == "finished"
    assert store.item("item_1")["state"] == "done"
    assert store.item("item_4")["from_item"] == "item_1"


def test_worker_reminder_then_attention(runtime):
    store = create(runtime)
    runtime.controller.run("miner")
    worker = worker_state(runtime, "agt_reverse_1")
    plugin = wf.WorkflowsPlugin(wf.WorkflowController({}))
    check = wf.WorkflowNotices(plugin)
    worker.entries = [SimpleNamespace(role="system"), SimpleNamespace(role="user")]
    check(worker)  # baseline
    worker.entries.append(SimpleNamespace(role="assistant", tool_calls=[]))
    check(worker)
    assert "without calling finish_task" in worker.pending_interrupts[-1]
    check(worker)  # same entry: no second reminder
    assert len(worker.pending_interrupts) == 1
    worker.entries += [SimpleNamespace(role="user"), SimpleNamespace(role="assistant", tool_calls=None)]
    check(worker)
    assert store.agent("agt_reverse_1")["state"] == "attention"


def test_worker_schema_lists_allowed_queues(runtime):
    create(runtime)
    runtime.controller.run("miner")
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

def test_command_parsing_and_dispatch(runtime):
    create(runtime)
    command = wf.WorkflowCommand(runtime.plugin)
    session = SimpleNamespace(state=runtime.state, inject=Mock())
    ctx = SimpleNamespace(state=runtime.state, session=session, defer=Mock())
    listing = command(ctx, raw_args="")
    assert "miner: paused" in listing.outputs[0].body
    with pytest.raises(CommandError, match="Usage"):
        command(ctx, raw_args=" run")
    with pytest.raises(CommandError, match="did you mean grp_reverse"):
        command(ctx, raw_args=" run miner reverse")
    with pytest.raises(CommandError, match="no workflow named"):
        command(ctx, raw_args=" pause nope")
    ctx.defer.assert_not_called()
    receipt = command(ctx, raw_args=" run miner")
    assert receipt.outputs[0].text == "Running miner..."
    work, then = ctx.defer.call_args.args[0], ctx.defer.call_args.kwargs["then"]
    result = asyncio.run(work())
    assert result["ok"] and "Started agt_reverse_1" in result["text"]
    notice = then(ctx, result)
    assert notice.outputs[0].text.startswith("Running miner from this session")
    assert "do not repeat" in session.inject.call_args.args[0]


# -- real pipelines ------------------------------------------------------------------

@pytest.mark.parametrize("kind,is_worker", [("default", False), ("wf:miner:agt_reverse_1", True),
                                            ("wf:miner:agt_reverse_1x", False), ("rlm", False)])
def test_plugin_builds_real_default_pipeline(tmp_path, monkeypatch, kind, is_worker):
    import agent_zoo.pipelines as pipelines
    import agent_zoo.plugins as plugins
    from agent_zoo.modes import default_modes
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
        session, config={}, skill_paths=[], max_idle=0, terminal_backend="headless",
        mode_defs=default_modes(), on_interrupt=lambda _: None)
    session.pipeline = pipeline
    session.ensure_initialized()
    tools = {c.name for c in pipeline if isinstance(c, Tool)}
    assert {"create_workflow", "update_workflow", "command_workflow", "set_agent_count", "message_agents",
            "restart_agent", "add_items", "move_items", "finish_task"} <= tools
    assert bool(tools & {"submit_rlm", "rlm_status", "cancel_rlm"}) == (not is_worker)
    assert any(type(c).__name__ == "WorkflowNotices" for c in pipeline)
    prompt = next(c for c in pipeline if type(c).__name__ == "WorkflowSystemPrompt")
    rendered = prompt.transform([{"role": "system", "content": "Base"}], session.state)[0]["content"]
    assert ("# Workflow agent" in rendered) == is_worker
    registration = next(c for c in pipeline if type(c).__name__ == "WorkflowSkillRegistration")
    registration(session.state)
    assert bool(session.state.skill_registry.get("Workflows")) == (not is_worker)
    session.state.buffer_manager  # registered by the pipeline
    buffers = next(c for c in pipeline if type(c).__name__ == "WorkflowBuffers")
    buffers(session.state)
    assert "There are no workflows yet" in session.state.buffer_manager.resolve_for_read(session.state, "wf:index").text
