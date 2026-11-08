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
    monkeypatch.setattr(wf, "saved_model_config", lambda config: None)
    launcher = FakeLauncher()
    state = SimpleNamespace(
        _session_id="parent", _instance_id="parent-a", _session_kind="default",
        _agent_zoo_context={"project_name": "test", "project_store_root": str(tmp_path / "registry")},
        session_launcher=launcher, session_cwd=str(tmp_path / "work"), _persistence_local_root=str(tmp_path / "local"),
        buffer_manager=Buffers(buf_def=DEFINITION, buf_a="# ResNet50\nhttps://a", buf_b="BERT", buf_c="GPT-2"),
        pending_interrupts=[],
    )
    controller = wf.WorkflowController({"llm": {"default": "offline", "models": {
        "offline": {"model": "a"}, "big": {"model": "b"}, "retired": {"model": "c", "enabled": False}}}})
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
    text = runtime.controller.run("pairs")
    assert "No agents started yet: q_b is empty, so grp_pair has nothing to take." in text
    assert "grp_pair will take them." in runtime.plugin.add_items(runtime.state, "pairs", "q_b", ["buf_b1"])
    assert runtime.controller.tick("pairs") == ["Started agt_pair_1 on item_1, item_2 from q_a and item_4 from q_b."]
    message = runtime.launcher.requests[-1].first_user_message
    assert all(f"{i} from q_" in message for i in ("item_1", "item_2", "item_4")) and "item_3" not in message
    assert store.agent("agt_pair_1")["items"] == ["item_1", "item_2", "item_4"]
    assert runtime.controller.idle_reason(store, None) == (
        "grp_pair takes 2 items from q_a at a time and q_a has 1 waiting; q_b is empty, so grp_pair has nothing "
        "to take.")
    text = runtime.controller.restart_agent("pairs", "agt_pair_1")
    assert "item_1, item_2, item_4 are back in their queues" in text
    assert all(m["state"] == "waiting" for m in store.items())


def test_batch_returns_on_archive_and_launch_failure(runtime):
    store = create_pairs(runtime)
    runtime.plugin.add_items(runtime.state, "pairs", "q_b", ["buf_b1"])
    runtime.controller.run("pairs")
    assert store.agent("agt_pair_1")["items"] == ["item_1", "item_2", "item_4"]
    runtime.controller.archive("pairs")
    assert all(m["state"] == "waiting" and m["taken_by"] is None for m in store.items())
    runtime.controller.unarchive("pairs")
    runtime.launcher.start = Mock(side_effect=OSError("boom"))
    runtime.controller.run("pairs")
    assert store.agent("agt_pair_2")["state"] == "stopped" and "grp_pair" in store.control()["paused_groups"]
    assert all(m["state"] == "waiting" and m["taken_by"] is None for m in store.items())


def test_batch_worker_outputs_record_every_origin(runtime):
    store = create_pairs(runtime)
    runtime.plugin.add_items(runtime.state, "pairs", "q_b", ["buf_b1"])
    runtime.controller.run("pairs")
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
    runtime.controller.run("miner")
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
    assert "item_2 is back in q_targets" in runtime.controller.restart_agent("miner", "agt_reverse_2")
    assert store.item("item_2")["state"] == "waiting"


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
    monkeypatch.setattr(wf, "saved_checkpoint", lambda project, sid, revision=None: SimpleNamespace(
        commit_id="c1", document={"core": {"done": False}, "entries": [
            {"messages": [{"role": "user", "content": "Continue the task"}], "tool_calls": []}]}))
    runtime.launcher.handles[1].status = "exited"
    runtime.controller.tick("miner")
    resumed = runtime.launcher.requests[-1]
    assert resumed.kind == "wf:miner:agt_reverse_1" and resumed.startup_mode == "paused"
    assert resumed.source.checkpoint.revision_id == "c1"
    assert store.agent("agt_reverse_1")["resumes"] == 1
    runtime.controller.tick("miner")
    assert runtime.sent[-1][1] == "continue"
    assert not any(action == "message" for _, action, _ in runtime.sent)
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


def test_unarchive_and_rename(runtime):
    store = create(runtime)
    with pytest.raises(wf.WorkflowError, match="is not archived"):
        runtime.controller.unarchive("miner")
    runtime.controller.archive("miner")
    with pytest.raises(wf.WorkflowError, match="already archived"):
        runtime.controller.archive("miner")
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
    runtime.controller.run("miner")
    with pytest.raises(wf.WorkflowError, match="pause it first"):
        runtime.controller.rename("miner", "catalog")
    runtime.controller.pause("miner")
    with pytest.raises(wf.WorkflowError, match="not finished"):
        runtime.controller.rename("miner", "catalog")
    assert store.exists() and "miner" in runtime.controller.managed
    runtime.controller.archive("miner")  # Stops every agent, so the name is free to change.
    runtime.controller.rename("miner", "catalog")
    assert wf.WorkflowStore("catalog").control()["state"] == "archived"


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
    assert store.item("item_4")["from_items"] == ["item_1"]


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
    assert listing.outputs[0].body.startswith("| Workflow | Description | Status | Waiting | Done | Updated |")
    with pytest.raises(CommandError, match="usage"):
        command(ctx, raw_args=" run")
    with pytest.raises(CommandError, match="name the group with a colon: /workflow run miner:grp_reverse"):
        command(ctx, raw_args=" run miner grp_reverse")
    with pytest.raises(CommandError, match="did you mean miner:grp_reverse"):
        command(ctx, raw_args=" run miner:reverse")
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


def command_context(runtime):
    session = SimpleNamespace(state=runtime.state, inject=Mock())
    return SimpleNamespace(state=runtime.state, session=session, defer=Mock())


def finish_deferred(ctx):
    work, then = ctx.defer.call_args.args[0], ctx.defer.call_args.kwargs["then"]
    result = asyncio.run(work())
    then(ctx, result)
    return result


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
    assert "It shows in /workflow list" in finish_deferred(ctx)["text"]
    assert "| `scratch` | Throwaway test of archiving | paused |" in command(ctx, raw_args="").outputs[0].body

    assert command(ctx, raw_args=" archive scratch").outputs[0].text == "Archiving scratch..."
    assert finish_deferred(ctx)["ok"]
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
    assert command(ctx, raw_args=" rename miner test_failures").outputs[0].text == "Renaming miner to test_failures..."
    assert finish_deferred(ctx)["ok"]
    assert "Renamed miner to test_failures" in ctx.session.inject.call_args.args[0]
    assert "| `test_failures` | — | paused |" in command(ctx, raw_args="").outputs[0].body
    command(ctx, raw_args=" unarchive scratch")
    finish_deferred(ctx)
    assert command(ctx, raw_args="").outputs[0].subtitle == "2 paused"


def test_list_when_empty_or_all_archived(runtime):
    command = wf.WorkflowCommand(runtime.plugin)
    ctx = command_context(runtime)
    empty = command(ctx, raw_args="").outputs[0]
    assert empty.body.startswith("No workflows yet.") and empty.subtitle == ""
    create(runtime)
    runtime.controller.archive("miner")
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
    ctx.defer.assert_not_called()


def test_command_set_validates_before_deferring(runtime):
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
    ctx.defer.assert_not_called()


def test_command_set_model_and_count(runtime):
    store = create(runtime)
    runtime.controller.run("miner")
    runtime.controller.tick("miner")
    command = wf.WorkflowCommand(runtime.plugin)
    ctx = command_context(runtime)
    receipt = command(ctx, raw_args=" set miner:grp_reverse model big")
    assert receipt.outputs[0].text == "Setting miner:grp_reverse model to big..."
    result = finish_deferred(ctx)
    assert result["ok"]
    assert "Set miner:grp_reverse model from default to big (def_2)." in result["text"]
    assert "agt_reverse_1 (offline), agt_reverse_2 (offline) keep their current model" in result["text"]
    assert "do not repeat" in ctx.session.inject.call_args.args[0]
    command(ctx, raw_args=" set miner:grp_reverse count 3")
    result = finish_deferred(ctx)
    assert "Started agt_reverse_3 on item_3 from q_targets." in result["text"]
    assert runtime.launcher.requests[-1].settings.model == "big"
    command(ctx, raw_args=" set miner:grp_reverse model default")
    finish_deferred(ctx)
    assert store.definition()["agents"]["grp_reverse"]["model"] is None
    command(ctx, raw_args=" set miner:q_catalog notify off")
    assert "no longer notify" in finish_deferred(ctx)["text"]
    command(ctx, raw_args=' set miner folder "~/work/my miner"')
    finish_deferred(ctx)
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
    assert {"create_workflow", "update_workflow", "command_workflow", "rename_workflow", "set_agent_count",
            "message_agents",
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
