import pytest

import workflows as wf

DEFINITION = """\
folder: /tmp/miner
queues:
  q_targets: {}
  q_catalog: { notify_on_new: true }
  q_changes: {}
agents:
  grp_discovery:
    count: 1
    every: 30m
    sends_to: [q_targets]
    instructions: |
      Find checkpoints. Check wf:{{workflow_name}}:q_targets first.
  grp_reverse:
    count: 4
    takes_from: q_targets
    sends_to: [q_catalog, q_changes]
    instructions: |
      Reverse engineer this.
      {{item}}
"""


@pytest.fixture
def store(tmp_path):
    s = wf.WorkflowStore("miner", tmp_path / "workflows")
    s.create(DEFINITION, by="user")
    return s


def test_parse_valid_definition():
    d = wf.parse_definition(DEFINITION)
    assert d["agents"]["grp_discovery"]["every"] == 1800
    assert d["agents"]["grp_reverse"]["takes_from"] == "q_targets"
    assert d["queues"]["q_catalog"]["notify_on_new"] is True


def test_parse_reports_every_problem_with_suggestions():
    text = """\
queues:
  targets: {}
agents:
  reverse:
    instructions: x
  grp_a:
    takes_from: targets
    sends_to: [q_missing]
    every: soon
    count: 99
    prompt: hi
"""
    with pytest.raises(wf.WorkflowError) as error:
        wf.parse_definition(text)
    message = str(error.value)
    assert "did you mean q_targets?" in message
    assert "did you mean grp_reverse?" in message
    assert 'sends_to "q_missing" is not listed under queues:' in message
    assert "count must be a whole number" in message
    assert 'unknown setting "prompt"' in message
    assert "instructions are required" in message
    assert "every only applies" in message


def test_names_are_checked():
    with pytest.raises(wf.WorkflowError, match="did you mean q_targets"):
        wf.check_queue_id("targets")
    with pytest.raises(wf.WorkflowError, match="item_42"):
        wf.check_item_id("42")
    for reserved in ("index", "run", "set"):
        with pytest.raises(wf.WorkflowError, match="reserved"):
            wf.WorkflowStore(reserved)
    with pytest.raises(wf.WorkflowError):
        wf.WorkflowStore("Bad Name")


def test_create_and_versions(store):
    assert store.version == 1
    assert store.definition_text().startswith("version: 1\n")
    assert store.control()["state"] == "paused"
    with pytest.raises(wf.WorkflowError, match="already exists"):
        store.create(DEFINITION, by="user")


def test_update_returns_diff_and_effects(store):
    new = DEFINITION.replace("count: 4", "count: 2").replace(
        "  q_changes: {}\n", "  q_changes: {}\n  q_review: {}\n")
    preview = store.update("version: 1\n" + new, reason="slow down", by="user", preview_only=True)
    assert "-    count: 4" in preview["diff"] and "+    count: 2" in preview["diff"]
    assert any("Adds q_review" in e for e in preview["effects"])
    assert store.version == 1
    result = store.update(new, reason="slow down", by="user")
    assert result["version"] == 2
    assert store.definition()["agents"]["grp_reverse"]["count"] == 2
    assert store.version_meta(2)["reason"] == "slow down"


def test_stale_update_is_rejected_with_diff(store):
    store.set_count("grp_reverse", 2, by="user")
    with pytest.raises(wf.StaleDefinition) as error:
        store.update("version: 1\n" + DEFINITION, reason="x", by="user")
    assert "def_2" in str(error.value) and "+    count: 2" in str(error.value)


def test_set_count_preserves_formatting(store):
    result = store.set_count("grp_discovery", 3, by="user")
    assert result["changed"] and store.version == 2
    body = store.definition_body()
    assert "count: 3" in body and "Find checkpoints." in body and "every: 30m" in body
    assert store.definition()["agents"]["grp_discovery"]["count"] == 3


def test_set_count_inserts_missing_count(tmp_path):
    s = wf.WorkflowStore("x", tmp_path)
    s.create("queues: {}\nagents:\n  grp_a:\n    instructions: hi\n", by="user")
    s.set_count("grp_a", 5, by="user")
    assert s.definition()["agents"]["grp_a"]["count"] == 5


def test_set_setting_text_edits_one_line():
    text = DEFINITION
    # Replace, insert, remove at group level; comments and instructions survive.
    edited = wf.set_setting_text(text, "agents", "grp_reverse", "model", "claude-opus-5-5")
    assert "    model: claude-opus-5-5\n    count: 4" in edited
    assert wf.parse_definition(edited)["agents"]["grp_reverse"]["model"] == "claude-opus-5-5"
    edited = wf.set_setting_text(edited, "agents", "grp_reverse", "model", "claude-sonnet-5-5")
    assert edited.count("model:") == 1
    assert wf.set_setting_text(edited, "agents", "grp_reverse", "model", None) == text
    removed = wf.set_setting_text(text, "agents", "grp_discovery", "every", None)
    assert "every" not in removed and "Find checkpoints." in removed
    # Top level.
    top = wf.set_setting_text(text, None, None, "model", "big")
    assert wf.parse_definition(top)["model"] == "big"
    assert wf.set_setting_text(top, None, None, "folder", "~/work new")
    # Flow-style and bare queue settings.
    flow = wf.set_setting_text(text, "queues", "q_catalog", "notify_on_new", False)
    assert "q_catalog: { notify_on_new: false }" in flow
    assert "q_targets: {}" in wf.set_setting_text(text.replace("q_targets: {}", "q_targets:"),
                                                  "queues", "q_targets", "notify_on_new", None)
    on = wf.set_setting_text(text, "queues", "q_targets", "notify_on_new", True)
    assert wf.parse_definition(on)["queues"]["q_targets"]["notify_on_new"] is True


def test_prepare_setting_explains_mistakes(store):
    with pytest.raises(wf.WorkflowError, match="count is set on an agent group; for example: "
                                               "/workflow set miner:grp_discovery count 4"):
        wf.prepare_setting(store, "workflow", None, "count", "4")
    with pytest.raises(wf.WorkflowError, match="every only applies to groups without takes_from"):
        wf.prepare_setting(store, "group", "grp_reverse", "every", "1h")
    with pytest.raises(wf.WorkflowError, match="not a duration"):
        wf.prepare_setting(store, "group", "grp_discovery", "every", "soon")
    with pytest.raises(wf.WorkflowError, match="0 to 64"):
        wf.prepare_setting(store, "group", "grp_reverse", "count", "100")
    with pytest.raises(wf.WorkflowError, match="unknown setting"):
        wf.prepare_setting(store, "group", "grp_reverse", "size", "3")
    with pytest.raises(wf.WorkflowError, match="this workflow has"):
        wf.prepare_setting(store, "group", "grp_missing", "count", "3")

    def check(profile):
        if profile != "big":
            raise wf.WorkflowError("enabled profiles: big")
        return profile

    with pytest.raises(wf.WorkflowError, match="enabled profiles: big"):
        wf.prepare_setting(store, "group", "grp_reverse", "model", "tiny", check_model=check)
    change = wf.prepare_setting(store, "group", "grp_reverse", "model", "big", check_model=check)
    assert change["value"] == "big" and change["old_text"] == "default" and not change["unchanged"]
    assert wf.prepare_setting(store, "group", "grp_reverse", "model", "default")["unchanged"]
    assert wf.prepare_setting(store, "group", "grp_discovery", "every", "off")["value"] is None
    assert wf.prepare_setting(store, "group", "grp_discovery", "every", "1800s")["unchanged"]
    assert wf.prepare_setting(store, "queue", "q_catalog", "notify", "on")["unchanged"]


def test_set_setting_records_version(store):
    change = wf.prepare_setting(store, "group", "grp_reverse", "count", "8")
    result = store.set_setting(change["section"], change["name"], change["key"], change["value"],
                               reason=change["reason"], by="user")
    assert result == {"changed": True, "version": 2}
    assert store.version_meta(2)["reason"] == "User set miner:grp_reverse count from 4 to 8"
    assert store.definition()["agents"]["grp_reverse"]["count"] == 8


def test_settings_listing(store):
    text = wf.render_settings(store, "group", "grp_reverse", "claude-sonnet-5-5")
    assert "count   4" in text and "model   claude-sonnet-5-5 (Agent Zoo default)" in text
    assert "every   not available" in text
    assert "notify  on" in wf.render_settings(store, "queue", "q_catalog")
    assert "Agent groups" in wf.render_settings(store, "workflow", None)
    assert "model claude-sonnet-5-5" in wf.render_overview(store, default_model="claude-sonnet-5-5")


def test_items_take_finish_move(store):
    a = store.add_item("q_targets", "# ResNet50\nhttps://example", by="user", note="from list")
    b = store.add_item("q_targets", "BERT", by="user")
    assert a["id"] == "item_1" and a["title"] == "ResNet50"
    with store.lock():
        agent = store.new_agent("grp_reverse", version=1, item_id=None, sends_to=["q_catalog", "q_changes"],
                                folder="/tmp", model=None)
        taken = store.take_next("q_targets", agent["id"])
    assert taken["id"] == "item_1" and agent["id"] == "agt_reverse_1"
    with pytest.raises(wf.WorkflowError, match="restart_agent"):
        store.move_items(["item_1"], "q_trash", by="user")
    store.update_agent(agent["id"], item="item_1", state="running")
    out = store.finish_agent(agent["id"], summary="done", outputs=[("q_catalog", "graph.mlir")])
    assert out["created"][0]["from_item"] == "item_1"
    assert store.item("item_1")["state"] == "done"
    with pytest.raises(wf.WorkflowError, match="only send to"):
        store.add_item("q_targets", "x", by=agent["id"], allowed={"q_catalog"})
    moved = store.move_items(["item_1", b["id"]], "q_trash", by="user", note="dup")
    assert all(m["queue"] == "q_trash" and m["state"] == "waiting" for m in moved)
    block = wf.item_block(store, store.item("item_1"))
    assert "from list" in block and "https://example" in block


def test_finish_rejects_unallowed_queue(store):
    with store.lock():
        agent = store.new_agent("grp_reverse", version=1, item_id=None, sends_to=["q_catalog"],
                                folder="/tmp", model=None)
    with pytest.raises(wf.WorkflowError, match="only send to"):
        store.finish_agent(agent["id"], summary="", outputs=[("q_targets", "x")])


def test_effects_mention_running_agents(store):
    with store.lock():
        agent = store.new_agent("grp_reverse", version=1, item_id=None, sends_to=["q_catalog", "q_changes"],
                                folder="/tmp", model=None)
    store.update_agent(agent["id"], state="running")
    store.add_item("q_changes", "fix", by="user")
    new = DEFINITION.replace("  q_changes: {}\n", "").replace("[q_catalog, q_changes]", "[q_catalog]")
    effects = store.update(new, reason="x", by="user", preview_only=True)["effects"]
    text = "\n".join(effects)
    assert "Removes q_changes. Its 1 waiting items stay on disk" in text
    assert "agt_reverse_1 can still send to q_changes" in text
    assert "Running agents agt_reverse_1 keep the current definition" in text


def test_views_render(store):
    store.add_item("q_targets", "ResNet", by="user")
    root = store.root
    assert "miner: paused" in wf.render_view("wf:index", root=root)
    overview = wf.render_view("wf:miner", root=root)
    assert "grp_reverse: 0 of 4 running" in overview and "q_targets: 1 waiting" in overview
    assert wf.render_view("wf:miner:definition", root=root).startswith("version: 1")
    assert "item_1: ResNet" in wf.render_view("wf:miner:q_targets", root=root)
    assert "Reverse engineer this." in wf.render_view("wf:miner:grp_reverse", root=root)
    assert "Text\nResNet" in wf.render_view("wf:miner:item_1", root=root)
    assert "def_1" in wf.render_view("wf:miner:history", root=root)
    assert "item_added" in wf.render_view("wf:miner:log", root=root)
    assert wf.render_view("wf:miner:def_1", root=root).startswith("version: 1")
    assert wf.render_view("buf_123", root=root) is None
    with pytest.raises(wf.WorkflowError, match="no workflow named"):
        wf.render_view("wf:nope", root=root)
