"""Named, persistent multi-agent workflows built from queues and agent groups.

A workflow definition (YAML) declares queues (``q_*``) and agent groups
(``grp_*``). A group with ``takes_from`` starts one agent per item taken from
that queue; a group without it keeps ``count`` agents running from the same
instructions. Agents send results to the queues in their group's ``sends_to``.
The scheduler runs inside the session that controls the workflow.

The first half of this module is a harness-free store; the Agent Zoo plugin
surface (tools, commands, buffers, scheduler) follows it.
"""
from __future__ import annotations

import contextlib
import difflib
import fcntl
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

PLUGIN_NAME = "azo-plugin-workflows"
TRASH = "q_trash"

WORKFLOW_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
QUEUE_ID = re.compile(r"q_[a-z0-9][a-z0-9_]{0,63}")
GROUP_ID = re.compile(r"grp_[a-z0-9][a-z0-9_]{0,63}")
AGENT_ID = re.compile(r"agt_(?P<group>[a-z0-9][a-z0-9_]*?)_(?P<number>\d+)")
ITEM_ID = re.compile(r"item_(?P<number>\d+)")
VERSION_ID = re.compile(r"def_(?P<number>\d+)")

ACTIVE_STATES = {"starting", "running", "paused"}
AGENT_STATES = ACTIVE_STATES | {"finished", "stopped", "attention", "restarted"}

TOP_KEYS = {"version", "folder", "model", "queues", "agents"}
QUEUE_KEYS = {"notify_on_new"}
GROUP_KEYS = {"count", "takes_from", "sends_to", "every", "instructions", "folder", "model"}

EXAMPLE_DEFINITION = """\
folder: ~/work/papers            # working directory for agents (optional)
queues:
  q_papers: {}
  q_summaries: { notify_on_new: true }   # wake the controlling agent on new items
agents:
  grp_finder:                    # no takes_from: keeps `count` agents running
    count: 1
    every: 30m                   # start a new one at most every 30 minutes
    sends_to: [q_papers]
    instructions: |
      Find recent papers on sparse attention. Read wf:{{workflow_name}}:q_papers
      first and skip anything already there. Add one item per paper.
  grp_reader:                    # takes_from: one agent per item
    count: 3
    takes_from: q_papers
    sends_to: [q_summaries]
    instructions: |
      Summarize this paper in one page.
      {{item}}
"""


class WorkflowError(ValueError):
    """A request that cannot be carried out, explained in plain words."""


class StaleDefinition(WorkflowError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Names and IDs
# ---------------------------------------------------------------------------

def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower()).strip("_")


def check_workflow_name(value) -> str:
    name = str(value or "").strip()
    if not WORKFLOW_NAME.fullmatch(name):
        hint = slug(name)
        raise WorkflowError(
            f'"{name}" is not a valid workflow name: use lowercase letters, digits and '
            f'underscores, starting with a letter' + (f' (for example "{hint}")' if hint else "") + ".")
    if name == "index":
        raise WorkflowError('"index" is reserved for the wf:index view; choose another workflow name')
    return name


def suggest_id(prefix: str, value: str) -> str:
    base = slug(value)
    for other in ("q_", "grp_", "agt_", "item_", "def_"):
        if base.startswith(other):
            base = base[len(other):]
    return f"{prefix}{base}" if base else f"{prefix}name"


def check_queue_id(value, known=None, *, what="queue") -> str:
    queue = str(value or "").strip()
    if not QUEUE_ID.fullmatch(queue):
        raise WorkflowError(f'unknown {what} "{queue}"; queue names start with q_ '
                            f'(did you mean {suggest_id("q_", queue)}?)')
    if known is not None and queue not in known:
        options = ", ".join(sorted(known)) or "none"
        raise WorkflowError(f'unknown {what} "{queue}"; this workflow has: {options}')
    return queue


def check_group_id(value, known=None) -> str:
    group = str(value or "").strip()
    if not GROUP_ID.fullmatch(group):
        raise WorkflowError(f'unknown agent group "{group}"; group names start with grp_ '
                            f'(did you mean {suggest_id("grp_", group)}?)')
    if known is not None and group not in known:
        options = ", ".join(sorted(known)) or "none"
        raise WorkflowError(f'unknown agent group "{group}"; this workflow has: {options}')
    return group


def check_item_id(value) -> str:
    item = str(value or "").strip()
    if not ITEM_ID.fullmatch(item):
        digits = re.sub(r"\D", "", item)
        hint = f" (did you mean item_{digits}?)" if digits else ""
        raise WorkflowError(f'"{item}" is not an item ID; item IDs look like item_42{hint}')
    return item


def check_agent_id(value) -> str:
    agent = str(value or "").strip()
    if not AGENT_ID.fullmatch(agent):
        raise WorkflowError(f'"{agent}" is not an agent ID; agent IDs look like agt_reverse_3')
    return agent


def group_short(group: str) -> str:
    return group[len("grp_"):]


def parse_duration(value) -> int:
    """Parse ``30s``, ``30m``, ``2h``, ``1d`` or a number of seconds."""
    if isinstance(value, bool):
        raise ValueError
    if isinstance(value, (int, float)):
        seconds = int(value)
    else:
        match = re.fullmatch(r"\s*(\d+)\s*([smhd]?)\s*", str(value))
        if not match:
            raise ValueError
        seconds = int(match[1]) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match[2]]
    if seconds <= 0:
        raise ValueError
    return seconds


def item_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line if len(line) <= 100 else line[:99] + "…"
    return "(empty)"


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

def split_version(text: str) -> tuple[int | None, str]:
    """Return the managed ``version:`` value and the text without that line."""
    found = None
    kept = []
    for line in text.splitlines(keepends=True):
        match = re.fullmatch(r"version:\s*(?:def_)?(\d+)\s*(#.*)?\n?", line)
        if match and found is None:
            found = int(match[1])
            continue
        kept.append(line)
    body = "".join(kept).lstrip("\n")
    if body and not body.endswith("\n"):
        body += "\n"
    return found, body


def parse_definition(text: str) -> dict:
    """Validate YAML text and return a normalized definition.

    Raises WorkflowError listing every problem found.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise WorkflowError(f"The definition is not valid YAML: {exc}") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise WorkflowError("The definition must be a YAML mapping with queues: and agents:.")
    problems: list[str] = []
    for key in raw:
        if key not in TOP_KEYS:
            problems.append(f'unknown top-level setting "{key}"; expected one of: '
                            + ", ".join(sorted(TOP_KEYS - {"version"})))

    def optional_text(owner, key, value):
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{owner}: {key} must be text")
            return None
        return value.strip()

    raw_queues = raw.get("queues") or {}
    queues: dict[str, dict] = {}
    if not isinstance(raw_queues, dict):
        problems.append("queues: must be a mapping of queue names to settings, e.g. q_targets: {}")
        raw_queues = {}
    for name, settings in raw_queues.items():
        name = str(name)
        if not QUEUE_ID.fullmatch(name):
            problems.append(f'queue "{name}": queue names start with q_ '
                            f'(did you mean {suggest_id("q_", name)}?)')
            continue
        if name == TRASH:
            problems.append(f"{TRASH} always exists; remove it from queues:")
            continue
        settings = settings or {}
        if not isinstance(settings, dict):
            problems.append(f"{name}: settings must be a mapping, e.g. {name}: {{}}")
            continue
        for key in settings:
            if key not in QUEUE_KEYS:
                problems.append(f'{name}: unknown setting "{key}"; expected one of: '
                                + ", ".join(sorted(QUEUE_KEYS)))
        notify = settings.get("notify_on_new", False)
        if not isinstance(notify, bool):
            problems.append(f"{name}: notify_on_new must be true or false")
            notify = False
        queues[name] = {"notify_on_new": notify}
    known_queues = set(queues) | {TRASH}

    def queue_ref(owner, key, value):
        value = str(value)
        if not QUEUE_ID.fullmatch(value):
            problems.append(f'{owner}: {key} "{value}" is not a queue name; queue names start '
                            f'with q_ (did you mean {suggest_id("q_", value)}?)')
            return None
        if value not in known_queues:
            problems.append(f'{owner}: {key} "{value}" is not listed under queues:')
            return None
        return value

    raw_groups = raw.get("agents") or {}
    groups: dict[str, dict] = {}
    if not isinstance(raw_groups, dict):
        problems.append("agents: must be a mapping of agent group names to settings")
        raw_groups = {}
    if not raw_groups:
        problems.append("agents: needs at least one agent group, e.g. grp_reader:")
    for name, settings in raw_groups.items():
        name = str(name)
        if not GROUP_ID.fullmatch(name):
            problems.append(f'agent group "{name}": group names start with grp_ '
                            f'(did you mean {suggest_id("grp_", name)}?)')
            continue
        if not isinstance(settings, dict):
            problems.append(f"{name}: settings must be a mapping with instructions:")
            continue
        for key in settings:
            if key not in GROUP_KEYS:
                problems.append(f'{name}: unknown setting "{key}"; expected one of: '
                                + ", ".join(sorted(GROUP_KEYS)))
        count = settings.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 64:
            problems.append(f"{name}: count must be a whole number from 0 to 64")
            count = 0
        takes_from = settings.get("takes_from")
        if takes_from is not None:
            takes_from = queue_ref(name, "takes_from", takes_from)
            if takes_from == TRASH:
                problems.append(f"{name}: cannot take items from {TRASH}")
                takes_from = None
        sends_to = settings.get("sends_to") or []
        if isinstance(sends_to, str):
            sends_to = [sends_to]
        if not isinstance(sends_to, list):
            problems.append(f"{name}: sends_to must be a list of queues, e.g. [q_results]")
            sends_to = []
        sends = []
        for queue in sends_to:
            queue = queue_ref(name, "sends_to", queue)
            if queue and queue not in sends:
                sends.append(queue)
        every = settings.get("every")
        every_seconds = None
        if every is not None:
            if settings.get("takes_from") is not None:
                problems.append(f"{name}: every only applies to groups without takes_from")
            else:
                try:
                    every_seconds = parse_duration(every)
                except ValueError:
                    problems.append(f'{name}: every "{every}" is not a duration like 45s, 30m, 2h or 1d')
        instructions = settings.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            problems.append(f"{name}: instructions are required")
            instructions = ""
        groups[name] = {
            "count": count, "takes_from": takes_from, "sends_to": sends,
            "every": every_seconds, "every_text": None if every is None else str(every),
            "instructions": instructions,
            "folder": optional_text(name, "folder", settings.get("folder")),
            "model": optional_text(name, "model", settings.get("model")),
        }
    folder = optional_text("definition", "folder", raw.get("folder"))
    model = optional_text("definition", "model", raw.get("model"))
    if problems:
        raise WorkflowError("The definition has problems:\n" + "\n".join(f"- {p}" for p in problems))
    return {"folder": folder, "model": model, "queues": queues, "agents": groups}


def set_group_count_text(text: str, group: str, count: int) -> str:
    """Change one group's count in definition text, preserving formatting."""
    root = yaml.compose(text)
    agents = next((value for key, value in root.value if key.value == "agents"), None)
    node = next((value for key, value in agents.value if key.value == group), None)
    if node is None or not isinstance(node, yaml.MappingNode):
        raise WorkflowError(f"{group} is not in the definition")
    current = next((value for key, value in node.value if key.value == "count"), None)
    if current is not None:
        start, end = current.start_mark.index, current.end_mark.index
        return text[:start] + str(count) + text[end:]
    if node.flow_style:
        raise WorkflowError(f"{group} is written in flow style; edit its count with update_workflow")
    first = node.value[0][0].start_mark
    line_start = first.index - first.column
    return text[:line_start] + " " * first.column + f"count: {count}\n" + text[line_start:]


def unified_diff(old: str, new: str, old_label: str, new_label: str) -> str:
    lines = list(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                      old_label, new_label))
    return "".join(line if line.endswith("\n") else line + "\n" for line in lines) or "(no changes)\n"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def state_root() -> Path:
    for variable in ("AGENT_ZOO_HOME", "AGENT_ZOO_STATE_ROOT"):
        if os.environ.get(variable):
            return Path(os.environ[variable]).expanduser()
    return Path.home() / ".local" / "share" / "agent-zoo"


def workflows_root() -> Path:
    return state_root() / "plugin-configs" / PLUGIN_NAME / "workflows"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value) -> None:
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def list_workflow_names(root: Path | None = None) -> list[str]:
    root = root or workflows_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and WORKFLOW_NAME.fullmatch(p.name) and (p / "control.json").exists())


class WorkflowStore:
    """Files for one workflow. Mutations hold an exclusive per-workflow lock."""

    def __init__(self, name: str, root: Path | None = None):
        self.name = check_workflow_name(name)
        self.root = root or workflows_root()
        self.dir = self.root / self.name

    # -- basics -------------------------------------------------------------

    def exists(self) -> bool:
        return (self.dir / "control.json").exists()

    def require(self) -> "WorkflowStore":
        if not self.exists():
            names = list_workflow_names(self.root)
            known = f"; existing workflows: {', '.join(names)}" if names else "; there are no workflows yet"
            raise WorkflowError(f'there is no workflow named "{self.name}"{known}')
        return self

    @contextlib.contextmanager
    def lock(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / ".lock", "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def counters(self) -> dict:
        return read_json(self.dir / "counters.json", {"item": 0, "version": 0, "agents": {}})

    def next_number(self, kind: str, group: str | None = None) -> int:
        counters = self.counters()
        if group is None:
            counters[kind] = counters.get(kind, 0) + 1
            number = counters[kind]
        else:
            counters.setdefault("agents", {})[group] = counters.get("agents", {}).get(group, 0) + 1
            number = counters["agents"][group]
        write_json(self.dir / "counters.json", counters)
        return number

    def log(self, event: str, **fields) -> None:
        record = {"at": now_iso(), "event": event, **fields}
        with open(self.dir / "log.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_records(self) -> list[dict]:
        try:
            lines = (self.dir / "log.jsonl").read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [json.loads(line) for line in lines if line.strip()]

    # -- control ------------------------------------------------------------

    def control(self) -> dict:
        return read_json(self.dir / "control.json", {})

    def update_control(self, **changes) -> dict:
        control = self.control()
        control.update(changes)
        write_json(self.dir / "control.json", control)
        return control

    # -- definitions ----------------------------------------------------------

    @property
    def version(self) -> int:
        return self.counters().get("version", 0)

    def definition_body(self, version: int | None = None) -> str:
        version = version or self.version
        path = self.dir / "versions" / f"def_{version}.yaml"
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise WorkflowError(f"{self.name} has no definition version def_{version}") from None

    def definition_text(self, version: int | None = None) -> str:
        version = version or self.version
        return f"version: {version}\n" + self.definition_body(version)

    def definition(self, version: int | None = None) -> dict:
        return parse_definition(self.definition_body(version))

    def version_meta(self, version: int) -> dict:
        return read_json(self.dir / "versions" / f"def_{version}.json", {})

    def _write_version(self, body: str, reason: str, by: str) -> int:
        version = self.next_number("version")
        atomic_write(self.dir / "versions" / f"def_{version}.yaml", body)
        write_json(self.dir / "versions" / f"def_{version}.json",
                   {"version": version, "reason": reason, "by": by, "at": now_iso()})
        self.log("definition_changed", version=f"def_{version}", reason=reason, by=by)
        return version

    def create(self, text: str, *, by: str, controller: dict | None = None) -> dict:
        _, body = split_version(text)
        definition = parse_definition(body)
        with self.lock():
            if self.exists():
                raise WorkflowError(f'a workflow named "{self.name}" already exists; '
                                    "use update_workflow to change it")
            (self.dir / "items").mkdir(parents=True, exist_ok=True)
            (self.dir / "agents").mkdir(parents=True, exist_ok=True)
            write_json(self.dir / "counters.json", {"item": 0, "version": 0, "agents": {}})
            self._write_version(body, "Created", by)
            write_json(self.dir / "control.json", {
                "state": "paused", "paused_groups": [], "epoch": 0,
                "controller": controller, "created_at": now_iso(), "last_started": {},
            })
            self.log("created", by=by)
        return definition

    def update(self, text: str, *, reason: str, by: str, preview_only: bool = False) -> dict:
        """Replace the definition. Returns diff, effects and the new version."""
        stated, body = split_version(text)
        new = parse_definition(body)
        with self.lock():
            self.require()
            current = self.version
            if stated is not None and stated != current:
                raise StaleDefinition(
                    f"This definition is based on def_{stated}, but {self.name} is now at "
                    f"def_{current}. Copy wf:{self.name}:definition again and reapply your change. "
                    f"Changes since your copy:\n"
                    + unified_diff(self.definition_body(stated), self.definition_body(current),
                                   f"def_{stated}", f"def_{current}"))
            old_body = self.definition_body(current)
            diff = unified_diff(old_body, body, f"def_{current}", "new definition")
            effects = self.update_effects(self.definition(current), new)
            result = {"diff": diff, "effects": effects, "from_version": current,
                      "changed": old_body != body, "version": current}
            if preview_only or old_body == body:
                return result
            result["version"] = self._write_version(body, reason, by)
            return result

    def set_count(self, group: str, count: int, *, by: str) -> dict:
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 64:
            raise WorkflowError("count must be a whole number from 0 to 64")
        with self.lock():
            self.require()
            definition = self.definition()
            check_group_id(group, definition["agents"])
            old = definition["agents"][group]["count"]
            if old == count:
                return {"version": self.version, "old": old, "changed": False}
            body = set_group_count_text(self.definition_body(), group, count)
            parse_definition(body)
            version = self._write_version(body, f"Set {group} count from {old} to {count}", by)
            return {"version": version, "old": old, "changed": True}

    def update_effects(self, old: dict, new: dict) -> list[str]:
        effects = []
        active = [a for a in self.agents() if a["state"] in ACTIVE_STATES]
        items = self.items()
        waiting = {}
        for item in items:
            if item["state"] == "waiting":
                waiting[item["queue"]] = waiting.get(item["queue"], 0) + 1
        for queue in sorted(set(new["queues"]) - set(old["queues"])):
            effects.append(f"Adds {queue}.")
        for queue in sorted(set(old["queues"]) - set(new["queues"])):
            count = waiting.get(queue, 0)
            kept = f" Its {count} waiting items stay on disk and return if it is added back." if count else ""
            effects.append(f"Removes {queue}.{kept}")
        for group in sorted(set(new["agents"]) - set(old["agents"])):
            effects.append(f"Adds {group} ({describe_group(new['agents'][group])}).")
        for group in sorted(set(old["agents"]) - set(new["agents"])):
            running = [a["id"] for a in active if a["group"] == group]
            tail = f" {', '.join(running)} keep running until they finish." if running else ""
            effects.append(f"Removes {group}; no new agents will start in it.{tail}")
        for group in sorted(set(old["agents"]) & set(new["agents"])):
            before, after = old["agents"][group], new["agents"][group]
            changes = []
            if before["count"] != after["count"]:
                changes.append(f"count {before['count']} → {after['count']}")
            if before["takes_from"] != after["takes_from"]:
                changes.append(f"takes_from {before['takes_from'] or 'nothing'} → {after['takes_from'] or 'nothing'}")
            if before["sends_to"] != after["sends_to"]:
                changes.append(f"sends_to [{', '.join(before['sends_to'])}] → [{', '.join(after['sends_to'])}]")
            if before["every"] != after["every"]:
                changes.append(f"every {before['every_text'] or 'unset'} → {after['every_text'] or 'unset'}")
            if before["instructions"] != after["instructions"]:
                changes.append("new instructions")
            if before["folder"] != after["folder"] or before["model"] != after["model"]:
                changes.append("new folder or model")
            if not changes:
                continue
            running = [a["id"] for a in active if a["group"] == group]
            tail = (f" Running agents {', '.join(running)} keep the current definition; "
                    "the change applies to agents started afterward.") if running else ""
            effects.append(f"Changes {group}: {'; '.join(changes)}.{tail}")
        for agent in active:
            orphaned = [q for q in agent.get("sends_to", []) if q != TRASH and q not in new["queues"]]
            if orphaned:
                effects.append(f"{agent['id']} can still send to {', '.join(orphaned)}, which will no longer "
                               "be in the definition; those items will wait until the queue is added back.")
        taken = {g["takes_from"] for g in new["agents"].values() if g["takes_from"]}
        for queue in sorted(new["queues"]):
            if waiting.get(queue) and queue not in taken:
                effects.append(f"{queue} has {waiting[queue]} waiting items and no group takes from it.")
        return effects or ["No effect on running agents or waiting items."]

    # -- items ----------------------------------------------------------------

    def item_paths(self, item_id: str) -> tuple[Path, Path]:
        return self.dir / "items" / f"{item_id}.md", self.dir / "items" / f"{item_id}.json"

    def item(self, item_id: str) -> dict:
        item_id = check_item_id(item_id)
        meta = read_json(self.item_paths(item_id)[1])
        if meta is None:
            raise WorkflowError(f"{self.name} has no {item_id}")
        return meta

    def item_text(self, item_id: str) -> str:
        return self.item_paths(check_item_id(item_id))[0].read_text(encoding="utf-8")

    def items(self, queue: str | None = None) -> list[dict]:
        folder = self.dir / "items"
        if not folder.is_dir():
            return []
        records = []
        for path in folder.glob("item_*.json"):
            meta = read_json(path)
            if meta and (queue is None or meta["queue"] == queue):
                records.append(meta)
        return sorted(records, key=lambda m: int(m["id"][5:]))

    def _save_item(self, meta: dict) -> None:
        write_json(self.item_paths(meta["id"])[1], meta)

    def add_item(self, queue: str, text: str, *, by: str, note: str | None = None,
                 from_item: str | None = None, allowed: set | None = None) -> dict:
        queue = check_queue_id(queue)
        if allowed is not None and queue not in allowed:
            raise WorkflowError(f"you can only send to: {', '.join(sorted(allowed))}")
        if not text.strip():
            raise WorkflowError("an item cannot be empty")
        with self.lock():
            self.require()
            number = self.next_number("item")
            item_id = f"item_{number}"
            at = now_iso()
            meta = {
                "id": item_id, "title": item_title(text), "queue": queue, "state": "waiting",
                "created_at": at, "created_by": by, "from_item": from_item, "taken_by": None,
                "notes": [{"text": note, "by": by, "at": at}] if note else [],
                "history": [{"at": at, "event": "added", "queue": queue, "by": by}],
            }
            atomic_write(self.item_paths(item_id)[0], text if text.endswith("\n") else text + "\n")
            self._save_item(meta)
            self.log("item_added", item=item_id, queue=queue, by=by)
        return meta

    def move_items(self, item_ids: list[str], to_queue: str, *, by: str, note: str | None = None,
                   allowed: set | None = None) -> list[dict]:
        to_queue = check_queue_id(to_queue)
        if allowed is not None and to_queue not in allowed:
            raise WorkflowError(f"you can only move items into: {', '.join(sorted(allowed))}")
        ids = [check_item_id(i) for i in item_ids]
        if not ids:
            raise WorkflowError("item_ids is empty")
        with self.lock():
            self.require()
            metas = [self.item(i) for i in ids]
            for meta in metas:
                if meta["state"] == "taken":
                    raise WorkflowError(f"{meta['id']} is being worked on by {meta['taken_by']}; "
                                        f"use restart_agent to stop that agent and return the item first")
            at = now_iso()
            for meta in metas:
                meta.update(queue=to_queue, state="waiting", taken_by=None)
                if note:
                    meta["notes"].append({"text": note, "by": by, "at": at})
                meta["history"].append({"at": at, "event": "moved", "queue": to_queue, "by": by})
                self._save_item(meta)
                self.log("item_moved", item=meta["id"], queue=to_queue, by=by)
        return metas

    def take_next(self, queue: str, agent_id: str) -> dict | None:
        """Lease the oldest waiting item in ``queue`` to ``agent_id``. Caller holds the lock."""
        for meta in self.items(queue):
            if meta["state"] == "waiting":
                meta.update(state="taken", taken_by=agent_id)
                meta["history"].append({"at": now_iso(), "event": "taken", "queue": queue, "by": agent_id})
                self._save_item(meta)
                return meta
        return None

    def finish_item(self, item_id: str, agent_id: str) -> None:
        """Caller holds the lock."""
        meta = self.item(item_id)
        if meta["state"] == "taken" and meta["taken_by"] == agent_id:
            meta.update(state="done", taken_by=None)
            meta["history"].append({"at": now_iso(), "event": "done", "queue": meta["queue"], "by": agent_id})
            self._save_item(meta)

    def return_item(self, item_id: str, agent_id: str, *, by: str) -> None:
        """Put a taken item back in its queue. Caller holds the lock."""
        meta = self.item(item_id)
        if meta["state"] == "taken" and meta["taken_by"] == agent_id:
            meta.update(state="waiting", taken_by=None)
            meta["history"].append({"at": now_iso(), "event": "returned", "queue": meta["queue"], "by": by})
            self._save_item(meta)

    # -- agents ---------------------------------------------------------------

    def agent_path(self, agent_id: str) -> Path:
        return self.dir / "agents" / f"{agent_id}.json"

    def agent(self, agent_id: str) -> dict:
        record = read_json(self.agent_path(check_agent_id(agent_id)))
        if record is None:
            raise WorkflowError(f"{self.name} has no agent {agent_id}")
        return record

    def agents(self, group: str | None = None) -> list[dict]:
        folder = self.dir / "agents"
        if not folder.is_dir():
            return []
        records = [read_json(p) for p in folder.glob("agt_*.json")]
        records = [r for r in records if r and (group is None or r["group"] == group)]
        return sorted(records, key=lambda r: (r["group"], r["number"]))

    def agent_by_session(self, session_id: str) -> dict | None:
        return next((a for a in self.agents() if a.get("session_id") == session_id), None)

    def save_agent(self, record: dict) -> None:
        record["updated_at"] = now_iso()
        write_json(self.agent_path(record["id"]), record)

    def new_agent(self, group: str, *, version: int, item_id: str | None, sends_to: list[str],
                  folder: str, model: str | None) -> dict:
        """Allocate an agent record. Caller holds the lock."""
        number = self.next_number("agent", group)
        record = {
            "id": f"agt_{group_short(group)}_{number}", "group": group, "number": number,
            "version": version, "item": item_id, "sends_to": sends_to, "folder": folder,
            "model": model, "state": "starting", "session_id": None, "instance_id": None,
            "created_at": now_iso(), "finished_at": None, "summary": None, "outputs": [],
            "problem": None, "reminders": 0, "resumes": 0,
        }
        self.save_agent(record)
        return record

    def update_agent(self, agent_id: str, **changes) -> dict:
        with self.lock():
            record = self.agent(agent_id)
            record.update(changes)
            self.save_agent(record)
            return record

    def finish_agent(self, agent_id: str, *, summary: str,
                     outputs: list[tuple[str, str]]) -> dict:
        """Record a worker's finish: create output items, mark its item done."""
        with self.lock():
            record = self.agent(agent_id)
            if record["state"] in {"finished", "restarted"}:
                raise WorkflowError(f"{agent_id} has already {record['state']}")
            allowed = set(record.get("sends_to", [])) | {TRASH}
            for queue, _ in outputs:
                check_queue_id(queue)
                if queue not in allowed:
                    raise WorkflowError(f"you can only send to: {', '.join(sorted(allowed))}")
        created = [self.add_item(queue, text, by=agent_id, from_item=record.get("item"),
                                 allowed=allowed) for queue, text in outputs]
        with self.lock():
            record = self.agent(agent_id)
            if record.get("item"):
                self.finish_item(record["item"], agent_id)
            record.update(state="finished", summary=summary, finished_at=now_iso(),
                          outputs=record.get("outputs", []) + [m["id"] for m in created],
                          problem=None)
            self.save_agent(record)
            self.log("agent_finished", agent=agent_id, item=record.get("item"),
                     outputs=[m["id"] for m in created], summary=summary)
        return {"agent": record, "created": created}


def describe_group(group: dict) -> str:
    parts = [f"takes from {group['takes_from']}" if group["takes_from"] else "standing"]
    parts.append(f"count {group['count']}")
    if group["every"]:
        parts.append(f"every {group['every_text']}")
    if group["sends_to"]:
        parts.append(f"sends to {', '.join(group['sends_to'])}")
    return ", ".join(parts)


def item_block(store: WorkflowStore, meta: dict) -> str:
    """The text {{item}} expands to in instructions."""
    lines = [f"{meta['id']} from {meta['queue']}: {meta['title']}"]
    origin = meta.get("created_by") or "unknown"
    if meta.get("from_item"):
        origin += f" while working on {meta['from_item']}"
    lines.append(f"Added by {origin}.")
    if meta.get("notes"):
        lines.append("Notes:")
        lines.extend(f"- {n['text']} ({n['by']})" for n in meta["notes"])
    lines.append("--- item text ---")
    lines.append(store.item_text(meta["id"]).rstrip("\n"))
    lines.append("--- end of item ---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Read-only views (the wf: buffer namespace)
# ---------------------------------------------------------------------------

def controller_label(control: dict, viewer_session_id: str = "") -> str:
    controller = control.get("controller") or {}
    session = controller.get("session_id")
    if not session:
        return "no session"
    if session == viewer_session_id:
        return "this session"
    return f"session {session[:8]}"


def state_label(control: dict) -> str:
    state = control.get("state", "paused")
    paused = control.get("paused_groups") or []
    if state == "running" and paused:
        return f"running ({', '.join(paused)} paused)"
    return state


def queue_counts(items: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for item in items:
        bucket = counts.setdefault(item["queue"], {"waiting": 0, "taken": 0, "done": 0})
        bucket[item["state"]] = bucket.get(item["state"], 0) + 1
    return counts


def render_index(root: Path | None = None, viewer_session_id: str = "") -> str:
    names = list_workflow_names(root)
    lines = ["Workflows. Open one with wf:<name>, e.g. wf:" + (names[0] if names else "miner") + "."]
    if not names:
        lines.append("There are no workflows yet. Create one with create_workflow.")
    for name in names:
        store = WorkflowStore(name, root)
        control = store.control()
        active = sum(a["state"] in ACTIVE_STATES for a in store.agents())
        waiting = sum(i["state"] == "waiting" for i in store.items() if i["queue"] != TRASH)
        attention = sum(a["state"] == "attention" for a in store.agents())
        line = (f"{name}: {state_label(control)}, controlled by {controller_label(control, viewer_session_id)}, "
                f"def_{store.version}; {active} agents active, {waiting} items waiting")
        if attention:
            line += f", {attention} agents need attention"
        lines.append(line)
    return "\n".join(lines) + "\n"


def render_overview(store: WorkflowStore, viewer_session_id: str = "") -> str:
    control = store.control()
    definition = store.definition()
    agents = store.agents()
    items = store.items()
    counts = queue_counts(items)
    name = store.name
    lines = [f"Workflow {name}: {state_label(control)}, controlled by "
             f"{controller_label(control, viewer_session_id)}, definition def_{store.version}.", "",
             "Agent groups"]
    paused_groups = set(control.get("paused_groups") or [])
    for group, settings in definition["agents"].items():
        mine = [a for a in agents if a["group"] == group]
        states = {s: sum(a["state"] == s for a in mine) for s in ("starting", "running", "paused", "finished")}
        active = states["starting"] + states["running"]
        status = f"{active} of {settings['count']} running"
        if states["paused"]:
            status += f", {states['paused']} paused"
        status += f", {states['finished']} finished"
        if group in paused_groups:
            status += " (group paused)"
        lines.append(f"  {group}: {status}. {describe_group(settings)}.")
    lines += ["", "Queues"]
    queue_names = list(definition["queues"]) + [q for q in counts if q not in definition["queues"]]
    for queue in queue_names:
        bucket = counts.get(queue, {"waiting": 0, "taken": 0, "done": 0})
        extra = "" if queue in definition["queues"] or queue == TRASH else " (not in the current definition)"
        lines.append(f"  {queue}: {bucket['waiting']} waiting, {bucket['taken']} being worked on, "
                     f"{bucket['done']} done{extra}")
    attention = [a for a in agents if a["state"] == "attention"]
    if attention:
        lines += ["", "Needs attention"]
        for agent in attention:
            lines.append(f"  {agent['id']}: {agent.get('problem') or 'unknown problem'}")
    recent = store.log_records()[-8:]
    if recent:
        lines += ["", "Recent events (newest first; full log in wf:" + name + ":log)"]
        lines.extend("  " + describe_event(r) for r in reversed(recent))
    lines += ["", f"More: wf:{name}:definition, wf:{name}:history, wf:{name}:<q_queue>, "
                  f"wf:{name}:<grp_group>, wf:{name}:<agt_agent>, wf:{name}:<item_N>"]
    return "\n".join(lines) + "\n"


def describe_event(record: dict) -> str:
    fields = {k: v for k, v in record.items() if k not in {"at", "event"}}
    detail = ", ".join(f"{k}={', '.join(v) if isinstance(v, list) else v}" for k, v in fields.items()
                       if v not in (None, "", []))
    return f"{record['at']} {record['event']}" + (f" ({detail})" if detail else "")


def render_queue(store: WorkflowStore, queue: str) -> str:
    definition = store.definition()
    items = store.items(queue)
    if queue not in definition["queues"] and queue != TRASH and not items:
        return f"{queue} is not in {store.name}'s definition and has no items.\n"
    counts = queue_counts(items).get(queue, {"waiting": 0, "taken": 0, "done": 0})
    takers = [g for g, s in definition["agents"].items() if s["takes_from"] == queue]
    senders = [g for g, s in definition["agents"].items() if queue in s["sends_to"]]
    lines = [f"{queue} in {store.name}: {counts['waiting']} waiting, {counts['taken']} being worked on, "
             f"{counts['done']} done."]
    lines.append(f"Taken from by: {', '.join(takers) or 'no group'}. Sent to by: {', '.join(senders) or 'no group'}.")
    if queue not in definition["queues"] and queue != TRASH:
        lines.append("This queue is not in the current definition; its items wait until it is added back.")

    def row(item, suffix=""):
        notes = f" [{len(item['notes'])} notes]" if item.get("notes") else ""
        return f"  {item['id']}: {item['title']} (added by {item['created_by']}){notes}{suffix}"

    waiting = [i for i in items if i["state"] == "waiting"]
    taken = [i for i in items if i["state"] == "taken"]
    done = [i for i in items if i["state"] == "done"]
    lines += ["", "Waiting (oldest first)"] + ([row(i) for i in waiting] or ["  none"])
    lines += ["", "Being worked on"] + ([row(i, f" → {i['taken_by']}") for i in taken] or ["  none"])
    lines += ["", f"Done (newest first{', last 50' if len(done) > 50 else ''})"]
    lines += [row(i) for i in reversed(done[-50:])] or ["  none"]
    lines += ["", f"Open an item with wf:{store.name}:item_N."]
    return "\n".join(lines) + "\n"


def render_group(store: WorkflowStore, group: str) -> str:
    definition = store.definition()
    agents = store.agents(group)
    settings = definition["agents"].get(group)
    if settings is None and not agents:
        return f"{group} is not in {store.name}'s definition and has no agents.\n"
    lines = [f"{group} in {store.name}: " + (describe_group(settings) + "." if settings
                                             else "not in the current definition.")]
    if group in (store.control().get("paused_groups") or []):
        lines.append("This group is paused.")
    for state in ("starting", "running", "paused", "attention", "stopped", "finished", "restarted"):
        mine = [a for a in agents if a["state"] == state]
        if not mine:
            continue
        lines += ["", state.capitalize()]
        for agent in (reversed(mine[-30:]) if state in {"finished", "restarted"} else mine):
            item = f" on {agent['item']}" if agent.get("item") else ""
            note = f": {agent['summary']}" if agent.get("summary") else (
                f": {agent['problem']}" if agent.get("problem") else "")
            lines.append(f"  {agent['id']}{item} (def_{agent['version']}){note}")
    if settings:
        lines += ["", "Instructions", settings["instructions"].rstrip("\n")]
    return "\n".join(lines) + "\n"


def render_agent(store: WorkflowStore, agent_id: str, transcript=None) -> str:
    agent = store.agent(agent_id)
    lines = [f"{agent['id']} in {store.name} ({agent['group']}, def_{agent['version']}): {agent['state']}."]
    if agent.get("item"):
        lines.append(f"Item: {agent['item']} (wf:{store.name}:{agent['item']})")
    lines.append(f"Can send to: {', '.join(agent.get('sends_to') or []) or 'nothing'}")
    lines.append(f"Folder: {agent.get('folder')}")
    if agent.get("session_id"):
        lines.append(f"Session: {agent['session_id']}")
    if agent.get("problem"):
        lines.append(f"Problem: {agent['problem']}")
    if agent.get("summary"):
        lines.append(f"Summary: {agent['summary']}")
    if agent.get("outputs"):
        lines.append(f"Outputs: {', '.join(agent['outputs'])}")
    if transcript is not None and agent.get("session_id"):
        try:
            text = transcript(agent["session_id"])
        except Exception as exc:  # A missing transcript should not hide the record.
            text = f"(transcript unavailable: {exc})"
        lines += ["", "Transcript", text.rstrip("\n")]
    return "\n".join(lines) + "\n"


def render_item(store: WorkflowStore, item_id: str) -> str:
    meta = store.item(item_id)
    lines = [f"{meta['id']} in {store.name}: {meta['title']}",
             f"Queue: {meta['queue']}; " + (f"being worked on by {meta['taken_by']}"
                                            if meta["state"] == "taken" else meta["state"]) + "."]
    origin = f"Added by {meta['created_by']}"
    if meta.get("from_item"):
        origin += f" while working on {meta['from_item']}"
    lines.append(origin + ".")
    made = [i["id"] for i in store.items() if i.get("from_item") == meta["id"]]
    if made:
        lines.append(f"Items made from it: {', '.join(made)}")
    if meta.get("notes"):
        lines += ["", "Notes"] + [f"  {n['at']} {n['by']}: {n['text']}" for n in meta["notes"]]
    lines += ["", "History"] + [f"  {h['at']} {h['event']} {h['queue']} by {h['by']}" for h in meta["history"]]
    lines += ["", "Text", store.item_text(meta["id"]).rstrip("\n")]
    return "\n".join(lines) + "\n"


def render_history(store: WorkflowStore) -> str:
    lines = [f"Definition history for {store.name}, newest first. Open one with wf:{store.name}:def_N."]
    for version in range(store.version, 0, -1):
        meta = store.version_meta(version)
        lines.append(f"  def_{version} {meta.get('at', '')} by {meta.get('by', '?')}: {meta.get('reason', '')}")
    return "\n".join(lines) + "\n"


def render_log(store: WorkflowStore) -> str:
    records = store.log_records()
    return (f"Event log for {store.name}, newest first.\n"
            + "\n".join(describe_event(r) for r in reversed(records)) + "\n")


def render_view(buffer_id: str, *, viewer_session_id: str = "", transcript=None,
                root: Path | None = None) -> str | None:
    """Text for a wf: buffer, or None when the ID is not a workflow view."""
    parts = buffer_id.split(":")
    if not parts or parts[0] != "wf" or len(parts) not in (2, 3):
        return None
    if parts[1] == "index" and len(parts) == 2:
        return render_index(root, viewer_session_id)
    store = WorkflowStore(parts[1], root).require()
    if len(parts) == 2:
        return render_overview(store, viewer_session_id)
    view = parts[2]
    if view == "definition":
        return store.definition_text()
    if view == "history":
        return render_history(store)
    if view == "log":
        return render_log(store)
    if match := VERSION_ID.fullmatch(view):
        return store.definition_text(int(match["number"]))
    if QUEUE_ID.fullmatch(view):
        return render_queue(store, view)
    if GROUP_ID.fullmatch(view):
        return render_group(store, view)
    if AGENT_ID.fullmatch(view):
        return render_agent(store, view, transcript)
    if ITEM_ID.fullmatch(view):
        return render_item(store, view)
    raise WorkflowError(f'unknown view "{view}"; use definition, history, log, def_N, q_*, grp_*, agt_* or item_N')
