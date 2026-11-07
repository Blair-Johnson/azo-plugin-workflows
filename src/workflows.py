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

RESERVED_NAMES = {"index", "list", "view", "run", "pause", "set", "rename", "archive", "unarchive", "archived"}

TOP_KEYS = {"version", "description", "folder", "model", "queues", "agents"}
QUEUE_KEYS = {"notify_on_new"}
GROUP_KEYS = {"count", "takes_from", "sends_to", "every", "instructions", "folder", "model"}

EXAMPLE_DEFINITION = """\
description: Summarize new sparse-attention papers   # one line, shown in /workflow list
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
    if name in RESERVED_NAMES:
        raise WorkflowError(f'"{name}" is reserved by the /workflow command and wf:index; '
                            "choose another workflow name")
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
    description = optional_text("definition", "description", raw.get("description"))
    if problems:
        raise WorkflowError("The definition has problems:\n" + "\n".join(f"- {p}" for p in problems))
    return {"description": description, "folder": folder, "model": model, "queues": queues, "agents": groups}


def yaml_scalar(value) -> str:
    lines = yaml.safe_dump(value, default_flow_style=True, width=1 << 20).splitlines()
    if lines and lines[-1] == "...":
        lines = lines[:-1]
    return "\n".join(lines)


def yaml_flow(mapping: dict) -> str:
    if not mapping:
        return "{}"
    return "{ " + ", ".join(f"{key}: {yaml_scalar(value)}" for key, value in mapping.items()) + " }"


def _mapping_get(node, key):
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
                return key_node, value_node
    return None, None


def set_setting_text(text: str, section: str | None, name: str | None, key: str, value) -> str:
    """Set (or, with value None, remove) one simple setting in definition text.

    ``section`` is None for a top-level setting, or "agents"/"queues" with ``name``
    naming the group or queue. Everything else in the text, including comments and
    instructions, is preserved.
    """
    root = yaml.compose(text)
    if not isinstance(root, yaml.MappingNode):
        raise WorkflowError("the definition is not a YAML mapping")
    node = root
    if section is not None:
        _, container = _mapping_get(root, section)
        key_node, node = _mapping_get(container, name)
        if node is None:
            raise WorkflowError(f"{name} is not in the definition")
        if isinstance(node, yaml.ScalarNode) and node.tag.endswith(":null"):
            # Written as a bare `q_name:`; give it a small flow mapping.
            line_end = text.find("\n", key_node.end_mark.index)
            line_end = len(text) if line_end == -1 else line_end
            mapping = {} if value is None else {key: value}
            return text[:key_node.end_mark.index] + ": " + yaml_flow(mapping) + text[line_end:]
        if not isinstance(node, yaml.MappingNode):
            raise WorkflowError(f"{name} must be a mapping of settings")
    if node.flow_style:
        start, end = node.start_mark.index, node.end_mark.index
        mapping = yaml.safe_load(text[start:end]) or {}
        if value is None:
            mapping.pop(key, None)
        else:
            mapping[key] = value
        return text[:start] + yaml_flow(mapping) + text[end:]
    key_node, current = _mapping_get(node, key)
    if current is not None:
        start, end = current.start_mark.index, current.end_mark.index
        block = isinstance(current, yaml.ScalarNode) and current.style in ("|", ">")
        if value is None:
            line_start = key_node.start_mark.index - key_node.start_mark.column
            if not block:
                newline = text.find("\n", end)
                end = len(text) if newline == -1 else newline + 1
            return text[:line_start] + text[end:]
        return text[:start] + yaml_scalar(value) + ("\n" if block else "") + text[end:]
    if value is None:
        return text
    first = node.value[0][0].start_mark
    line_start = first.index - first.column
    return text[:line_start] + " " * first.column + f"{key}: {yaml_scalar(value)}\n" + text[line_start:]


def unified_diff(old: str, new: str, old_label: str, new_label: str) -> str:
    lines = list(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                      old_label, new_label))
    return "".join(line if line.endswith("\n") else line + "\n" for line in lines) or "(no changes)\n"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def state_root() -> Path:
    return Path(resolve_state_root())  # Imported with the harness section below.


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

    def last_activity(self) -> str:
        """Timestamp of the newest logged event, or the creation time."""
        try:
            lines = (self.dir / "log.jsonl").read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            lines = []
        for line in reversed(lines):
            if line.strip():
                return json.loads(line).get("at", "")
        return self.control().get("created_at", "")

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

    def rename(self, new_name: str, *, by: str) -> "WorkflowStore":
        """Move this workflow to a new name. Returns the store under the new name.

        Worker sessions are durably bound to ``wf:<name>:<agent>``, so renaming is
        refused while the workflow runs or still has agents that could resume.
        """
        target = WorkflowStore(new_name, self.root)
        if target.name == self.name:
            raise WorkflowError(f"{self.name} already has that name")
        with self.lock():
            self.require()
            if target.exists():
                raise WorkflowError(f'a workflow named "{target.name}" already exists; choose another name')
            if self.control().get("state") == "running":
                raise WorkflowError(f"{self.name} is running; pause it first with /workflow pause {self.name}")
            unfinished = [a["id"] for a in self.agents() if a["state"] in ACTIVE_STATES | {"attention"}]
            if unfinished:
                raise WorkflowError(
                    f"{', '.join(unfinished)} {'is' if len(unfinished) == 1 else 'are'} not finished, and their "
                    f"sessions are tied to the name {self.name}. Run {self.name} until they finish, or use "
                    "restart_agent on them, then rename it")
            if target.dir.exists():
                # A bare directory (at most a stale lock file) is left by lookups of missing names.
                if {p.name for p in target.dir.iterdir()} - {".lock"}:
                    raise WorkflowError(f"{target.dir} already exists and is not a workflow; move it aside first")
                (target.dir / ".lock").unlink(missing_ok=True)
                target.dir.rmdir()
            self.log("renamed", old_name=self.name, new_name=target.name, by=by)
            os.rename(self.dir, target.dir)
        return target

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
        definition = self.require().definition()
        check_group_id(group, definition["agents"])
        old = definition["agents"][group]["count"]
        if old == count:
            return {"version": self.version, "old": old, "changed": False}
        result = self.set_setting("agents", group, "count", count,
                                  reason=f"Set {group} count from {old} to {count}", by=by)
        return {"version": result["version"], "old": old, "changed": result["changed"]}

    def set_setting(self, section: str | None, name: str | None, key: str, value, *,
                    reason: str, by: str) -> dict:
        """Change one simple setting, recording a new definition version."""
        with self.lock():
            self.require()
            body = self.definition_body()
            new_body = set_setting_text(body, section, name, key, value)
            if new_body == body:
                return {"changed": False, "version": self.version}
            parse_definition(new_body)
            return {"changed": True, "version": self._write_version(new_body, reason, by)}

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


# ---------------------------------------------------------------------------
# Simple settings (/workflow set)
# ---------------------------------------------------------------------------

SETTINGS = {"workflow": ("description", "model", "folder"), "group": ("count", "model", "every", "folder"),
            "queue": ("notify",)}
SETTING_PLACE = {"count": "an agent group", "every": "an agent group", "notify": "a queue",
                 "description": "the workflow",
                 "model": "the workflow or an agent group", "folder": "the workflow or an agent group"}


def effective_model(definition: dict, group: dict | None, default_model: str = "") -> tuple[str, str]:
    """The model a group's new agents get, and where that value comes from."""
    if group is not None and group.get("model"):
        return group["model"], "set on this group"
    if definition.get("model"):
        return definition["model"], "from the workflow"
    return default_model or "the Agent Zoo default", "Agent Zoo default"


def target_label(store: "WorkflowStore", kind: str, target: str | None) -> str:
    return store.name if kind == "workflow" else f"{store.name}:{target}"


def settings_example(store: "WorkflowStore", definition: dict, setting: str) -> str:
    if setting == "notify":
        queue = next(iter(definition["queues"]), "q_results")
        return f"/workflow set {store.name}:{queue} notify on"
    if setting == "description":
        return f'/workflow set {store.name} description "What this workflow is for"'
    group = next(iter(definition["agents"]), "grp_name")
    value = {"count": "4", "every": "30m", "model": "PROFILE", "folder": "PATH"}[setting]
    return f"/workflow set {store.name}:{group} {setting} {value}"


def prepare_setting(store: "WorkflowStore", kind: str, target: str | None, setting: str, raw: str,
                    *, check_model=None, default_model: str = "") -> dict:
    """Validate a /workflow set request and describe the change. Raises WorkflowError."""
    definition = store.definition()
    label = target_label(store, kind, target)
    setting = setting.strip().lower()
    known = sorted({s for names in SETTINGS.values() for s in names})
    if setting not in known:
        raise WorkflowError(f'unknown setting "{setting}"; you can set: count, model, every and folder on '
                            f"agent groups, description, model and folder on the workflow, and notify on queues. "
                            f"See what {label} has with /workflow set {label}")
    if setting not in SETTINGS[kind]:
        raise WorkflowError(f"{setting} is set on {SETTING_PLACE[setting]}; for example: "
                            + settings_example(store, definition, setting))
    group = queue = None
    if kind == "group":
        check_group_id(target, definition["agents"])
        group = definition["agents"][target]
    elif kind == "queue":
        if target == TRASH:
            raise WorkflowError(f"{TRASH} has no settings")
        check_queue_id(target, definition["queues"])
        queue = definition["queues"][target]
    raw = str(raw or "").strip()
    if not raw:
        raise WorkflowError(f"give a value, for example: {settings_example(store, definition, setting)}")
    word = raw.lower()
    section, name = {"workflow": (None, None), "group": ("agents", target), "queue": ("queues", target)}[kind]
    key = "notify_on_new" if setting == "notify" else setting

    if setting == "count":
        if not re.fullmatch(r"\d+", raw) or not 0 <= int(raw) <= 64:
            raise WorkflowError("count must be a whole number from 0 to 64")
        value, old_value = int(raw), group["count"]
        old_text, new_text, unchanged = str(old_value), str(value), value == old_value
    elif setting == "every":
        if group["takes_from"]:
            raise WorkflowError(f"every only applies to groups without takes_from; {target} starts one "
                                f"agent per item from {group['takes_from']}")
        if word == "off":
            value = None
        else:
            try:
                parse_duration(raw)
            except ValueError:
                raise WorkflowError(f'"{raw}" is not a duration like 45s, 30m, 2h or 1d, or off') from None
            value = raw
        old_text = group["every_text"] or "off"
        new_text = value or "off"
        unchanged = (group["every"] or None) == (parse_duration(value) if value else None)
    elif setting == "notify":
        if word not in {"on", "off", "true", "false", "yes", "no"}:
            raise WorkflowError("notify must be on or off")
        value = word in {"on", "true", "yes"}
        old_text, new_text = ("on" if queue["notify_on_new"] else "off"), ("on" if value else "off")
        unchanged = value == queue["notify_on_new"]
    else:  # description, model or folder
        current = (group or definition).get(setting)
        unset = "none" if setting == "description" else "default"
        if word == unset:
            value = None
        elif setting == "model":
            value = check_model(raw) if check_model else raw
        else:
            value = raw
        old_text = current or unset
        new_text = value or unset
        unchanged = (current or None) == value
    return {"kind": kind, "target": target, "label": label, "setting": setting, "section": section,
            "name": name, "key": key, "value": value, "old_text": old_text, "new_text": new_text,
            "unchanged": unchanged, "reason": f"User set {label} {setting} from {old_text} to {new_text}"}


def render_settings(store: "WorkflowStore", kind: str, target: str | None, default_model: str = "") -> str:
    """Markdown for the /workflow set NAME[:TARGET] popup."""
    definition = store.definition()
    control = store.control()
    label = target_label(store, kind, target)
    if kind == "queue" and target == TRASH:
        return f"{TRASH} has no settings.\n"
    lines = [f"Change one with `/workflow set {label} SETTING VALUE`.", ""]
    rows = []
    if kind == "workflow":
        model, source = effective_model(definition, None, default_model)
        folder = definition["folder"] or control.get("default_folder") or "the session's folder"
        rows.append(["`description`", definition["description"] or "none",
                     "shown in `/workflow list`; `none` removes it"])
        rows.append(["`model`", f"`{model}`",
                     ("set on the workflow" if definition["model"] else source) + "; `default` for the Agent Zoo default"])
        rows.append(["`folder`", f"`{folder}`",
                     ("set on the workflow" if definition["folder"] else "where it was created") + "; `default` to go back"])
        lines += md_table(["Setting", "Value", "Notes"], rows)
        if definition["agents"]:
            lines += ["", "Agent groups have their own count, model, schedule and folder: "
                      + ", ".join(f"`/workflow set {store.name}:{g}`" for g in definition["agents"])]
        if definition["queues"]:
            queue = next(iter(definition["queues"]))
            lines += ["", f"Queues can tell the controlling agent about new items, e.g. "
                          f"`/workflow set {store.name}:{queue} notify on`."]
    elif kind == "group":
        check_group_id(target, definition["agents"])
        group = definition["agents"][target]
        model, source = effective_model(definition, group, default_model)
        folder = group["folder"] or definition["folder"] or control.get("default_folder") or "the session's folder"
        rows.append(["`count`", str(group["count"]), "agents at once, 0 to 64"])
        rows.append(["`model`", f"`{model}`", f"{source}; `default` for the workflow's model"])
        if group["takes_from"]:
            rows.append(["`every`", "—", f"not available: this group takes items from `{group['takes_from']}`"])
        else:
            rows.append(["`every`", group["every_text"] or "off", "start a new agent at most this often, or `off`"])
        rows.append(["`folder`", f"`{folder}`",
                     ("set on this group" if group["folder"] else "from the workflow") + "; `default` for the workflow's"])
        lines += md_table(["Setting", "Value", "Notes"], rows)
        stale = [f"`{a['id']}` ({a.get('model')})" for a in store.agents(target)
                 if a["state"] in ACTIVE_STATES and a.get("model") and a.get("model") != model]
        if stale:
            lines += ["", "Still on a different model (started before a change): " + ", ".join(stale)]
    else:
        check_queue_id(target, definition["queues"])
        notify = definition["queues"][target]["notify_on_new"]
        lines += md_table(["Setting", "Value", "Notes"],
                          [["`notify`", "on" if notify else "off", "tell the controlling agent about new items"]])
    return "\n".join(lines) + "\n"

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
    stores = [WorkflowStore(name, root) for name in names]
    archived = [s.name for s in stores if s.control().get("state") == "archived"]
    current = [s for s in stores if s.name not in archived]
    example = current[0].name if current else (names[0] if names else "miner")
    lines = [f"Workflows. Open one with wf:<name>, e.g. wf:{example}."]
    if not names:
        lines.append("There are no workflows yet. Create one with create_workflow.")
    for store in current:
        control = store.control()
        active = sum(a["state"] in ACTIVE_STATES for a in store.agents())
        waiting = sum(i["state"] == "waiting" for i in store.items() if i["queue"] != TRASH)
        attention = sum(a["state"] == "attention" for a in store.agents())
        line = (f"{store.name}: {state_label(control)}, controlled by "
                f"{controller_label(control, viewer_session_id)}, def_{store.version}; "
                f"{active} agents active, {waiting} items waiting")
        if attention:
            line += f", {attention} agents need attention"
        description = store.definition().get("description")
        lines.append(line + (f". {description}" if description else ""))
    if archived:
        lines.append(f"Archived (hidden from the user's /workflow list; command_workflow unarchive restores one): "
                     + ", ".join(archived))
    return "\n".join(lines) + "\n"


# -- the /workflow list popup (markdown) -----------------------------------------

def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def relative_time(at: str, now: datetime | None = None) -> str:
    try:
        then = datetime.fromisoformat(at)
    except (TypeError, ValueError):
        return "unknown"
    minutes = max(0, int(((now or datetime.now(timezone.utc)) - then).total_seconds())) // 60
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes}m ago"
    if minutes < 24 * 60:
        return f"{minutes // 60}h ago"
    if minutes < 30 * 24 * 60:
        return f"{minutes // (24 * 60)}d ago"
    return then.astimezone().strftime("%Y-%m-%d")


NBSP = "\u00a0"  # The TUI wraps table cells only at ordinary spaces.


def unbroken(text: str) -> str:
    return str(text).replace(" ", NBSP)


def workflow_row(store: WorkflowStore) -> dict:
    """One workflow in the /workflow list. Never raises: a damaged workflow still gets a row."""
    try:
        control = store.control()
        agents = store.agents()
        items = [i for i in store.items() if i["queue"] != TRASH]
        state = control.get("state", "paused")
        attention = state_count(agents, "attention")
        description = store.definition().get("description") or "—"
        last = store.last_activity()
        counts = [state_count(agents, "starting", "running"), state_count(items, "waiting"),
                  state_count(items, "done")]
    except Exception as exc:  # Show the problem in the list instead of hiding the workflow.
        state, attention, description, last, counts = "paused", 0, f"Could not read it: {exc}", "", ["?"] * 3
    return {"name": store.name, "state": state, "last": last, "attention": attention,
            "cells": [f"`{store.name}`", description, unbroken("⚠ attention" if attention else state), *counts,
                      unbroken(relative_time(last) if last else "never")]}


def workflow_table(rows: list[dict], *, show_status: bool = True) -> list[str]:
    """Running first, then most recently active. Only Description is free text; it absorbs the width."""
    ordered = sorted(rows, key=lambda r: r["last"], reverse=True)
    ordered.sort(key=lambda r: (r["state"] != "running", not r["attention"]))
    headers = ["Workflow", "Description", "Status", "Agents", "Waiting", "Done", "Updated"]
    cells = [list(r["cells"]) for r in ordered]
    # Agent counts only matter while something runs; a column of zeros is noise.
    hidden = {2} if not show_status else set()
    if all(c[3] == 0 for c in cells):
        hidden.add(3)
    return md_table([h for i, h in enumerate(headers) if i not in hidden],
                    [[v for i, v in enumerate(c) if i not in hidden] for c in cells])


def render_list(root: Path | None = None, viewer_session_id: str = "", *,
                archived: bool = False) -> tuple[str, str]:
    """Subtitle and markdown body for /workflow list (or /workflow list archived)."""
    rows = [workflow_row(WorkflowStore(name, root)) for name in list_workflow_names(root)]
    shelved = [r for r in rows if r["state"] == "archived"]
    current = [r for r in rows if r["state"] != "archived"]
    running = sum(r["state"] == "running" for r in current)
    tally = {"running": running, "paused": len(current) - running, "archived": len(shelved)}
    subtitle = " · ".join(f"{n} {state}" for state, n in tally.items() if n)

    if archived:
        if not shelved:
            return subtitle, "No archived workflows. Hide a one-off workflow with `/workflow archive NAME`.\n"
        lines = [*workflow_table(shelved, show_status=False), "",
                 "Bring one back with `/workflow unarchive NAME`."]
        return subtitle, "\n".join(lines) + "\n"

    if not rows:
        return subtitle, ("No workflows yet. Ask the agent to set one up, for example *“make a workflow that fixes "
                          "each failing test with its own agent”*. Nothing runs until you start it with "
                          "`/workflow run NAME`.\n")
    lines = workflow_table(current) if current else ["No active workflows."]
    lines.append("")
    if any(r["attention"] for r in current):
        lines.append("⚠ Some agents need attention; `/workflow view NAME` shows what happened.")
    if any(r["cells"][1] == "—" for r in current):
        lines.append('Add a description with `/workflow set NAME description "..."`.')
    lines.append("Open one with `/workflow view NAME`. Other commands: run, pause, set, rename, archive.")
    if shelved:
        lines.append(f"{plural(len(shelved), 'archived workflow')} hidden: `/workflow list archived`.")
    return subtitle, "\n".join(lines) + "\n"


# -- the /workflow view popups (markdown) ----------------------------------------

UUID_LIKE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
RECENT_EVENTS = 8
FINISHED_SHOWN = 30
DONE_SHOWN = 50


def shorten_ids(text) -> str:
    """Session IDs are long; their first 8 characters identify them in a popup."""
    return UUID_LIKE.sub(lambda m: m[0][:8], str(text))


def md_cell(value) -> str:
    return re.sub(r"[ \t\r\n]+", " ", str(value)).strip().replace("|", "\\|")


def md_table(headers: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
            *("| " + " | ".join(md_cell(c) for c in row) + " |" for row in rows)]


def md_quote(text: str) -> list[str]:
    """Quote free text so it wraps (code blocks truncate long lines), keeping each line its own paragraph."""
    lines = []
    for line in text.strip("\n").splitlines():
        if lines:
            lines.append(">")
        if line.strip():
            lines.append("> " + line)
    return lines


def clip(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def state_count(records: list[dict], *states: str) -> int:
    return sum(r["state"] in states for r in records)


def event_text(record: dict) -> str:
    details = [f"{key.replace('_', ' ')} {clip(', '.join(map(str, value)) if isinstance(value, list) else value, 100)}"
               for key, value in record.items() if key not in {"at", "event"} and value not in (None, "", [])]
    return shorten_ids(" · ".join([f"**{record['event'].replace('_', ' ')}**", *details]))


def agents_summary(count: int, mine: list[dict], group_paused: bool) -> str:
    parts = [f"{state_count(mine, 'starting', 'running')} of {count} running"]
    if paused := state_count(mine, "paused"):
        parts.append(f"{paused} paused mid-task")
    if attention := state_count(mine, "attention"):
        parts.append(f"⚠ {attention} {'needs' if attention == 1 else 'need'} attention")
    parts.append(f"{state_count(mine, 'finished')} finished")
    if group_paused:
        parts.append("group paused")
    return " · ".join(parts)


def view_footer(store: WorkflowStore, definition: dict, state: str) -> list[str]:
    name = store.name
    group = next(iter(definition["agents"]), "grp_GROUP")
    queue = next(iter(definition["queues"]), "q_QUEUE")
    lines = ["**Commands**", "",
             f"- `/workflow view {name}:{group}` · `/workflow view {name}:{queue}` — one group or queue"]
    if state == "archived":
        lines.append(f"- `/workflow unarchive {name}` — bring it back, paused")
    else:
        lines.append(f"- `/workflow {'pause' if state == 'running' else 'run'} {name}`")
        lines.append(f"- `/workflow set {name}` · `/workflow set {name}:{group}` — model, folder, count and schedule")
    lines.append(f"- `/workflow rename {name} NEW_NAME`")
    lines.append("- `/workflow list` — all workflows")
    return lines


def render_overview_md(store: WorkflowStore, viewer_session_id: str = "",
                       default_model: str = "") -> tuple[str, str]:
    """Subtitle and markdown body for /workflow view NAME."""
    control = store.control()
    definition = store.definition()
    agents = store.agents()
    counts = queue_counts(store.items())
    state = control.get("state", "paused")
    subtitle = [state_label(control), f"def_{store.version}"]
    if (control.get("controller") or {}).get("session_id"):
        subtitle.append(f"controlled by {controller_label(control, viewer_session_id)}")
    model, _ = effective_model(definition, None, default_model)
    folder = definition["folder"] or control.get("default_folder") or "the session's folder"
    lines = [f"**Model** `{model}` · **Folder** `{folder}`", "", "## Agent groups", ""]

    groups = definition["agents"]
    models = {g: effective_model(definition, s, default_model)[0] for g, s in groups.items()}
    show_models = any(m != model for m in models.values())
    paused_groups = set(control.get("paused_groups") or []) if state == "running" else set()
    rows = []
    for group, settings in groups.items():
        # Plain text: the TUI pads inline code with spaces, which reads badly before a comma.
        source = settings["takes_from"] or "standing"
        if settings["every"]:
            source += f", every {settings['every_text']}"
        row = [f"`{group}`", source, ", ".join(settings["sends_to"]) or "—",
               agents_summary(settings["count"], [a for a in agents if a["group"] == group], group in paused_groups)]
        rows.append(row + ([f"`{models[group]}`"] if show_models else []))
    headers = ["Group", "Takes from", "Sends to", "Agents"] + (["Model"] if show_models else [])
    lines += md_table(headers, rows) if rows else ["No agent groups."]

    lines += ["", "## Queues", ""]
    queue_names = list(definition["queues"]) + [q for q in counts if q not in definition["queues"]]
    rows = []
    for queue in queue_names:
        bucket = counts.get(queue, {"waiting": 0, "taken": 0, "done": 0})
        if queue in definition["queues"]:
            note = "notifies the controlling agent" if definition["queues"][queue]["notify_on_new"] else ""
        else:
            note = "discarded items" if queue == TRASH else "not in the current definition"
        rows.append([f"`{queue}`", bucket["waiting"], bucket["taken"], bucket["done"], note])
    lines += md_table(["Queue", "Waiting", "In progress", "Done", "Notes"], rows) if rows else ["No queues."]

    attention = [a for a in agents if a["state"] == "attention"]
    if attention:
        lines += ["", "## ⚠ Needs attention", ""]
        lines += [f"- `{a['id']}` — {clip(a.get('problem') or 'unknown problem', 300)}" for a in attention]
    recent = store.log_records()[-RECENT_EVENTS:]
    if recent:
        lines += ["", "## Recent activity", ""]
        lines += md_table(["When", "Event"], [[relative_time(r["at"]), event_text(r)] for r in reversed(recent)])
    lines += ["", *view_footer(store, definition, state)]
    return " · ".join(subtitle), "\n".join(lines) + "\n"


def render_group_md(store: WorkflowStore, group: str, default_model: str = "") -> tuple[str, str]:
    """Subtitle and markdown body for /workflow view NAME:GROUP."""
    definition = store.definition()
    control = store.control()
    agents = store.agents(group)
    settings = definition["agents"].get(group)
    if settings is None and not agents:
        check_group_id(group, definition["agents"])
    subtitle = describe_group(settings) if settings else "not in the current definition"
    if group in (control.get("paused_groups") or []) and control.get("state") == "running":
        subtitle += " · paused"
    lines = []
    if settings:
        model, source = effective_model(definition, settings, default_model)
        folder = (settings["folder"] or definition["folder"] or control.get("default_folder")
                  or "the session's folder")
        lines += [f"**Model** `{model}` ({source}) · **Folder** `{folder}`", ""]

    lines += ["## Agents", ""]
    current = [a for state in ("starting", "running", "paused", "attention", "stopped")
               for a in agents if a["state"] == state]
    ended = [a for a in agents if a["state"] in {"finished", "restarted"}]
    shown = current + list(reversed(ended[-FINISHED_SHOWN:]))
    rows = [[f"`{a['id']}`", ("⚠ " if a["state"] == "attention" else "") + a["state"],
             f"`{a['item']}`" if a.get("item") else "—", f"def_{a['version']}",
             clip(a.get("summary") or a.get("problem") or "")] for a in shown]
    lines += md_table(["Agent", "State", "Item", "Version", "Summary"], rows) if rows else ["No agents yet."]
    if len(ended) > FINISHED_SHOWN:
        lines += ["", f"{len(ended) - FINISHED_SHOWN} older finished agents are not shown."]
    if settings:
        lines += ["", "## Instructions", "", *md_quote(settings["instructions"])]

    lines += ["", "**Commands**", ""]
    if settings:
        lines.append(f"- `/workflow set {store.name}:{group}` — count, model, schedule and folder")
        if settings["takes_from"]:
            lines.append(f"- `/workflow view {store.name}:{settings['takes_from']}` — the items it takes")
    lines.append(f"- `/workflow view {store.name}` — the whole workflow")
    return subtitle, "\n".join(lines) + "\n"


def render_queue_md(store: WorkflowStore, queue: str) -> tuple[str, str]:
    """Subtitle and markdown body for /workflow view NAME:QUEUE."""
    definition = store.definition()
    items = store.items(queue)
    known = queue in definition["queues"] or queue == TRASH
    if not known and not items:
        check_queue_id(queue, definition["queues"])
    counts = queue_counts(items).get(queue, {"waiting": 0, "taken": 0, "done": 0})
    subtitle = f"{counts['waiting']} waiting · {counts['taken']} in progress · {counts['done']} done"
    takers = [f"`{g}`" for g, s in definition["agents"].items() if s["takes_from"] == queue]
    senders = [f"`{g}`" for g, s in definition["agents"].items() if queue in s["sends_to"]]
    if queue == TRASH:
        lines = ["Discarded items. Move one back to a queue to restore it."]
    else:
        lines = [f"**Filled by** {', '.join(senders) or 'nobody; items come from you or the agent'} · "
                 f"**Taken by** {', '.join(takers) or 'no group, so items stay here'}"]
    if not known:
        lines += ["", "⚠ This queue is not in the current definition; its items wait until it is added back."]

    def added_by(item):
        return shorten_ids(item["created_by"]) + (f" (from `{item['from_item']}`)" if item.get("from_item") else "")

    def notes(item):
        return plural(len(item["notes"]), "note") if item.get("notes") else ""

    waiting = [i for i in items if i["state"] == "waiting"]
    taken = [i for i in items if i["state"] == "taken"]
    done = [i for i in items if i["state"] == "done"]
    if taken:
        lines += ["", "## In progress", ""]
        lines += md_table(["Item", "Title", "Agent"],
                          [[f"`{i['id']}`", i["title"], f"`{i['taken_by']}`"] for i in taken])
    if waiting:
        lines += ["", "## Waiting (oldest first)", ""]
        lines += md_table(["Item", "Title", "Added by", "Notes"],
                          [[f"`{i['id']}`", i["title"], added_by(i), notes(i)] for i in waiting])
    if done:
        heading = "## Done (newest first" + (f", last {DONE_SHOWN} of {len(done)})" if len(done) > DONE_SHOWN else ")")
        lines += ["", heading, ""]
        lines += md_table(["Item", "Title", "Added by", "Notes"],
                          [[f"`{i['id']}`", i["title"], added_by(i), notes(i)] for i in reversed(done[-DONE_SHOWN:])])
    if not items:
        lines += ["", "No items yet."]
    lines += ["", "**Commands**", "", f"- `/workflow view {store.name}` — the whole workflow"]
    if takers:
        lines.append(f"- `/workflow view {store.name}:{takers[0].strip('`')}` — the group that takes these items")
    return subtitle, "\n".join(lines) + "\n"

def render_overview(store: WorkflowStore, viewer_session_id: str = "", default_model: str = "") -> str:
    control = store.control()
    definition = store.definition()
    agents = store.agents()
    items = store.items()
    counts = queue_counts(items)
    name = store.name
    model, _ = effective_model(definition, None, default_model)
    folder = definition["folder"] or control.get("default_folder") or "the session's folder"
    lines = [f"Workflow {name}: {state_label(control)}, controlled by "
             f"{controller_label(control, viewer_session_id)}, definition def_{store.version}.",
             f"Model: {model}. Folder: {folder}. Change these with /workflow set {name} model|folder VALUE.", "",
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
        group_model, _ = effective_model(definition, settings, default_model)
        lines.append(f"  {group}: {status}. {describe_group(settings)}, model {group_model}.")
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


def render_group(store: WorkflowStore, group: str, default_model: str = "") -> str:
    definition = store.definition()
    agents = store.agents(group)
    settings = definition["agents"].get(group)
    if settings is None and not agents:
        return f"{group} is not in {store.name}'s definition and has no agents.\n"
    lines = [f"{group} in {store.name}: " + (describe_group(settings) + "." if settings
                                             else "not in the current definition.")]
    if settings:
        model, source = effective_model(definition, settings, default_model)
        lines.append(f"New agents use model {model} ({source}). Settings: /workflow set {store.name}:{group}")
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
    if agent.get("model"):
        lines.append(f"Model: {agent['model']}")
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
                root: Path | None = None, default_model: str = "") -> str | None:
    """Text for a wf: buffer, or None when the ID is not a workflow view."""
    parts = buffer_id.split(":")
    if not parts or parts[0] != "wf" or len(parts) not in (2, 3):
        return None
    if parts[1] == "index" and len(parts) == 2:
        return render_index(root, viewer_session_id)
    store = WorkflowStore(parts[1], root).require()
    if len(parts) == 2:
        return render_overview(store, viewer_session_id, default_model)
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
        return render_group(store, view, default_model)
    if AGENT_ID.fullmatch(view):
        return render_agent(store, view, transcript)
    if ITEM_ID.fullmatch(view):
        return render_item(store, view)
    raise WorkflowError(f'unknown view "{view}"; use definition, history, log, def_N, q_*, grp_*, agt_* or item_N')


# ===========================================================================
# Agent Zoo integration
# ===========================================================================

import asyncio
import collections
import logging
import shlex
import threading
import uuid
from copy import deepcopy
from dataclasses import replace

from agent_utils import Feature, Tool, MODEL_RENDER_CHANNEL
from agent_utils.components import (
    InterruptCheck, InterruptDelivery, ToolDispatchStart, RenderTransformRegistrar, MessageRenderer,
)
from agent_utils.files.buffer_manager import ReadonlyBufferView
from agent_utils.session_cwd import get_session_cwd
from agent_utils.session_repository import SessionRepository
from agent_utils.skills import SkillSource
from agents.llm import enabled_llm_models, require_model_enabled
from agents.system_prompt import SystemPromptSkillList
from agent_zoo.config_loader import CONFIG_PATH, load_layered_config
from agent_zoo.live_session_registry import list_live_sessions
from agent_zoo.projects import resolve_state_root
from agent_zoo.runtime.commands import Command, CommandError, CommandResult
from agent_zoo.runtime.models import InspectorDocument
from agent_zoo.runtime_launch import CheckpointRef, Fresh, InstanceRef, Resume, RuntimeLaunch, RuntimeSettings
from agent_zoo.session_store import resolve_local_root
from agent_zoo.tools.special_buffers import RegisterSpecialBuffers
from agent_zoo.wsctl import AzoWs
from tmux_pilot.process_identity import ProcessIdentity, capture_process_identity, observe_process

LOG = logging.getLogger(__name__)

WORKER_KIND = re.compile(r"wf:(?P<workflow>[a-z][a-z0-9_]{0,63}):(?P<agent>agt_[a-z0-9_]+_\d+)\Z")
TICK_SECONDS = 2.0
READY_TIMEOUT_SECONDS = 120
MAX_QUICK_RESUMES = 3
WORKER_TOOLS_GROUP = "Workflows"


def worker_identity(state):
    """Workers are identified by their durable session kind, never by environment."""
    return WORKER_KIND.fullmatch(str(getattr(state, "_session_kind", "") or ""))


def project_sessions_dir(project: str) -> Path:
    if not project or project in {".", ".."} or Path(project).name != project:
        raise WorkflowError(f"invalid project name {project!r}")
    return Path(resolve_state_root()) / "projects" / project / "sessions"


def saved_checkpoint(project: str, session_id: str, revision_id: str | None = None):
    repository = SessionRepository(project_sessions_dir(project) / session_id)
    saved = repository.load(session_id=session_id, commit_id=revision_id)
    if not saved.commit_id:
        raise WorkflowError("the saved checkpoint has no committed revision")
    return saved


def checkpoint_recovery(document: dict) -> tuple[str, str]:
    """Classify durable evidence, never infer that a missing result means no effect."""
    if not isinstance(document, dict):
        return "uncertain", "checkpoint document is malformed"
    core = document.get("core")
    entries = document.get("entries")
    if not isinstance(core, dict) or not isinstance(entries, list) or not entries:
        return "uncertain", "checkpoint is missing canonical core/entries"
    if core.get("done"):
        return "done", "checkpoint is already done; reconcile the worker record before restarting"
    latest_role = None
    latest_calls = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("messages"), list):
            return "uncertain", "checkpoint contains a malformed entry"
        calls = entry.get("tool_calls") or []
        if not isinstance(calls, list):
            return "uncertain", "checkpoint contains malformed tool calls"
        messages = entry["messages"]
        if any(not isinstance(m, dict) for m in messages):
            return "uncertain", "checkpoint contains a malformed message"
        results = {m["tool_call_id"] for m in messages
                   if m.get("role") == "tool" and isinstance(m.get("tool_call_id"), str)}
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
                return "uncertain", "checkpoint contains a tool call without an identity"
            if call.get("result") is None or call["id"] not in results:
                return "uncertain", f"unresolved tool call {call['id']} ({call.get('name', 'unknown tool')})"
        # Canonical entries carry parsed calls as well as provider messages.
        if any(m.get("tool_calls") for m in messages) and not calls:
            return "uncertain", "checkpoint has tool-call messages without canonical call records"
        for message in messages:
            if message.get("role") in {"user", "assistant"}:
                latest_role, latest_calls = message["role"], calls
    if latest_role == "user" or latest_role == "assistant" and latest_calls:
        return "continue", "checkpoint has a pending reply with complete tool results"
    if latest_role == "assistant":
        return "message", "worker ended its turn without finishing its task"
    return "uncertain", "checkpoint has no resumable user or assistant turn"


def saved_model_config(config) -> dict | None:
    """Saved config layers only: never the session's model change or CLI --set overrides."""
    path = ((config or {}).get("_runtime_launch", {}) or {}).get("config_path")
    try:
        if path and Path(path).resolve() == CONFIG_PATH.resolve():
            path = None
        return load_layered_config(state_root=resolve_state_root(), config_path=path or None)
    except Exception:
        LOG.warning("Cannot read saved Agent Zoo model settings", exc_info=True)
        return None


def saved_default_model(config) -> str:
    for source in (saved_model_config(config), config):
        model = str(((source or {}).get("llm", {}) or {}).get("default") or "")
        if model:
            return model
    return ""


def check_model(config, profile: str) -> str:
    """The profile must be configured in llm.models and enabled."""
    profile = str(profile or "").strip()
    models = ((config or {}).get("llm", {}) or {}).get("models", {}) or {}
    if not models:
        return profile  # Nothing to check against.
    try:
        enabled = list(enabled_llm_models(config))
    except Exception:
        enabled = sorted(models)
    choices = ", ".join(enabled) or "none"
    if profile not in models:
        raise WorkflowError(f'unknown model profile "{profile}"; enabled profiles: {choices}')
    try:
        require_model_enabled(config, profile)
    except ValueError:
        raise WorkflowError(f'model profile "{profile}" is disabled; enabled profiles: {choices}') from None
    return profile


def check_definition_models(definition: dict, config) -> None:
    problems = []
    for owner, model in [("definition", definition.get("model"))] + [
            (group, settings.get("model")) for group, settings in definition["agents"].items()]:
        if model:
            try:
                check_model(config, model)
            except WorkflowError as exc:
                problems.append(f"{owner}: {exc}")
    if problems:
        raise WorkflowError("The definition has problems:\n" + "\n".join(f"- {p}" for p in problems))


def process_evidence(identity: dict | None) -> str:
    """'alive', 'dead' or 'unknown' for a recorded process identity."""
    if not identity:
        return "unknown"
    try:
        return observe_process(ProcessIdentity.from_dict(identity)).state.value.lower()
    except Exception:
        return "unknown"


def run_async(factory, timeout: float = 30):
    """Run a coroutine factory to completion from synchronous code on any thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(asyncio.wait_for(factory(), timeout))
    box = {}

    def target():
        try:
            box["value"] = asyncio.run(asyncio.wait_for(factory(), timeout))
        except BaseException as exc:  # Re-raised on the calling thread.
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout + 5)
    if "error" in box:
        raise box["error"]
    return box.get("value")


async def send_to_agent(target: dict, action: str, message: str = "") -> None:
    """Send one frame to a live agent after proving its websocket identity."""
    client = AzoWs.attach(target["websocket_url"], modern=False, terminate_on_close=False,
                          connect_timeout_s=10)
    try:
        await client.connect()
        hello = client.observation().hello or {}
        if (hello.get("session_id") != target["session_id"]
                or hello.get("instance_id") != target["instance_id"] or client.protocol_errors):
            raise WorkflowError("the agent's websocket belongs to a different session")
        if action == "message":
            await client.send_user(message, client_id=uuid.uuid4().hex)
        elif action == "shutdown":
            await client.control("shutdown", {"save": True, "reason": "workflow"})
        else:
            await client.slash("/" + action)
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Transcript rendering (from azo-plugin-swarm)
# ---------------------------------------------------------------------------

def saved_transcript(project: str, session_id: str) -> str:
    """Readable projection of all messages in an agent's shared checkpoint."""
    header = f"Session {session_id}\nSaved transcript, not a live stream. Newest turn first.\n"
    try:
        saved = saved_checkpoint(project, session_id)
    except FileNotFoundError:
        return header + "No saved checkpoint is available yet.\n"
    lines = [header, f"Source Commit: {saved.commit_id}", f"Source Instance: {saved.instance_id}"]
    entries = saved.document.get("entries")
    if not isinstance(entries, list):
        nested = saved.document.get("state")
        entries = nested.get("entries", []) if isinstance(nested, dict) else []
    turns, current = [], []
    title = "Context before first user turn"
    for position, entry in enumerate(entries or []):
        if not isinstance(entry, dict):
            continue
        index = entry.get("index", position)
        if type(index) is not int:
            index = position
        messages = [m for m in (entry.get("messages") or []) if isinstance(m, dict)]
        calls_seen, results_seen = set(), set()
        for ordinal, message in enumerate(messages, 1):
            role = str(message.get("role") or message.get("type") or "message")
            generated = bool(entry.get("system_generated") or message.get("system_generated"))
            if role == "user" and not generated:
                if current:
                    turns.append((title, current))
                current = []
                title = f"Turn [entry {index}]"
            if role in {"tool", "function", "tool_result", "function_call_output"}:
                label = "TOOL RESULT"
            elif role in {"tool_use", "function_call"}:
                label = "TOOL CALL"
            else:
                label = role.upper()
            if generated:
                label += " (system-generated)"
            pointer = f"entry {index}; message {ordinal}"
            call_id = message.get("tool_call_id") or message.get("call_id")
            if call_id:
                pointer += f"; call {call_id}"
                if role in {"tool", "function"}:
                    results_seen.add(str(call_id))
            current.extend(["", f"### {label} [{pointer}]"])
            body = _content_text(message if message.get("type") in {
                "tool_use", "function_call", "tool_result", "function_call_output"
            } else message.get("content"))
            if body:
                current.append(body)
            message_calls = [*(message.get("tool_calls") or []), *(message.get("provider_tool_calls") or [])]
            if isinstance(message.get("function_call"), dict):
                message_calls.append(message["function_call"])
            for call in message_calls:
                if isinstance(call, dict):
                    key = str(call.get("id") or call.get("call_id") or "")
                    if key and key in calls_seen:
                        continue
                    current.extend(_call_lines(call))
                    if key:
                        calls_seen.add(key)
            content = message.get("content")
            for block in [message, *(content if isinstance(content, list) else [])]:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"tool_use", "function_call"}:
                    calls_seen.add(str(block.get("id") or block.get("call_id") or ""))
                elif block.get("type") in {"tool_result", "function_call_output"}:
                    results_seen.add(str(block.get("tool_use_id") or block.get("call_id") or ""))
        for call in entry.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            key = str(call.get("id") or call.get("call_id") or "")
            if not key or key not in calls_seen:
                current.extend(["", f"### TOOL CALL [entry {index}]"])
                current.extend(_call_lines(call))
            if call.get("result") is not None and (not key or key not in results_seen):
                status = " (error)" if call.get("error") else ""
                current.extend(["", f"### TOOL RESULT{status} [entry {index}; call {key}]",
                                _content_text(call["result"])])
    if current:
        turns.append((title, current))
    for title, messages in reversed(turns):
        lines.extend(["", f"## {title}", *messages])
    if not turns:
        lines.append("No saved messages.")
    return "\n".join(lines).rstrip() + "\n"


def _argument_text(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _call_lines(call):
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    label = str(function.get("name") or "(unnamed)")
    call_id = call.get("id") or call.get("call_id")
    lines = [f"Tool call: {label}" + (f" [call {call_id}]" if call_id else "")]
    arguments = next((function[key] for key in ("arguments", "raw_args", "input", "parsed_args")
                      if key in function), None)
    if arguments is not None:
        lines.extend(["Arguments:", _argument_text(arguments)])
    return lines


def _content_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(text for item in value if (text := _content_text(item)))
    if not isinstance(value, dict):
        return str(value)
    kind = value.get("type")
    if kind in {"redacted_thinking", "encrypted_reasoning"}:
        return ""
    if kind in {"tool_use", "function_call"}:
        return "\n".join(_call_lines(value))
    if kind in {"tool_result", "function_call_output"}:
        call_id = value.get("tool_use_id") or value.get("call_id") or ""
        status = " (error)" if value.get("is_error") else ""
        return f"Tool result{status} [call {call_id}]:\n{_content_text(value.get('content', value.get('output')))}"
    if kind in {"image", "image_url", "input_image", "audio", "input_audio", "video",
                "file", "input_file", "document"}:
        return f"[{kind}: non-text payload omitted]"
    text_fields = ("text", "content", "thinking", "refusal", "summary")
    if kind in {"text", "input_text", "output_text", "summary_text", "thinking", "reasoning", "refusal"} or (
            not kind and value and set(value).issubset(text_fields)):
        return "\n".join(filter(None, (_content_text(value[key]) for key in text_fields if key in value)))
    return json.dumps(value, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

def render_instructions(store: WorkflowStore, group: dict, agent: dict, item: dict | None,
                        viewer_session_id: str = "") -> str:
    text = group["instructions"]
    item_text = item_block(store, item) if item else "(this agent did not take an item)"
    if item and "{{item}}" not in text:
        text = text.rstrip("\n") + "\n\n{{item}}\n"
    text = text.replace("{{item}}", item_text)
    text = text.replace("{{agent}}", agent["id"]).replace("{{workflow_name}}", store.name)

    def expand(match):
        try:
            return render_view(match[1], viewer_session_id=viewer_session_id, root=store.root) or match[0]
        except WorkflowError as exc:
            return f"(could not read {match[1]}: {exc})"

    return re.sub(r"\{\{buffer://(wf:[^}]+)\}\}", expand, text)


def render_folder(store: WorkflowStore, template: str, agent: dict) -> str:
    folder = template.replace("{{agent}}", agent["id"]).replace("{{workflow_name}}", store.name)
    return str(Path(folder).expanduser())


# ---------------------------------------------------------------------------
# Controller: control, scheduling, live agents
# ---------------------------------------------------------------------------

class WorkflowController:
    """Per-process controller. All scheduling for a workflow happens in the
    session that controls it; other sessions only change files."""

    def __init__(self, config, session=None):
        self.config, self.session = config, session
        self.lock = threading.RLock()
        # Keep each owned process until poll() reaps it, even after a workflow
        # is archived/transferred or the same logical agent has been relaunched.
        self.handles: dict[tuple[str, str, str], object] = {}
        self.managed: dict[str, dict] = {}
        self.me: dict | None = None
        self.launch: dict | None = None
        self.launcher = None
        self.notices = collections.deque()
        self.notified: set = set()
        self.errors: dict[str, str] = {}
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.adopted_instance = ""
        self.worker_record: dict | None = None
        self.worker_reminded_entry = -1

    # -- identity and launch settings (owner thread) -------------------------

    def bind(self, state) -> None:
        session_id = str(getattr(state, "_session_id", "") or "")
        instance_id = str(getattr(state, "_instance_id", "") or "")
        if session_id and instance_id:
            self.me = {"session_id": session_id, "instance_id": instance_id}
        context = dict(getattr(state, "_agent_zoo_context", {}) or {})
        provenance = (self.config or {}).get("_runtime_launch", {}) or {}
        self.launcher = getattr(state, "session_launcher", None) or self.launcher
        if context.get("project_name") and context.get("project_store_root"):
            local_root = getattr(state, "_persistence_local_root", "") or ""
            if not local_root:
                try:
                    local_root = str(resolve_local_root(self.config))
                except Exception:
                    local_root = ""
            self.launch = {
                "project": context["project_name"], "store_root": context["project_store_root"],
                "state_root": str(resolve_state_root()), "local_state_root": str(local_root),
                "config_path": str(provenance.get("config_path") or ""),
                "config_sets": tuple(provenance.get("config_sets") or ()),
                "env_name": str(((self.config or {}).get("pilot", {}) or {}).get("_env_name") or ""),
                "cwd": get_session_cwd(state),
            }

    def me_label(self) -> str:
        return self.me["session_id"] if self.me else "unknown session"

    def require_launch(self) -> None:
        if not self.me:
            raise WorkflowError("this session has no runtime identity yet; try again in a moment")
        if self.launch is None or self.launcher is None:
            raise WorkflowError("this session cannot launch agents (it has no project or runtime launcher)")

    def default_model(self) -> str:
        return saved_default_model(self.config)

    def check_model(self, profile: str) -> str:
        return check_model(self.config, profile)

    def model_for(self, definition: dict, group: dict) -> str:
        model = group.get("model") or definition.get("model") or self.default_model()
        if not model:
            raise WorkflowError("no model is configured; set one with /workflow set NAME model PROFILE")
        return self.check_model(model)

    def settings_for(self, folder: str, model: str) -> RuntimeSettings:
        launch = self.launch
        return RuntimeSettings(
            project=launch["project"], workdir=folder, state_root=launch["state_root"],
            store_root=launch["store_root"], local_state_root=launch["local_state_root"],
            config_path=launch["config_path"], config_sets=launch["config_sets"], model=model,
            env_name=launch["env_name"], pipeline_key="default", host="127.0.0.1", port=0,
        )

    def notice(self, text: str, key=None) -> None:
        if key is not None:
            if key in self.notified:
                return
            self.notified.add(key)
        self.notices.append(text)

    # -- control ----------------------------------------------------------------

    def take_control(self, store: WorkflowStore) -> dict | None:
        """Record this session as the controller and bump the epoch."""
        self.require_launch()
        with store.lock():
            store.require()
            control = store.control()
            if control.get("state") == "archived":
                raise WorkflowError(f"{store.name} is archived; restore it with /workflow unarchive {store.name} first")
            previous = control.get("controller")
            epoch = int(control.get("epoch", 0)) + 1
            store.update_control(controller=dict(self.me, project=self.launch["project"]), epoch=epoch)
            if not control.get("default_folder"):
                store.update_control(default_folder=self.launch["cwd"])
            store.log("control_taken", by=self.me["session_id"], epoch=epoch)
        self.managed[store.name] = {"epoch": epoch, "watermark": store.counters().get("item", 0)}
        self.ensure_thread()
        return previous

    def ensure_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.loop, name="workflows-scheduler", daemon=True)
        self.thread.start()

    def loop(self) -> None:
        while not self.stop_event.wait(TICK_SECONDS):
            for name in list(self.managed):
                try:
                    with self.lock:
                        self.tick(name)
                    self.errors.pop(name, None)
                except Exception as exc:
                    LOG.exception("Workflow %s scheduling failed", name)
                    message = f"Workflow {name}: scheduling failed: {exc}"
                    if self.errors.get(name) != message:
                        self.errors[name] = message
                        self.notice(message)
            self.reap_children()

    def reap_children(self) -> None:
        """Reap owned children independently of workflow scheduling ownership."""
        with self.lock:
            for key, handle in list(self.handles.items()):
                try:
                    if handle.poll().status == "exited":
                        self.handles.pop(key, None)
                except Exception:
                    # Keep the handle: unknown is not evidence of process exit.
                    LOG.exception("Could not poll workflow child %s", key)

    def handle_key(self, store: WorkflowStore, agent: dict) -> tuple[str, str, str]:
        return store.name, agent["id"], agent.get("instance_id", "")

    def run(self, name: str, group: str | None = None) -> str:
        with self.lock:
            store = WorkflowStore(name).require()
            definition = store.definition()
            if group is not None:
                check_group_id(group, definition["agents"])
            previous = self.take_control(store)
            with store.lock():
                control = store.control()
                if group is None:
                    store.update_control(state="running", paused_groups=[])
                else:
                    paused = set(control.get("paused_groups") or [])
                    if control.get("state") != "running":
                        paused = set(definition["agents"])
                    paused.discard(group)
                    store.update_control(state="running", paused_groups=sorted(paused))
                store.log("run", group=group, by=self.me["session_id"])
            resumed, problems = self.resume_paused(store, group)
            started = self.tick(name)
            lines = []
            if previous and previous.get("session_id") != self.me["session_id"]:
                lines.append(f"Took control of {name} from session {previous['session_id'][:8]}.")
            scope = f"{group} in {name}" if group else name
            lines.append(f"Running {scope} from this session (def_{store.version}).")
            if resumed:
                lines.append("Resumed " + ", ".join(resumed) + " where they left off.")
            if started:
                lines.extend(started)
            if not resumed and not started:
                lines.append("No agents started yet: " + self.idle_reason(store, group))
            lines.extend(problems)
            lines.append(f"See wf:{name} for progress.")
            return "\n".join(lines)

    def idle_reason(self, store: WorkflowStore, group: str | None) -> str:
        definition = store.definition()
        reasons = []
        for name, settings in definition["agents"].items():
            if group and name != group:
                continue
            if settings["count"] == 0:
                reasons.append(f"{name} has count 0")
            elif settings["takes_from"]:
                waiting = sum(i["state"] == "waiting" for i in store.items(settings["takes_from"]))
                if not waiting:
                    reasons.append(f"{settings['takes_from']} is empty, so {name} has nothing to take")
        return "; ".join(reasons) + "." if reasons else "all groups are at their count."

    def pause(self, name: str, group: str | None = None) -> str:
        with self.lock:
            store = WorkflowStore(name).require()
            definition = store.definition()
            if group is not None:
                check_group_id(group, definition["agents"])
            with store.lock():
                control = store.control()
                if control.get("state") == "archived":
                    raise WorkflowError(f"{name} is archived")
                if group is None:
                    store.update_control(state="paused")
                else:
                    paused = set(control.get("paused_groups") or []) | {group}
                    store.update_control(paused_groups=sorted(paused))
                store.log("pause", group=group, by=self.me_label())
            stopped, starting, problems = [], [], []
            for agent in store.agents(group):
                if agent["state"] == "starting":
                    starting.append(agent["id"])
                elif agent["state"] == "running":
                    problem = self.pause_one(store, agent)
                    (problems if problem else stopped).append(problem or agent["id"])
            scope = f"{group} in {name}" if group else name
            text = f"Paused {scope}. No new agents will start."
            if stopped:
                text += f" Interrupted {', '.join(stopped)}; their work is kept and resumes when the workflow runs again."
            if starting:
                text += f" {', '.join(starting)} will pause as soon as they finish starting."
            if starting and name not in self.managed:
                text += f" (That happens in the session that controls {name}.)"
            return "\n".join([text, *problems])

    def pause_one(self, store: WorkflowStore, agent: dict) -> str | None:
        try:
            self.send(store, agent, "interrupt")
        except Exception as exc:
            return f"Could not interrupt {agent['id']}: {exc}"
        store.update_agent(agent["id"], state="paused")
        return None

    def archive(self, name: str) -> str:
        with self.lock:
            store = WorkflowStore(name).require()
            if store.control().get("state") == "archived":
                raise WorkflowError(f"{name} is already archived")
            self.pause(name)
            ended = []
            for agent in store.agents():
                if agent["state"] not in ACTIVE_STATES | {"attention"}:
                    continue
                try:
                    if self.liveness(store, agent) == "alive":
                        self.send(store, agent, "shutdown")
                except Exception as exc:
                    LOG.warning("Could not shut down %s: %s", agent["id"], exc)
                with store.lock():
                    if agent.get("item"):
                        store.return_item(agent["item"], agent["id"], by="archive")
                store.update_agent(agent["id"], state="stopped", problem="workflow archived")
                ended.append(agent["id"])
            with store.lock():
                store.update_control(state="archived", controller=None)
                store.log("archived", by=self.me_label())
            self.managed.pop(name, None)
            text = (f"Archived {name}. It no longer appears in /workflow; its definition, items and history "
                    f"stay readable at wf:{name}, and /workflow unarchive {name} brings it back.")
            if ended:
                text += f" Shut down {', '.join(ended)}."
            return text

    def unarchive(self, name: str) -> str:
        with self.lock:
            store = WorkflowStore(name).require()
            with store.lock():
                if store.control().get("state") != "archived":
                    raise WorkflowError(f"{name} is not archived")
                store.update_control(state="paused", paused_groups=[])
                store.log("unarchived", by=self.me_label())
            return f"Restored {name}. It is paused; start it with /workflow run {name}."

    def rename(self, name: str, new_name: str) -> str:
        with self.lock:
            renamed = WorkflowStore(name).require().rename(new_name, by=self.me_label())
            if name in self.managed:
                self.managed[renamed.name] = self.managed.pop(name)
            self.errors.pop(name, None)
            return (f"Renamed {name} to {renamed.name}. Open it with /workflow view {renamed.name}; "
                    f"agents read it at wf:{renamed.name}.")

    # -- agents -----------------------------------------------------------------

    def target(self, store: WorkflowStore, agent: dict) -> dict:
        """Current websocket target for a live agent, refreshed from the live registry if needed."""
        if agent.get("websocket_url") and process_evidence(agent.get("process_identity")) == "alive":
            return {"session_id": agent["session_id"], "instance_id": agent["instance_id"],
                    "websocket_url": agent["websocket_url"]}
        root = (self.launch or {}).get("store_root")
        if root and agent.get("session_id"):
            for row in list_live_sessions(root, include_stale=False):
                identity = row.process_identity.to_dict() if row.process_identity else None
                if row.session_id == agent["session_id"] and process_evidence(identity) == "alive":
                    store.update_agent(agent["id"], instance_id=row.instance_id,
                                       websocket_url=row.websocket_url, process_identity=identity)
                    return {"session_id": row.session_id, "instance_id": row.instance_id,
                            "websocket_url": row.websocket_url}
        raise WorkflowError(f"{agent['id']} is not running")

    def send(self, store: WorkflowStore, agent: dict, action: str, message: str = "") -> None:
        target = self.target(store, agent)
        run_async(lambda: send_to_agent(target, action, message), timeout=30)

    def liveness(self, store: WorkflowStore, agent: dict) -> str:
        handle = self.handles.get(self.handle_key(store, agent))
        if handle is not None and getattr(handle, "ref", None) is not None \
                and handle.ref.instance_id == agent.get("instance_id"):
            status = handle.poll().status
            if status in {"exited"}:
                return "dead"
            if status in {"starting", "ready"}:
                return "alive"
        return process_evidence(agent.get("process_identity"))

    def launch_agent(self, store: WorkflowStore, agent: dict, message: str, checkpoint=None,
                     recovery_action: str | None = None) -> None:
        """Start (or resume) one agent's session. Persists identity before spawning."""
        session_id = agent.get("session_id") or str(uuid.uuid4())
        target = InstanceRef.new_instance(session_id)
        request = RuntimeLaunch(
            target=target, source=Resume(checkpoint) if checkpoint else Fresh(),
            settings=self.settings_for(agent["folder"], agent["model"]),
            parent_session_id=self.me["session_id"], kind=f"wf:{store.name}:{agent['id']}",
            title=f"{store.name}:{agent['id']}", lifetime="independent",
            startup_mode="paused" if checkpoint else "normal",
            first_user_message="" if checkpoint else message,
        )
        Path(agent["folder"]).mkdir(parents=True, exist_ok=True)
        plan = self.launcher.prepare(request, cwd=agent["folder"],
                                     env=dict(os.environ, AZO_WORKFLOW=store.name, AZO_WORKFLOW_AGENT=agent["id"]))
        store.update_agent(agent["id"], session_id=session_id, instance_id=target.instance_id, state="starting",
                           project=self.launch["project"], launch_started=time.time(), websocket_url=None,
                           process_identity=None, release="resume" if checkpoint else None,
                           recovery_action=recovery_action)
        try:
            handle = self.launcher.start(plan)
        except Exception:
            store.update_agent(agent["id"], process_identity=None, problem="launch did not complete")
            raise
        self.handles[(store.name, agent["id"], target.instance_id)] = handle
        identity = capture_process_identity(handle.process.pid)
        store.update_agent(agent["id"], pid=handle.process.pid,
                           process_identity=identity.to_dict() if identity else None)

    def resume_agent(self, store: WorkflowStore, agent: dict) -> str | None:
        """Relaunch a stopped agent from its saved checkpoint. Returns a problem or None."""
        if not agent.get("session_id"):
            return f"{agent['id']} never started, so there is nothing to resume"
        with store.lock():
            # Claim the relaunch so a second controller (mid-handoff) cannot start a duplicate writer.
            current = store.agent(agent["id"])
            mine = self.managed.get(store.name)
            if mine is not None and int(store.control().get("epoch", 0)) != mine["epoch"]:
                return f"{store.name} is now controlled by another session"
            if current.get("instance_id") != agent.get("instance_id") or current["state"] not in {
                    "starting", "running", "paused", "attention"}:
                return None  # Someone else already resumed or ended it.
            current["state"] = "starting"
            store.save_agent(current)
        try:
            saved = saved_checkpoint(agent.get("project") or self.launch["project"], agent["session_id"])
        except FileNotFoundError:
            return f"{agent['id']} has no saved checkpoint to resume from"
        except Exception as exc:
            return f"{agent['id']}'s checkpoint could not be read: {exc}"
        action, reason = checkpoint_recovery(saved.document)
        if action in {"uncertain", "done"}:
            return (f"{agent['id']} checkpoint {saved.commit_id}: {reason}. "
                    "External work may already have happened; inspect its transcript and artifacts before retrying")
        try:
            self.launch_agent(store, agent, "", checkpoint=CheckpointRef(agent["session_id"], saved.commit_id),
                              recovery_action=action)
        except Exception as exc:
            return f"{agent['id']} could not be relaunched: {exc}"
        store.update_agent(agent["id"], resumes=int(agent.get("resumes", 0)) + 1, last_resume=time.time())
        return None

    def resume_paused(self, store: WorkflowStore, group: str | None) -> tuple[list[str], list[str]]:
        resumed, problems = [], []
        for agent in store.agents(group):
            if agent["state"] != "paused":
                continue
            alive = self.liveness(store, agent)
            if alive == "alive" and agent.get("websocket_url"):
                try:
                    if agent.get("release") in {"resume", "continue"}:
                        self.release_ready(store, agent, True)
                    else:
                        self.send(store, agent, "continue")
                        store.update_agent(agent["id"], state="running")
                    if store.agent(agent["id"])["state"] == "running":
                        resumed.append(agent["id"])
                except Exception as exc:
                    problems.append(f"Could not resume {agent['id']}: {exc}")
            elif alive == "alive":
                store.update_agent(agent["id"], state="starting", release=agent.get("release") or "continue")
                resumed.append(agent["id"])
            elif alive == "dead":
                problem = self.resume_agent(store, agent)
                if problem:
                    self.mark_attention(store, agent, problem + "; use restart_agent to give its item to a fresh agent")
                    problems.append(problem + ".")
                else:
                    resumed.append(agent["id"])
            else:
                problems.append(f"Could not tell whether {agent['id']} is still running; left it paused.")
        return resumed, problems

    def mark_attention(self, store: WorkflowStore, agent: dict, problem: str) -> None:
        store.update_agent(agent["id"], state="attention", problem=problem)
        store.log("agent_needs_attention", agent=agent["id"], problem=problem)

    # -- scheduling ---------------------------------------------------------------

    def tick(self, name: str) -> list[str]:
        """One scheduling pass. Returns descriptions of agents started."""
        store = WorkflowStore(name)
        if not store.exists():
            self.managed.pop(name, None)
            return []
        control = store.control()
        mine = self.managed.get(name)
        holder = control.get("controller") or {}
        if mine is None:
            return []
        if (holder.get("instance_id") != (self.me or {}).get("instance_id")
                or int(control.get("epoch", 0)) != mine["epoch"]):
            self.managed.pop(name, None)
            if control.get("state") != "archived":
                who = holder.get("session_id", "another session")[:8] if holder else "nobody"
                self.notice(f"Workflow {name} is now controlled by session {who}; this session stopped scheduling it.")
            return []
        definition = store.definition()
        paused_groups = set(control.get("paused_groups") or [])
        running = control.get("state") == "running"
        for agent in store.agents():
            if agent["state"] in {"starting", "running"}:
                self.observe(store, agent, definition, running and agent["group"] not in paused_groups)
        self.report_attention(store)
        started = []
        if running:
            started = self.start_agents(store, definition, paused_groups)
        self.report_new_items(store, definition)
        self.reap_children()
        return started

    def release_ready(self, store: WorkflowStore, agent: dict, should_run: bool) -> None:
        """Release a ready process before claiming it is running; pause wins."""
        agent = store.agent(agent["id"])
        if agent["state"] not in ACTIVE_STATES:
            return
        control = store.control()
        mine = self.managed.get(store.name)
        if (mine is None or control.get("epoch") != mine["epoch"]
                or (control.get("controller") or {}).get("instance_id") != self.me["instance_id"]):
            return
        should_run = (should_run and control.get("state") == "running"
                      and agent["group"] not in control.get("paused_groups", []))
        try:
            release = agent.get("release")
            if not should_run:
                if release in {"resume", "continue"}:
                    store.update_agent(agent["id"], state="paused")
                else:
                    problem = self.pause_one(store, agent)
                    if problem:
                        raise WorkflowError(problem)
                return
            if release in {"resume", "continue"}:
                action = "continue" if release == "continue" else agent.get("recovery_action")
                if action not in {"continue", "message"}:
                    self.mark_attention(store, agent, "recovery action is missing; checkpoint must be reviewed")
                    return
                message = ("Your session was restarted from a checkpoint after an interruption. "
                           "Inspect saved results and existing processes/artifacts before continuing. "
                           "Do not repeat a command just because its output is missing; if its outcome is "
                           "uncertain, report that uncertainty instead of retrying it. "
                           "Continue from confirmed progress and call finish_task when done.")
                self.send(store, agent, action, message if action == "message" else "")
            with store.lock():
                current = store.agent(agent["id"])
                if current["state"] in ACTIVE_STATES and current.get("instance_id") == agent.get("instance_id"):
                    current.update(state="running", release=None)
                    store.save_agent(current)
        except Exception as exc:
            self.mark_attention(store, agent,
                                f"could not confirm recovery release: {exc}; inspect the live session before retrying")

    def observe(self, store: WorkflowStore, agent: dict, definition: dict, should_run: bool) -> None:
        agent = store.agent(agent["id"])
        if agent["state"] not in {"starting", "running"}:
            return
        handle = self.handles.get(self.handle_key(store, agent))
        if agent["state"] == "starting":
            observation = handle.poll() if handle is not None else None
            if observation is not None and observation.status == "ready":
                ready = observation.ready or {}
                agent = store.update_agent(agent["id"], websocket_url=ready.get("websocket_url"),
                                           instance_id=ready.get("instance_id", agent["instance_id"]),
                                           process_identity=ready.get("process_identity") or agent.get("process_identity"))
                self.release_ready(store, agent, should_run)
                return
            if observation is not None and observation.status == "starting":
                if time.time() - float(agent.get("launch_started") or 0) > READY_TIMEOUT_SECONDS:
                    self.mark_attention(store, agent, "did not finish starting within two minutes")
                return
            if observation is None and process_evidence(agent.get("process_identity")) == "alive":
                # Launched by an earlier instance of this session: find it in the live registry.
                try:
                    self.target(store, agent)
                except WorkflowError:
                    if time.time() - float(agent.get("launch_started") or 0) > READY_TIMEOUT_SECONDS:
                        self.mark_attention(store, agent, "is running but cannot be reached")
                    return
                agent = store.agent(agent["id"])
                self.release_ready(store, agent, should_run)
                return
        status = self.liveness(store, agent)
        if status != "dead":
            return
        agent = store.agent(agent["id"])
        if agent["state"] not in {"starting", "running"}:
            return  # It finished (or was changed) while we looked.
        self.handles.pop(self.handle_key(store, agent), None)
        if not should_run:
            store.update_agent(agent["id"], state="paused")
            return
        recent = time.time() - float(agent.get("last_resume") or 0) < 600
        if recent and int(agent.get("resumes", 0)) >= MAX_QUICK_RESUMES:
            self.mark_attention(store, agent, f"stopped unexpectedly {agent['resumes']} times in a row; "
                                              "read its transcript, then use restart_agent")
            return
        problem = self.resume_agent(store, agent)
        if problem:
            self.mark_attention(store, agent, problem + "; use restart_agent to give its item to a fresh agent")

    def start_agents(self, store: WorkflowStore, definition: dict, paused_groups: set) -> list[str]:
        planned = []
        with store.lock():
            control = store.control()
            mine = self.managed.get(store.name)
            if (mine is None or control.get("epoch") != mine["epoch"]
                    or (control.get("controller") or {}).get("instance_id") != self.me["instance_id"]
                    or control.get("state") != "running"):
                return []
            definition = store.definition()
            paused_groups = set(control.get("paused_groups") or [])
            last_started = dict(control.get("last_started") or {})
            for group, settings in definition["agents"].items():
                if group in paused_groups or settings["count"] <= 0:
                    continue
                active = sum(a["state"] in ACTIVE_STATES for a in store.agents(group))
                room = settings["count"] - active
                while room > 0:
                    item = None
                    if settings["takes_from"]:
                        number = store.counters().get("agents", {}).get(group, 0) + 1
                        provisional = f"agt_{group_short(group)}_{number}"
                        item = store.take_next(settings["takes_from"], provisional)
                        if item is None:
                            break
                    elif settings["every"]:
                        last = float(last_started.get(group) or 0)
                        if time.time() - last < settings["every"]:
                            break
                    folder_template = (settings["folder"] or definition["folder"]
                                       or control.get("default_folder") or self.launch["cwd"])
                    agent = store.new_agent(group, version=store.version, item_id=item["id"] if item else None,
                                            sends_to=list(settings["sends_to"]), folder="", model=None)
                    agent["folder"] = render_folder(store, folder_template, agent)
                    store.save_agent(agent)
                    last_started[group] = time.time()
                    planned.append((agent, settings, item))
                    room -= 1
            store.update_control(last_started=last_started)
        started = []
        for agent, settings, item in planned:
            try:
                model = self.model_for(definition, settings)
                agent = store.update_agent(agent["id"], model=model)
                message = render_instructions(store, settings, agent, item, self.me["session_id"])
                self.launch_agent(store, agent, message)
                store.log("agent_started", agent=agent["id"], item=item["id"] if item else None)
                started.append(f"Started {agent['id']}" + (f" on {item['id']} from {item['queue']}." if item else "."))
            except Exception as exc:
                self.launch_failed(store, agent, item, settings, exc)
        return started

    def launch_failed(self, store, agent, item, settings, exc) -> None:
        group = agent["group"]
        with store.lock():
            if item:
                store.return_item(item["id"], agent["id"], by="scheduler")
            control = store.control()
            paused = set(control.get("paused_groups") or []) | {group}
            store.update_control(paused_groups=sorted(paused))
        store.update_agent(agent["id"], state="stopped", problem=f"could not start: {exc}")
        store.log("agent_start_failed", agent=agent["id"], error=str(exc))
        self.notice(f"Workflow {store.name}: could not start an agent in {group}: {exc}. Paused {group}; "
                    f"fix the problem (see wf:{store.name}:{group}), then run it again.")

    def report_attention(self, store: WorkflowStore) -> None:
        for agent in store.agents():
            if agent["state"] == "attention":
                self.notice(f"Workflow {store.name}: {agent['id']} needs attention: {agent.get('problem')}. "
                            f"See wf:{store.name}:{agent['id']}.",
                            key=(store.name, agent["id"], agent.get("problem")))

    def report_new_items(self, store: WorkflowStore, definition: dict) -> None:
        mine = self.managed.get(store.name)
        if mine is None:
            return
        watermark = mine.get("watermark", 0)
        latest = store.counters().get("item", 0)
        if latest <= watermark:
            return
        mine["watermark"] = latest
        watched = {q for q, s in definition["queues"].items() if s["notify_on_new"]}
        if not watched:
            return
        fresh = [i for i in store.items() if int(i["id"][5:]) > watermark and i["queue"] in watched]
        by_queue = collections.defaultdict(list)
        for item in fresh:
            by_queue[item["queue"]].append(f"{item['id']} ({item['title']})")
        for queue, entries in by_queue.items():
            self.notice(f"Workflow {store.name}: new items in {queue}: {', '.join(entries)}. "
                        f"See wf:{store.name}:{queue}.")

    # -- other controller operations ---------------------------------------------

    def message_agents(self, name: str, message: str, group: str | None) -> str:
        store = WorkflowStore(name).require()
        if group is not None:
            check_group_id(group, {a["group"] for a in store.agents()} | set(store.definition()["agents"]))
        if not message.strip():
            raise WorkflowError("message is empty")
        sent, skipped, problems = [], [], []
        for agent in store.agents(group):
            if agent["state"] == "running":
                try:
                    self.send(store, agent, "message", message)
                    sent.append(agent["id"])
                except Exception as exc:
                    problems.append(f"Could not reach {agent['id']}: {exc}")
            elif agent["state"] in {"paused", "starting"}:
                skipped.append(agent["id"])
        if not sent and not problems:
            text = "No running agents" + (f" in {group}" if group else "") + "; nothing was sent."
        else:
            text = f"Sent to {', '.join(sent)}." if sent else "Sent to no agents."
        if skipped:
            text += f" Skipped {', '.join(skipped)} (paused or still starting)."
        return "\n".join([text, *problems])

    def restart_agent(self, name: str, agent_id: str) -> str:
        with self.lock:
            store = WorkflowStore(name).require()
            agent = store.agent(check_agent_id(agent_id))
            if agent["state"] in {"finished", "restarted"}:
                raise WorkflowError(f"{agent_id} has already {agent['state']}")
            if self.liveness(store, agent) == "alive":
                try:
                    self.send(store, agent, "shutdown")
                except Exception as exc:
                    LOG.warning("Could not shut down %s: %s", agent_id, exc)
            with store.lock():
                if agent.get("item"):
                    store.return_item(agent["item"], agent_id, by="restart_agent")
            store.update_agent(agent_id, state="restarted", problem=None)
            store.log("agent_restarted", agent=agent_id)
            text = f"Stopped {agent_id} and discarded its progress."
            if agent.get("item"):
                text += f" {agent['item']} is back in {store.item(agent['item'])['queue']} for a fresh agent."
            if name not in self.managed:
                text += f" This session does not control {name}; run it to start new agents."
            return text

    def change_setting(self, name: str, kind: str, target: str | None, setting: str, raw: str) -> str:
        """Apply one /workflow set change and describe its effect."""
        with self.lock:
            store = WorkflowStore(name).require()
            change = prepare_setting(store, kind, target, setting, raw, check_model=self.check_model,
                                     default_model=self.default_model())
            label, setting = change["label"], change["setting"]
            if change["unchanged"]:
                return f"{label} {setting} is already {change['new_text']}; nothing changed."
            result = store.set_setting(change["section"], change["name"], change["key"], change["value"],
                                       reason=change["reason"], by="user")
            text = (f"Set {label} {setting} from {change['old_text']} to {change['new_text']} "
                    f"(def_{result['version']}).")
            definition = store.definition()
            active = [a for a in store.agents() if a["state"] in ACTIVE_STATES]
            if setting == "count":
                if change["value"] < int(change["old_text"]):
                    text += " Extra running agents finish their current item first."
                elif name in self.managed:
                    started = self.tick(name)
                    text += (" " + " ".join(started)) if started else ""
                elif store.control().get("state") == "running":
                    text += " The session controlling it will start agents shortly."
                else:
                    text += f" {name} is paused; agents start when it runs."
            elif setting == "model":
                stale = []
                for agent in active:
                    group = definition["agents"].get(agent["group"])
                    if kind == "group" and agent["group"] != target or group is None:
                        continue
                    model, _ = effective_model(definition, group, self.default_model())
                    if agent.get("model") and agent["model"] != model:
                        stale.append(f"{agent['id']} ({agent['model']})")
                text += " New agents use it."
                if stale:
                    text += (f" {', '.join(stale)} keep their current model; restart them with restart_agent "
                             "or let them finish to switch.")
            elif setting == "folder":
                scope = [a["id"] for a in active if kind == "workflow" or a["group"] == target]
                text += " New agents start there."
                if scope:
                    text += f" {', '.join(scope)} keep their current folder."
            elif setting == "every":
                text += " It applies from the group's next start."
            elif setting == "description":
                text += " It shows in /workflow list."
            elif setting == "notify":
                text += (f" The controlling agent will be told about new items in {target}."
                         if change["value"] else f" New items in {target} no longer notify anyone.")
            return text

    # -- startup adoption ---------------------------------------------------------

    def adopt(self, state) -> None:
        """After a restart or reload, keep running workflows this session controlled."""
        self.bind(state)
        if not self.me or self.adopted_instance == self.me["instance_id"]:
            return
        self.adopted_instance = self.me["instance_id"]
        for name in list_workflow_names():
            store = WorkflowStore(name)
            control = store.control()
            holder = control.get("controller") or {}
            if (control.get("state") != "running" or holder.get("session_id") != self.me["session_id"]
                    or holder.get("instance_id") == self.me["instance_id"]):
                continue
            try:
                self.take_control(store)
                self.notice(f"Workflow {name} is running again under this session after a restart. "
                            f"See wf:{name}.")
            except Exception as exc:
                self.notice(f"Workflow {name} could not resume after a restart: {exc}")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def function_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    # Strict providers may require every property. Null preserves omission semantics.
    properties = deepcopy(properties)
    for key, prop in properties.items():
        if key not in required:
            kinds = prop["type"] if isinstance(prop["type"], list) else [prop["type"]]
            prop["type"] = list(dict.fromkeys([*kinds, "null"]))
            prop["description"] = prop.get("description", "") + " Omit or pass null to use the default."
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


def text_property(description: str) -> dict:
    return {"type": "string", "description": description}


WORKFLOW_NAME_PROPERTY = text_property('Name of the workflow, e.g. "miner".')
AGENT_GROUP_PROPERTY = text_property("Limit to one agent group, e.g. grp_reverse. Omit for the whole workflow.")
NOTE_PROPERTY = text_property("Optional note shown to the next agent that takes these items.")


def buffer_text(state, buffer_id: str) -> str:
    manager = getattr(state, "buffer_manager", None)
    if manager is None:
        raise WorkflowError("buffers are unavailable in this session")
    try:
        return manager.resolve_for_read(state, str(buffer_id)).text
    except KeyError:
        raise WorkflowError(f"there is no buffer {buffer_id}") from None


class WorkflowTool(Tool):
    """A tool with separate behavior (and schema) for controlling sessions and workers."""
    group = WORKER_TOOLS_GROUP
    optional_reads = {"_session_kind", "_session_id", "_instance_id", "_agent_zoo_context", "buffer_manager",
                      "session_launcher", "_persistence_local_root", "session_cwd"}
    writes = {"done", "result", "shutdown", "shutdown_reason", "shutdown_save", "shutdown_registry_status",
              "shutdown_spawned_policy"}

    def __init__(self, name, *, controller_schema=None, worker_schema=None, on_controller=None, on_worker=None,
                 plugin=None):
        self.name = name
        self.controller_schema, self.worker_schema = controller_schema, worker_schema
        self.on_controller, self.on_worker = on_controller, on_worker
        self.plugin = plugin
        static = controller_schema or worker_schema
        if callable(static):
            static = static({"sends_to": []})  # Workers see their own queues; this is the generic form.
        self.schema = static
        self.description = static["function"]["description"]

    def available(self, state) -> bool:
        return bool(self.worker_schema if worker_identity(state) else self.controller_schema)

    def _publish(self, state):
        worker = worker_identity(state)
        schema = self.worker_schema if worker else self.controller_schema
        if worker and callable(schema):
            schema = schema(self.plugin.worker_context(state))
        state.tool_schemas[self.name] = {"schema": schema or self.schema, "available": bool(schema)}

    def transform(self, state):
        for call, kwargs in self.pending(state):
            worker = worker_identity(state)
            handler = self.on_worker if worker else self.on_controller
            try:
                if handler is None:
                    raise WorkflowError(f"{self.name} is not available in this session")
                if not worker:
                    self.plugin.controller.bind(state)
                call.result = str(handler(state, **(kwargs or {})))
            except (WorkflowError, TypeError, ValueError, OSError) as exc:
                call.result = f"ERROR: {exc}"
                call.error = True
            except Exception as exc:
                LOG.exception("Workflow tool %s failed", self.name)
                call.result = f"ERROR: {type(exc).__name__}: {exc}"
                call.error = True
            self._notify(state, call)
        return state


class WorkflowsPlugin:
    """Tool handlers and worker context, shared by the components of one pipeline."""

    def __init__(self, controller: WorkflowController):
        self.controller = controller

    # -- worker context ---------------------------------------------------------

    def worker_context(self, state, refresh: bool = False) -> dict:
        identity = worker_identity(state)
        cached = self.controller.worker_record
        if cached is None or refresh or cached.get("id") != identity["agent"]:
            store = WorkflowStore(identity["workflow"])
            try:
                cached = store.agent(identity["agent"])
            except WorkflowError:
                cached = {"id": identity["agent"], "group": "", "sends_to": [], "item": None, "state": "unknown"}
            cached["workflow"] = identity["workflow"]
            self.controller.worker_record = cached
        return cached

    # -- controller handlers -------------------------------------------------------

    def create_workflow(self, state, workflow_name, definition_buffer):
        store = WorkflowStore(workflow_name)
        text = buffer_text(state, definition_buffer)
        check_definition_models(parse_definition(split_version(text)[1]), self.controller.config)
        definition = store.create(text, by=self.controller.me_label())
        store.update_control(default_folder=get_session_cwd(state))
        groups = "\n".join(f"  {g}: {describe_group(s)}" for g, s in definition["agents"].items())
        queues = ", ".join(definition["queues"]) or "none"
        return (f"Created workflow {store.name} (def_1). Nothing is running yet.\n"
                f"Queues: {queues}\nAgent groups:\n{groups}\n"
                f"Add starting items with add_items, then ask the user before running it with "
                f'command_workflow(workflow_name="{store.name}", command="run"). The user can also type '
                f"/workflow run {store.name}. See wf:{store.name}.")

    def update_workflow(self, state, workflow_name, definition_buffer, reason, preview_only=False):
        store = WorkflowStore(workflow_name).require()
        if not str(reason or "").strip():
            raise WorkflowError("reason is required: one sentence saying why the definition changed")
        text = buffer_text(state, definition_buffer)
        check_definition_models(parse_definition(split_version(text)[1]), self.controller.config)
        result = store.update(text, reason=str(reason).strip(), by=self.controller.me_label(),
                              preview_only=bool(preview_only))
        effects = "\n".join(f"- {e}" for e in result["effects"])
        if not result["changed"]:
            return f"The definition is unchanged (still def_{result['version']})."
        if preview_only:
            head = f"Preview only; nothing changed. Applying it would create def_{result['from_version'] + 1}."
        else:
            head = f"Updated {store.name} from def_{result['from_version']} to def_{result['version']}."
        return f"{head}\n\nDiff:\n{result['diff']}\nEffects:\n{effects}"

    def command_workflow(self, state, workflow_name, command, agent_group=None):
        command = str(command or "").strip().lower()
        agent_group = None if agent_group in (None, "") else agent_group
        if command == "run":
            return self.controller.run(workflow_name, agent_group)
        if command == "pause":
            return self.controller.pause(workflow_name, agent_group)
        if command == "archive":
            if agent_group:
                raise WorkflowError("archive applies to the whole workflow; omit agent_group")
            return self.controller.archive(workflow_name)
        if command == "unarchive":
            if agent_group:
                raise WorkflowError("unarchive applies to the whole workflow; omit agent_group")
            return self.controller.unarchive(workflow_name)
        raise WorkflowError(f'unknown command "{command}"; use run, pause, archive or unarchive')

    def rename_workflow(self, state, workflow_name, new_name):
        return self.controller.rename(workflow_name, new_name)

    def set_agent_count(self, state, workflow_name, agent_group, count):
        store = WorkflowStore(workflow_name).require()
        result = store.set_count(agent_group, int(count), by=self.controller.me_label())
        if not result["changed"]:
            return f"{agent_group} already has count {count}."
        text = f"Set {agent_group} count from {result['old']} to {count} (def_{result['version']})."
        if int(count) < result["old"]:
            text += " Extra running agents finish their current item first."
        elif workflow_name in self.controller.managed:
            with self.controller.lock:
                started = self.controller.tick(store.name)
            if started:
                text += " " + " ".join(started)
        elif store.control().get("state") == "running":
            text += " The controlling session will start agents shortly."
        return text

    def message_agents(self, state, workflow_name, message, agent_group=None):
        agent_group = None if agent_group in (None, "") else agent_group
        return self.controller.message_agents(workflow_name, str(message), agent_group)

    def restart_agent(self, state, workflow_name, agent):
        return self.controller.restart_agent(workflow_name, agent)

    def add_items(self, state, workflow_name, queue, from_buffers, note=None):
        store = WorkflowStore(workflow_name).require()
        known = set(store.definition()["queues"]) | {TRASH}
        check_queue_id(queue, known)
        buffers = [from_buffers] if isinstance(from_buffers, str) else list(from_buffers or [])
        if not buffers:
            raise WorkflowError("from_buffers is empty")
        texts = [buffer_text(state, b) for b in buffers]
        created = [store.add_item(queue, text, by="user", note=note) for text in texts]
        return (f"Added {', '.join(m['id'] for m in created)} to {queue}."
                + self.taker_hint(store, queue))

    def move_items(self, state, workflow_name, item_ids, to_queue, note=None):
        store = WorkflowStore(workflow_name).require()
        known = set(store.definition()["queues"]) | {TRASH} | {i["queue"] for i in store.items()}
        check_queue_id(to_queue, known)
        ids = [item_ids] if isinstance(item_ids, str) else list(item_ids or [])
        moved = store.move_items(ids, to_queue, by="user", note=note)
        return f"Moved {', '.join(m['id'] for m in moved)} to {to_queue}." + self.taker_hint(store, to_queue)

    @staticmethod
    def taker_hint(store, queue) -> str:
        if queue == TRASH:
            return " Move them back out of q_trash to restore them."
        takers = [g for g, s in store.definition()["agents"].items() if s["takes_from"] == queue]
        if not takers:
            return f" No agent group takes from {queue}."
        if store.control().get("state") != "running":
            return f" {', '.join(takers)} will take them when {store.name} runs."
        return f" {', '.join(takers)} will take them."

    # -- worker handlers ------------------------------------------------------------

    def worker_store(self, state):
        context = self.worker_context(state, refresh=True)
        return WorkflowStore(context["workflow"]), context

    def finish_task(self, state, summary, outputs=None):
        store, context = self.worker_store(state)
        pairs = []
        for entry in outputs or []:
            if not isinstance(entry, dict) or "buffer" not in entry or "queue" not in entry:
                raise WorkflowError('each output must look like {"buffer": "buf_...", "queue": "q_..."}')
            pairs.append((check_queue_id(entry["queue"]), buffer_text(state, entry["buffer"])))
        result = store.finish_agent(context["id"], summary=str(summary or ""), outputs=pairs)
        state.done = True
        state.result = str(summary or "")
        state.shutdown_reason = "suspend"
        state.shutdown_save = True
        state.shutdown_registry_status = "completed"
        state.shutdown_spawned_policy = "terminate"
        state.shutdown = True
        created = ", ".join(f"{m['id']} in {m['queue']}" for m in result["created"])
        return ("Task finished" + (f"; created {created}" if created else "") + ". This session is ending.")

    def worker_add_items(self, state, queue, from_buffers, note=None):
        store, context = self.worker_store(state)
        allowed = set(context.get("sends_to") or []) | {TRASH}
        buffers = [from_buffers] if isinstance(from_buffers, str) else list(from_buffers or [])
        if not buffers:
            raise WorkflowError("from_buffers is empty")
        texts = [buffer_text(state, b) for b in buffers]
        created = [store.add_item(queue, text, by=context["id"], note=note, from_item=context.get("item"),
                                  allowed=allowed) for text in texts]
        ids = [m["id"] for m in created]
        record = store.agent(context["id"])
        store.update_agent(context["id"], outputs=record.get("outputs", []) + ids)
        return f"Added {', '.join(ids)} to {queue}. Keep working, and call finish_task when your task is done."

    def worker_move_items(self, state, item_ids, to_queue, note=None):
        store, context = self.worker_store(state)
        allowed = set(context.get("sends_to") or []) | {TRASH}
        ids = [item_ids] if isinstance(item_ids, str) else list(item_ids or [])
        moved = store.move_items(ids, to_queue, by=context["id"], note=note, allowed=allowed)
        return f"Moved {', '.join(m['id'] for m in moved)} to {to_queue}."


def build_tools(plugin: WorkflowsPlugin) -> list[Tool]:
    outputs_property = {
        "type": "array",
        "description": "One entry per result, e.g. [{\"buffer\": \"buf_a1\", \"queue\": \"q_results\"}]. "
                       "Each buffer becomes one item; its first line is the title. May be empty.",
        "items": {"type": "object", "properties": {
            "buffer": text_property("Buffer holding the result text."),
            "queue": text_property("Queue to send it to."),
        }, "required": ["buffer", "queue"]},
    }

    def allowed_text(context) -> str:
        queues = list(context.get("sends_to") or []) + [TRASH]
        return "You can send to: " + ", ".join(queues) + "."

    def finish_schema(context):
        return function_schema(
            "finish_task",
            "Finish your assigned task and end this session. Your item is marked done whatever you send. "
            + allowed_text(context),
            {"summary": text_property("A few sentences on what you did and found, for whoever runs the workflow."),
             "outputs": outputs_property},
            ["summary"])

    def worker_add_schema(context):
        return function_schema(
            "add_items",
            "Add items to a queue before you finish, e.g. results you want delivered as you find them. "
            "Each buffer becomes one item; its first line is the title. " + allowed_text(context),
            {"queue": text_property("Queue to add to, e.g. q_results."),
             "from_buffers": {"type": "array", "items": {"type": "string"},
                              "description": "Buffer IDs; each becomes one item."},
             "note": NOTE_PROPERTY},
            ["queue", "from_buffers"])

    def worker_move_schema(context):
        return function_schema(
            "move_items",
            "Move waiting or finished items into another queue, e.g. to send blocked items back for another "
            "try. You cannot move items another agent is working on. " + allowed_text(context),
            {"item_ids": {"type": "array", "items": {"type": "string"}, "description": 'e.g. ["item_12"]'},
             "to_queue": text_property("Destination queue."),
             "note": text_property("Why you moved them; shown to the next agent.")},
            ["item_ids", "to_queue"])

    create = function_schema(
        "create_workflow",
        "Create a new workflow from a YAML definition stored in a buffer. Nothing runs until it is started "
        "with command_workflow or /workflow run. Queues are named q_*, agent groups grp_*. A group with "
        "takes_from starts one agent per item from that queue; a group without it keeps `count` agents "
        "running. Example definition:\n" + EXAMPLE_DEFINITION,
        {"workflow_name": text_property('New workflow name: lowercase letters, digits and underscores, e.g. "miner".'),
         "definition_buffer": text_property("Buffer holding the YAML definition.")},
        ["workflow_name", "definition_buffer"])
    update = function_schema(
        "update_workflow",
        "Replace a workflow's definition with the full YAML in a buffer. Copy wf:<name>:definition into a "
        "buffer, edit it, and pass it here. Returns a diff and the effects on running agents and waiting items. "
        "Running agents finish under the old definition; the new one applies to agents started afterward.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY,
         "definition_buffer": text_property("Buffer holding the complete new YAML definition."),
         "reason": text_property("One sentence saying why; saved in the workflow's history."),
         "preview_only": {"type": "boolean", "description": "Show the diff and effects without changing anything."}},
        ["workflow_name", "definition_buffer", "reason"])
    command = function_schema(
        "command_workflow",
        "Run, pause, archive or unarchive a workflow. run: take control in this session, resume paused agents "
        "where they left off, and start new agents up to each group's count (agents cost money; confirm with the "
        "user first). pause: start no new agents and interrupt running ones, keeping their work. archive: stop "
        "all agents and hide the workflow from the user's /workflow list (e.g. one-off or test workflows); its "
        "history stays readable. unarchive: bring an archived workflow back, paused.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY,
         "command": {"type": "string", "enum": ["run", "pause", "archive", "unarchive"], "description": "What to do."},
         "agent_group": AGENT_GROUP_PROPERTY},
        ["workflow_name", "command"])
    rename = function_schema(
        "rename_workflow",
        "Give a workflow a new name, keeping its definition, items, agents and history. The workflow must be "
        "paused or archived with no unfinished agents, because agent sessions are tied to the old name.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY,
         "new_name": text_property('New name: lowercase letters, digits and underscores, e.g. "test_failures".')},
        ["workflow_name", "new_name"])
    count = function_schema(
        "set_agent_count",
        "Set how many agents in a group run at once. Takes effect immediately and records a new definition "
        "version. When lowering, extra agents finish their current item first.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY,
         "agent_group": text_property("Agent group, e.g. grp_reverse."),
         "count": {"type": "integer", "description": "Agents to run at once, 0 to 64."}},
        ["workflow_name", "agent_group", "count"])
    message = function_schema(
        "message_agents",
        "Send a message to a workflow's running agents, as if the user typed it in their sessions.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY, "message": text_property("The message."),
         "agent_group": text_property("Only message this agent group, e.g. grp_reverse. Omit for all running agents.")},
        ["workflow_name", "message"])
    restart = function_schema(
        "restart_agent",
        "Stop one agent, discard its progress, and put its item back in its queue for a fresh agent. Use when an "
        "agent is stuck or cannot resume.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY, "agent": text_property("Agent ID, e.g. agt_reverse_3.")},
        ["workflow_name", "agent"])
    add = function_schema(
        "add_items",
        "Add items to a workflow queue. Each buffer becomes one item; its first line is the item's title.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY, "queue": text_property("Queue to add to, e.g. q_targets."),
         "from_buffers": {"type": "array", "items": {"type": "string"},
                          "description": "Buffer IDs; each becomes one item."},
         "note": NOTE_PROPERTY},
        ["workflow_name", "queue", "from_buffers"])
    move = function_schema(
        "move_items",
        "Move items to another queue: retry finished items by moving them back, reroute them, or discard them "
        "by moving them to q_trash (they can be moved back). Items an agent is working on cannot be moved; "
        "use restart_agent first.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY,
         "item_ids": {"type": "array", "items": {"type": "string"}, "description": 'e.g. ["item_12", "item_17"]'},
         "to_queue": text_property("Destination queue, e.g. q_targets or q_trash."),
         "note": text_property("Why they moved; shown to the next agent that takes them.")},
        ["workflow_name", "item_ids", "to_queue"])

    p = plugin
    return [
        WorkflowTool("create_workflow", controller_schema=create, on_controller=p.create_workflow, plugin=p),
        WorkflowTool("update_workflow", controller_schema=update, on_controller=p.update_workflow, plugin=p),
        WorkflowTool("command_workflow", controller_schema=command, on_controller=p.command_workflow, plugin=p),
        WorkflowTool("rename_workflow", controller_schema=rename, on_controller=p.rename_workflow, plugin=p),
        WorkflowTool("set_agent_count", controller_schema=count, on_controller=p.set_agent_count, plugin=p),
        WorkflowTool("message_agents", controller_schema=message, on_controller=p.message_agents, plugin=p),
        WorkflowTool("restart_agent", controller_schema=restart, on_controller=p.restart_agent, plugin=p),
        WorkflowTool("add_items", controller_schema=add, worker_schema=worker_add_schema,
                     on_controller=p.add_items, on_worker=p.worker_add_items, plugin=p),
        WorkflowTool("move_items", controller_schema=move, worker_schema=worker_move_schema,
                     on_controller=p.move_items, on_worker=p.worker_move_items, plugin=p),
        WorkflowTool("finish_task", worker_schema=finish_schema, on_worker=p.finish_task, plugin=p),
    ]


# ---------------------------------------------------------------------------
# Slash command
# ---------------------------------------------------------------------------

USAGE = ("/workflow list [archived]  |  /workflow view NAME[:GROUP|:QUEUE]  |  /workflow run NAME[:GROUP]  |  "
         "/workflow pause NAME[:GROUP]  |  /workflow rename NAME NEW_NAME  |  /workflow archive NAME  |  "
         "/workflow unarchive NAME  |  /workflow set NAME[:GROUP|:QUEUE] [SETTING VALUE]")
COMMAND_VERBS = "list, view, run, pause, set, rename, archive, unarchive"


def parse_target(text: str, *, allow_queue: bool = False) -> tuple[str, str, str | None]:
    """Parse ``miner``, ``miner:grp_reverse`` or (if allowed) ``miner:q_catalog``."""
    name, _, part = str(text).partition(":")
    name = check_workflow_name(name)
    if not part:
        return name, "workflow", None
    if allow_queue and part.startswith("q_"):
        return name, "queue", check_queue_id(part)
    if part.startswith("grp_"):
        return name, "group", check_group_id(part)
    kinds = "an agent group (grp_...) or a queue (q_...)" if allow_queue else "an agent group (grp_...)"
    raise WorkflowError(f'"{part}" is not {kinds}; did you mean {name}:{suggest_id("grp_", part)}?')


def parse_command(raw_args: str) -> dict:
    try:
        tokens = shlex.split(raw_args or "")
    except ValueError as exc:
        raise WorkflowError(f"could not parse the command: {exc}") from None
    if not tokens:
        return {"action": "list"}
    action = tokens[0].lower()
    if action == "list":
        if tokens[1:] in ([], ["archived"]):
            return {"action": "archived" if tokens[1:] else "list"}
        raise WorkflowError("usage: /workflow list [archived]")
    if action == "view":
        if len(tokens) != 2:
            raise WorkflowError("usage: /workflow view NAME[:GROUP|:QUEUE], e.g. /workflow view miner:grp_reverse")
        name, kind, target = parse_target(tokens[1], allow_queue=True)
        return {"action": "show", "name": name, "kind": kind, "target": target}
    if action in {"archive", "unarchive"}:
        if len(tokens) != 2 or ":" in tokens[1]:
            raise WorkflowError(f"usage: /workflow {action} NAME ({action} applies to a whole workflow)")
        return {"action": action, "name": check_workflow_name(tokens[1])}
    if action == "rename":
        if len(tokens) != 3:
            raise WorkflowError("usage: /workflow rename NAME NEW_NAME, e.g. /workflow rename fixes_1a2b test_failures")
        return {"action": action, "name": check_workflow_name(tokens[1]), "new_name": check_workflow_name(tokens[2])}
    if action in {"run", "pause"}:
        if len(tokens) == 3 and ":" not in tokens[1]:
            raise WorkflowError(f"name the group with a colon: /workflow {action} {tokens[1]}:{tokens[2]}")
        if len(tokens) != 2:
            raise WorkflowError(f"usage: /workflow {action} NAME[:GROUP], e.g. /workflow {action} miner:grp_reverse")
        name, kind, target = parse_target(tokens[1])
        return {"action": action, "name": name, "group": target}
    if action == "set":
        if len(tokens) < 2:
            raise WorkflowError("usage: /workflow set NAME[:GROUP|:QUEUE] SETTING VALUE, e.g. "
                                "/workflow set miner:grp_reverse count 4")
        name, kind, target = parse_target(tokens[1], allow_queue=True)
        if len(tokens) == 2:
            return {"action": "settings", "name": name, "kind": kind, "target": target}
        if len(tokens) == 3:
            raise WorkflowError(f"give a value, e.g. /workflow set {tokens[1]} {tokens[2]} VALUE; "
                                f"see the settings with /workflow set {tokens[1]}")
        return {"action": "set", "name": name, "kind": kind, "target": target,
                "setting": tokens[2].lower(), "value": " ".join(tokens[3:])}
    if len(tokens) == 1 and tokens[0].partition(":")[0] in list_workflow_names():
        raise WorkflowError(f'unknown command "{tokens[0]}"; to open a workflow, use /workflow view {tokens[0]}')
    raise WorkflowError(f'unknown command "{tokens[0]}"; use one of: {COMMAND_VERBS}. Usage: {USAGE}')


class WorkflowCommand:
    def __init__(self, plugin: WorkflowsPlugin):
        self.plugin = plugin

    @staticmethod
    def document(title: str, body: str, subtitle: str = "") -> CommandResult:
        return CommandResult(outputs=(InspectorDocument(title=title, subtitle=subtitle, body=body,
                                                        format="markdown"),))

    def __call__(self, ctx, *, raw_args=""):
        controller = self.plugin.controller
        try:
            request = parse_command(raw_args)
            if worker_identity(ctx.state):
                raise WorkflowError("workflow agents cannot control workflows")
            controller.bind(ctx.state)
            viewer = (controller.me or {}).get("session_id", "")
            if request["action"] in {"list", "archived"}:
                archived = request["action"] == "archived"
                subtitle, body = render_list(None, viewer, archived=archived)
                return self.document("Archived workflows" if archived else "Workflows", body, subtitle)
            store = WorkflowStore(request["name"]).require()
            state = store.control().get("state")
            if request["action"] == "archive" and state == "archived":
                raise WorkflowError(f"{store.name} is already archived")
            if request["action"] == "unarchive" and state != "archived":
                raise WorkflowError(f"{store.name} is not archived")
            if request["action"] == "rename" and WorkflowStore(request["new_name"]).exists():
                raise WorkflowError(f'a workflow named "{request["new_name"]}" already exists; choose another name')
            if request["action"] == "show":
                kind, target = request["kind"], request["target"]
                if kind == "workflow":
                    subtitle, body = render_overview_md(store, viewer, controller.default_model())
                elif kind == "group":
                    subtitle, body = render_group_md(store, target, controller.default_model())
                else:
                    subtitle, body = render_queue_md(store, target)
                return self.document(target_label(store, kind, target), body, subtitle)
            if request["action"] == "settings":
                text = render_settings(store, request["kind"], request["target"], controller.default_model())
                return self.document(f"Settings: {target_label(store, request['kind'], request['target'])}", text)
            if request["action"] == "set":
                # Validate now so mistakes are reported without touching the workflow.
                change = prepare_setting(store, request["kind"], request["target"], request["setting"],
                                         request["value"], check_model=controller.check_model,
                                         default_model=controller.default_model())
                if change["unchanged"]:
                    return CommandResult.notice(
                        f"{change['label']} {change['setting']} is already {change['new_text']}; nothing changed.")
            if request["action"] == "run":
                controller.require_launch()
        except WorkflowError as exc:
            raise CommandError(str(exc)) from exc

        if request["action"] == "set":
            def action():
                return controller.change_setting(request["name"], request["kind"], request["target"],
                                                 request["setting"], request["value"])
            receipt = f"Setting {change['label']} {change['setting']} to {change['new_text']}..."
        elif request["action"] == "rename":
            def action():
                return controller.rename(request["name"], request["new_name"])
            receipt = f"Renaming {request['name']} to {request['new_name']}..."
        elif request["action"] in {"archive", "unarchive"}:
            method = controller.archive if request["action"] == "archive" else controller.unarchive

            def action():
                return method(request["name"])
            receipt = f"{'Archiving' if request['action'] == 'archive' else 'Restoring'} {request['name']}..."
        else:
            method = controller.run if request["action"] == "run" else controller.pause

            def action():
                return method(request["name"], request["group"])
            verb = "Running" if request["action"] == "run" else "Pausing"
            scope = f"{request['name']}:{request['group']}" if request["group"] else request["name"]
            receipt = f"{verb} {scope}..."

        async def work():
            try:
                return {"ok": True, "text": await asyncio.to_thread(action)}
            except Exception as exc:
                return {"ok": False, "text": f"Workflow {request['name']}: {exc}"}

        ctx.defer(work, then=self.complete)
        return CommandResult.notice(receipt)

    @staticmethod
    def complete(ctx, result):
        if result["ok"]:
            ctx.session.inject("The user typed a /workflow command. Result (already done; do not repeat it):\n"
                               + result["text"], role="user", system_generated=True)
            return CommandResult.notice(result["text"].splitlines()[0])
        return CommandResult.notice(result["text"], level="warning")


# ---------------------------------------------------------------------------
# Buffers, prompts, notices
# ---------------------------------------------------------------------------

class WorkflowBuffers:
    reads = {"buffer_manager"}
    writes = {"buffer_manager"}
    optional_reads = {"_session_id", "_agent_zoo_context"}

    def __init__(self, plugin: "WorkflowsPlugin | None" = None):
        self.plugin = plugin

    def __call__(self, state):
        state.buffer_manager.register_special_buffer_namespace("wf:", self.render, replace=True)
        return state

    def render(self, state, buffer_id):
        if not buffer_id.startswith("wf:"):
            return None
        default_model = ""
        if self.plugin is not None and not worker_identity(state):
            default_model = self.plugin.controller.default_model()

        def transcript(session_id):
            parts = buffer_id.split(":")
            agent = WorkflowStore(parts[1]).agent(parts[2])
            return saved_transcript(agent.get("project") or "default", session_id)

        text = render_view(buffer_id, viewer_session_id=str(getattr(state, "_session_id", "") or ""),
                           transcript=transcript, default_model=default_model)
        if text is None:
            return None
        return ReadonlyBufferView(id=buffer_id, path=f"workflow://{buffer_id[3:]}", text=text)


WORKER_GUIDANCE = """\
# Workflow agent
You are {agent} in workflow "{workflow}" (group {group}). Your task is in the first user message.
- When the task is done, call finish_task(summary=..., outputs=[{{"buffer": "buf_...", "queue": "q_..."}}]).
  Put each result in its own buffer first (create_buffer, append_lines). finish_task ends this session.
- You can send results to: {queues}. Use add_items to deliver results before you finish.
- Read the workflow with view(buffer="wf:{workflow}") and queues with wf:{workflow}:<queue>, e.g. to avoid duplicates.
- Do the work yourself; do not start other agents. Messages from the person running the workflow arrive as user messages."""


class WorkflowSystemPrompt(RenderTransformRegistrar):
    """Durable role guidance on the system entry for workers and controllers."""
    reads = {"entries"}
    optional_reads = {"_session_kind", "_session_id"}
    transform_name = "workflow role guidance"

    def __init__(self, plugin: WorkflowsPlugin):
        self.plugin = plugin

    def __call__(self, state):
        if not state.entries:
            return state
        entry = state.entries[0]
        if self.transform_name not in entry.render_transform_names(MODEL_RENDER_CHANNEL, pending=True):
            entry.append_render_transform(MODEL_RENDER_CHANNEL, self.transform, name=self.transform_name)
        return state

    def transform(self, messages, state):
        if not messages or messages[0].get("role") != "system":
            return messages
        identity = worker_identity(state)
        if identity:
            context = self.plugin.worker_context(state)
            block = WORKER_GUIDANCE.format(agent=identity["agent"], workflow=identity["workflow"],
                                           group=context.get("group") or "unknown",
                                           queues=", ".join(list(context.get("sends_to") or []) + [TRASH]))
        elif self.plugin.controller.managed:
            names = ", ".join(sorted(self.plugin.controller.managed))
            block = (f"# Workflows\nThis session controls running workflows: {names}. "
                     'Activate skill(name="Workflows") before changing them; read view(buffer="wf:<name>") '
                     "for their state.")
        else:
            return messages
        first = dict(messages[0])
        content = first.get("content") or ""
        if isinstance(content, list):
            kind = "input_text" if any(i.get("type") == "input_text" for i in content if isinstance(i, dict)) else "text"
            first["content"] = [*content, {"type": kind, "text": block}]
        else:
            first["content"] = str(content).rstrip() + "\n\n" + block
        return [first, *messages[1:]]


class WorkflowSkillRegistration(RenderTransformRegistrar):
    """Offer the bundled Workflows skill to controlling sessions only."""
    optional_reads = {"skill_registry", "_session_kind"}
    writes = {"skill_registry"}

    def __init__(self):
        self.registry = None

    def __call__(self, state):
        registry = getattr(state, "skill_registry", None)
        if registry is None or registry is self.registry:
            return state
        if worker_identity(state):
            existing = registry.get("Workflows")
            if existing is not None:
                registry.add_exclusions(existing.root)
        else:
            relative = Path("skills/workflows/SKILL.md")
            installed = Path(resolve_state_root()) / "plugin-configs" / PLUGIN_NAME / relative
            bundled = Path(__file__).resolve().parent.parent / relative
            package = bundled if bundled.is_file() else installed
            if package.is_file():
                registry.add_sources(SkillSource(package, kind=PLUGIN_NAME, scope="bundled",
                                                 precedence=-5000, native=True, always_list=True))
        self.registry = registry
        return state


class WorkflowNotices(InterruptCheck):
    """Owner-thread hook: adopt workflows after restart, deliver notices, remind workers."""
    check_name = "workflow_notices"
    cooldown = 0
    optional_reads = {"_session_kind", "_session_id", "_instance_id", "_agent_zoo_context", "session_launcher",
                      "_persistence_local_root", "session_cwd", "entries", "buffer_manager", "done"}
    writes = {"pending_interrupts", "buffer_manager"}
    init = {"pending_interrupts": list}

    def __init__(self, plugin: WorkflowsPlugin):
        self.plugin = plugin

    def __call__(self, state):
        if getattr(state, "buffer_manager", None) is not None:
            WorkflowBuffers(self.plugin)(state)  # Paused startups skip ordinary components.
        if worker_identity(state):
            self.remind_worker(state)
            return state
        controller = self.plugin.controller
        if getattr(state, "_instance_id", ""):
            try:
                controller.adopt(state)
            except Exception:
                LOG.exception("Workflow adoption failed")
        while controller.notices:
            state.pending_interrupts.append(controller.notices.popleft())
        return state

    def remind_worker(self, state):
        """A worker that ends its turn without finish_task gets one reminder, then needs attention."""
        if getattr(state, "done", False) or getattr(state, "shutdown", False):
            return
        entries = getattr(state, "entries", None) or []
        if not entries:
            return
        last = entries[-1]
        index = len(entries) - 1
        controller = self.plugin.controller
        if controller.worker_reminded_entry < 0:
            # Baseline at startup so a restored session is not reminded about an old reply.
            controller.worker_reminded_entry = index
            return
        if getattr(last, "role", None) != "assistant" or getattr(last, "tool_calls", None):
            return
        if index <= controller.worker_reminded_entry:
            return
        controller.worker_reminded_entry = index
        try:
            context = self.plugin.worker_context(state, refresh=True)
            store = WorkflowStore(context["workflow"])
            if context.get("state") in {"finished", "restarted"}:
                return
            reminders = int(context.get("reminders", 0))
            if reminders == 0:
                store.update_agent(context["id"], reminders=1)
                state.pending_interrupts.append(
                    "You ended your turn without calling finish_task. If your task is done, call finish_task now "
                    "with your summary and outputs. If it is not done, keep working.")
            else:
                store.update_agent(context["id"], reminders=reminders + 1, state="attention",
                                   problem="ended its turn again without calling finish_task; message it or use "
                                           "restart_agent")
        except Exception:
            LOG.exception("Workflow finish reminder failed")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_CONTROLLERS: dict[int, WorkflowController] = {}


def register_features(builder, *, session, config):
    state = getattr(session, "state", None)
    controller = _CONTROLLERS.get(id(session)) if session is not None else None
    if controller is None or controller.session is not session:
        controller = WorkflowController(config, session)
        if session is not None:
            _CONTROLLERS[id(session)] = controller
    controller.config = config
    plugin = WorkflowsPlugin(controller)
    if worker_identity(state) and builder.has("rlm"):
        # Workers do their own work: remove agent-facing RLM tools, keep compaction machinery.
        feature = builder._features["rlm"]
        builder.add(replace(feature, components=[c for c in feature.components if not isinstance(c, Tool)]))
    builder.add(Feature("workflows", components=[
        WorkflowBuffers(plugin), WorkflowNotices(plugin), *build_tools(plugin),
        Command("/workflow", "List, show, run, pause, rename, archive or change workflows", WorkflowCommand(plugin),
                raw=True,
                usage=USAGE, section="Workflows"),
    ], order=[RegisterSpecialBuffers, WorkflowBuffers, Command, WorkflowNotices, InterruptDelivery,
              ToolDispatchStart]))
    builder.add(Feature("workflow_guidance", components=[
        WorkflowSkillRegistration(), WorkflowSystemPrompt(plugin),
    ], order=[WorkflowSkillRegistration, WorkflowSystemPrompt, SystemPromptSkillList, MessageRenderer]))
