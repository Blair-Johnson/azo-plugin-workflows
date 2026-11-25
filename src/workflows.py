"""Named, persistent multi-agent workflows built from queues and agent groups.

A workflow definition (YAML) declares queues (``q_*``) and agent groups
(``grp_*``). A group with ``takes_from`` starts one agent per item taken from
that queue, or per batch when ``takes_from`` maps queues to batch sizes; a
group without it keeps ``count`` agents running from the same instructions. Agents send results to the queues in their group's ``sends_to``.
The scheduler runs inside the session that controls the workflow.

The first half of this module is a harness-free store; the Agent Zoo plugin
surface (tools, commands, buffers, scheduler) follows it.
"""
from __future__ import annotations

import contextlib
import difflib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agent_zoo.durability import (
    atomic_write_json,
    atomic_write_text,
    file_lock,
    fsync_dir,
    fsync_fd,
)

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
MAX_BATCH = 100

EXAMPLE_DEFINITION = """\
description: Summarize new sparse-attention papers   # one line, shown in /workflow list
folder: ~/work/papers            # working directory for agents (optional)
queues:
  q_papers: {}
  q_summaries: { notify_on_new: true }   # wake the controlling agent on new items
  q_topics: {}
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
  grp_survey:                    # one agent per batch: 5 summaries and 1 topic
    takes_from: { q_summaries: 5, q_topics: 1 }
    instructions: |
      Write a short survey of these papers on the topic.
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
        # takes_from: q_a takes one item; {q_a: 2, q_b: 1} takes a batch from each queue.
        raw_takes = settings.get("takes_from")
        if isinstance(raw_takes, dict):
            pairs = list(raw_takes.items())
            if not pairs:
                problems.append(f"{name}: takes_from needs at least one queue, e.g. {{q_items: 2}}")
        elif isinstance(raw_takes, list):
            problems.append(f"{name}: takes_from is one queue (takes_from: q_items) or a mapping of queues to "
                            "items per agent (takes_from: {q_a: 2, q_b: 1})")
            pairs = []
        else:
            pairs = [] if raw_takes is None else [(raw_takes, 1)]
        takes_from = {}
        for queue, size in pairs:
            queue = queue_ref(name, "takes_from", queue)
            if queue is None:
                continue
            if queue == TRASH:
                problems.append(f"{name}: cannot take items from {TRASH}")
            elif isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_BATCH:
                problems.append(f"{name}: takes_from {queue} must be a whole number of items from 1 to {MAX_BATCH}")
            else:
                takes_from[queue] = size
        takes_from = takes_from or None
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
    lines = yaml.safe_dump(value, default_flow_style=True, width=1 << 20, sort_keys=False).splitlines()
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
    atomic_write_text(path, text)


def write_json(path: Path, value) -> None:
    atomic_write_json(path, value, indent=2)


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
        with file_lock(self.dir / ".lock"):
            yield

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
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "log.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            fsync_fd(stream.fileno())

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
            fsync_dir(self.root)
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
                changes.append(f"takes_from {takes_text(before['takes_from']) or 'nothing'} → "
                               f"{takes_text(after['takes_from']) or 'nothing'}")
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
        taken = {q for g in new["agents"].values() for q in g["takes_from"] or {}}
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

    def _add_item_unlocked(
        self,
        queue: str,
        text: str,
        *,
        by: str,
        note: str | None = None,
        from_items: list[str] | None = None,
        allowed: set | None = None,
    ) -> dict:
        queue = check_queue_id(queue)
        if allowed is not None and queue not in allowed:
            raise WorkflowError(f"you can only send to: {', '.join(sorted(allowed))}")
        if not text.strip():
            raise WorkflowError("an item cannot be empty")
        self.require()
        number = self.next_number("item")
        item_id = f"item_{number}"
        at = now_iso()
        meta = {
            "id": item_id, "title": item_title(text), "queue": queue, "state": "waiting",
            "created_at": at, "created_by": by, "from_items": list(from_items or []), "taken_by": None,
            "notes": [{"text": note, "by": by, "at": at}] if note else [],
            "history": [{"at": at, "event": "added", "queue": queue, "by": by}],
        }
        atomic_write(self.item_paths(item_id)[0], text if text.endswith("\n") else text + "\n")
        self._save_item(meta)
        self.log("item_added", item=item_id, queue=queue, by=by)
        return meta

    def add_item(self, queue: str, text: str, *, by: str, note: str | None = None,
                 from_items: list[str] | None = None, allowed: set | None = None) -> dict:
        queue = check_queue_id(queue)
        if allowed is not None and queue not in allowed:
            raise WorkflowError(f"you can only send to: {', '.join(sorted(allowed))}")
        if not text.strip():
            raise WorkflowError("an item cannot be empty")
        with self.lock():
            return self._add_item_unlocked(
                queue, text, by=by, note=note, from_items=from_items, allowed=allowed
            )

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

    def take_batch(self, takes: dict[str, int], agent_id: str) -> list[dict]:
        """Lease the oldest ``size`` waiting items of each queue to ``agent_id``, or nothing
        if any queue is short. Caller holds the lock."""
        batch = []
        for queue, size in takes.items():
            waiting = [m for m in self.items(queue) if m["state"] == "waiting"][:size]
            if len(waiting) < size:
                return []
            batch += waiting
        at = now_iso()
        for meta in batch:
            meta.update(state="taken", taken_by=agent_id)
            meta["history"].append({"at": at, "event": "taken", "queue": meta["queue"], "by": agent_id})
            self._save_item(meta)
        return batch

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

    def new_agent(self, group: str, *, version: int, item_ids: list[str], sends_to: list[str],
                  folder: str, model: str | None) -> dict:
        """Allocate an agent record. Caller holds the lock."""
        number = self.next_number("agent", group)
        record = {
            "id": f"agt_{group_short(group)}_{number}", "group": group, "number": number,
            "version": version, "items": list(item_ids), "sends_to": sends_to, "folder": folder,
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
        """Record a worker's finish atomically under one lock: create outputs, mark inputs done."""
        with self.lock():
            record = self.agent(agent_id)
            if record["state"] in {"finished", "restarted"}:
                raise WorkflowError(f"{agent_id} has already {record['state']}")
            allowed = set(record.get("sends_to", [])) | {TRASH}
            for queue, text in outputs:
                check_queue_id(queue)
                if queue not in allowed:
                    raise WorkflowError(f"you can only send to: {', '.join(sorted(allowed))}")
                if not str(text or "").strip():
                    raise WorkflowError("an item cannot be empty")
            input_ids = agent_items(record)
            created = [
                self._add_item_unlocked(
                    queue, text, by=agent_id, from_items=input_ids, allowed=allowed
                )
                for queue, text in outputs
            ]
            for item_id in input_ids:
                self.finish_item(item_id, agent_id)
            record.update(state="finished", summary=summary, finished_at=now_iso(),
                          outputs=record.get("outputs", []) + [m["id"] for m in created],
                          problem=None)
            self.save_agent(record)
            self.log("agent_finished", agent=agent_id, items=input_ids,
                     outputs=[m["id"] for m in created], summary=summary)
        return {"agent": record, "created": created}


def and_list(parts: list[str]) -> str:
    return " and ".join(parts) if len(parts) < 3 else ", ".join(parts[:-1]) + " and " + parts[-1]


def takes_text(takes: dict | None) -> str:
    """``q_a`` for one item per agent, else e.g. ``2 from q_a and 1 from q_b``."""
    if not takes:
        return ""
    if list(takes.values()) == [1]:
        return next(iter(takes))
    return and_list([f"{size} from {queue}" for queue, size in takes.items()])


def takers(definition: dict, queue: str) -> list[str]:
    """Groups that take items from ``queue``."""
    return [g for g, s in definition["agents"].items() if queue in (s["takes_from"] or {})]


def agent_items(record: dict) -> list[str]:
    """Item IDs an agent took; records from before batches have a single ``item``."""
    if "items" in record:
        return list(record["items"] or [])
    return [record["item"]] if record.get("item") else []


def item_origins(meta: dict) -> list[str]:
    """Items whose agent made this item; older items have a single ``from_item``."""
    if "from_items" in meta:
        return list(meta["from_items"] or [])
    return [meta["from_item"]] if meta.get("from_item") else []


def batch_text(items: list[dict]) -> str:
    """e.g. ``item_1, item_2 from q_a and item_3 from q_b``."""
    by_queue: dict[str, list[str]] = {}
    for meta in items:
        by_queue.setdefault(meta["queue"], []).append(meta["id"])
    return and_list([f"{', '.join(ids)} from {queue}" for queue, ids in by_queue.items()])


def describe_group(group: dict) -> str:
    takes = group["takes_from"]
    if takes:
        source = f"takes {'from ' if list(takes.values()) == [1] else ''}{takes_text(takes)}"
    else:
        source = "standing"
    parts = [source]
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
            raise WorkflowError(f"every only applies to groups without takes_from; {target} starts agents as "
                                f"items arrive in {', '.join(group['takes_from'])}")
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
            rows.append(["`every`", "—", "not available: this group takes items from "
                         + ", ".join(f"`{q}`" for q in group["takes_from"])])
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
    if item_origins(meta):
        origin += f" while working on {', '.join(item_origins(meta))}"
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
        source = takes_text(settings["takes_from"]) or "standing"
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
             ", ".join(f"`{i}`" for i in agent_items(a)) or "—", f"def_{a['version']}",
             clip(a.get("summary") or a.get("problem") or "")] for a in shown]
    lines += md_table(["Agent", "State", "Items", "Version", "Summary"], rows) if rows else ["No agents yet."]
    if len(ended) > FINISHED_SHOWN:
        lines += ["", f"{len(ended) - FINISHED_SHOWN} older finished agents are not shown."]
    if settings:
        lines += ["", "## Instructions", "", *md_quote(settings["instructions"])]

    lines += ["", "**Commands**", ""]
    if settings:
        lines.append(f"- `/workflow set {store.name}:{group}` — count, model, schedule and folder")
        for queue in settings["takes_from"] or {}:
            lines.append(f"- `/workflow view {store.name}:{queue}` — the items it takes")
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
    taken_by = [f"`{g}`" for g in takers(definition, queue)]
    senders = [f"`{g}`" for g, s in definition["agents"].items() if queue in s["sends_to"]]
    if queue == TRASH:
        lines = ["Discarded items. Move one back to a queue to restore it."]
    else:
        lines = [f"**Filled by** {', '.join(senders) or 'nobody; items come from you or the agent'} · "
                 f"**Taken by** {', '.join(taken_by) or 'no group, so items stay here'}"]
    if not known:
        lines += ["", "⚠ This queue is not in the current definition; its items wait until it is added back."]

    def added_by(item):
        origins = ", ".join(f"`{i}`" for i in item_origins(item))
        return shorten_ids(item["created_by"]) + (f" (from {origins})" if origins else "")

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
    if taken_by:
        lines.append(f"- `/workflow view {store.name}:{taken_by[0].strip('`')}` — the group that takes these items")
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
    senders = [g for g, s in definition["agents"].items() if queue in s["sends_to"]]
    lines = [f"{queue} in {store.name}: {counts['waiting']} waiting, {counts['taken']} being worked on, "
             f"{counts['done']} done."]
    lines.append(f"Taken from by: {', '.join(takers(definition, queue)) or 'no group'}. Sent to by: {', '.join(senders) or 'no group'}.")
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
            item = f" on {', '.join(agent_items(agent))}" if agent_items(agent) else ""
            note = f": {agent['summary']}" if agent.get("summary") else (
                f": {agent['problem']}" if agent.get("problem") else "")
            lines.append(f"  {agent['id']}{item} (def_{agent['version']}){note}")
    if settings:
        lines += ["", "Instructions", settings["instructions"].rstrip("\n")]
    return "\n".join(lines) + "\n"


def render_agent(store: WorkflowStore, agent_id: str, transcript=None) -> str:
    agent = store.agent(agent_id)
    lines = [f"{agent['id']} in {store.name} ({agent['group']}, def_{agent['version']}): {agent['state']}."]
    if agent_items(agent):
        lines.append("Items: " + ", ".join(f"{i} (wf:{store.name}:{i})" for i in agent_items(agent)))
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
    if item_origins(meta):
        origin += f" while working on {', '.join(item_origins(meta))}"
    lines.append(origin + ".")
    made = [i["id"] for i in store.items() if meta["id"] in item_origins(i)]
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
# Agent Zoo integration (100% pure Sans-I/O + commit-class WorkflowStoreDriver)
# ===========================================================================

import collections
import inspect
import logging
import shlex
import uuid
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

from agent_utils.core.events import (
    Event,
    IoCompleted,
    IoFailed,
    IoLost,
    IoReattached,
    RenderRequested,
    SessionRestored,
    TimerFired,
    ToolCallFinished,
    TurnSettled,
)
from agent_utils.core.io import IoRequest, OpId, StartTimer
from agent_utils.core.pipeline import EmitFn, Feature
from agent_utils.core.state import MODEL_RENDER_CHANNEL, DurableSpec, State, ToolCall
from agent_utils.core.tool import Tool
from agent_utils.features.buffers import ReadonlyBufferView
from agent_utils.features.core import (
    MessageRenderer,
    RenderPlanInitializer,
    RenderTransformRegistrar,
    SchemaPublisher,
)
from agent_utils.features.interrupts import InterruptCheck, InterruptQueued
from agent_utils.features.skills import SkillSource
from agent_utils.storage.repository import SessionRepository
from agent_zoo.config.loader import CONFIG_PATH, load_layered_config
from agent_zoo.core.feature_state import ZooState, session_state
from agent_zoo.durability import QueuedWorkerDriver
from agent_zoo.events import (
    AdoptSession,
    ConfigureSessionLifecycle,
    DiscoverSessions,
    DiscoverSessionsResult,
    EmitNotice,
    ExternalSessionExited,
    ExternalSessionReady,
    LiveSessionInfo,
    ObserveSession,
    PacingCheck,
    ProjectionRequested,
    PublishInspector,
    SendSessionData,
    SendSessionInput,
    SessionArtifact,
    SessionDataReceived,
    SessionDeliveryResult,
    SessionObservation,
    SessionReady,
    SessionSource,
    SessionSpawnResult,
    SessionSpec,
    SpawnSession,
    StopSession,
    SuspendRequested,
)
from agent_zoo.features.commands import Command, CommandError, CommandResult
from agent_zoo.features.special_buffers import RegisterSpecialBuffers
from agent_zoo.features.system_prompt import SystemPromptSkillList
from agent_zoo.features.turn import LLMCaller
from agent_zoo.llm import enabled_llm_models, require_model_enabled
from agent_zoo.storage.projects import resolve_state_root
from agent_zoo.storage.store import resolve_local_root
from agent_zoo.transport.models import CommandNotice, InspectorDocument

LOG = logging.getLogger(__name__)

WORKER_KIND = re.compile(r"wf:(?P<workflow>[a-z][a-z0-9_]{0,63}):(?P<agent>agt_[a-z0-9_]+_\d+)\Z")
TICK_SECONDS = 2.0
READY_TIMEOUT_SECONDS = 120
MAX_QUICK_RESUMES = 3
WORKER_TOOLS_GROUP = "Workflows"

DEFAULT_EMPTY_INDEX = (
    "Workflows. Open one with wf:<name>, e.g. wf:miner.\n"
    "There are no workflows yet. Create one with create_workflow.\n"
)

RECOVERY_MESSAGE = (
    "Your session was restarted from a checkpoint after an interruption. "
    "Inspect saved results and existing processes/artifacts before continuing. "
    "Do not repeat a command just because its output is missing; if its outcome is "
    "uncertain, report that uncertainty instead of retrying it. "
    "Continue from confirmed progress and call finish_task when done."
)


# ---------------------------------------------------------------------------
# Durable & ephemeral per-session state
# ---------------------------------------------------------------------------

@dataclass
class WorkflowDurableState:
    """Durable per-session workflow control state stored under ``state.features['workflows']``."""

    managed: dict[str, dict[str, Any]] = field(default_factory=dict)


WORKFLOWS_DURABLE_SPEC = DurableSpec(
    name="workflows",
    cls=WorkflowDurableState,
    version=1,
    on_fork="reset",
)


@dataclass
class _WorkflowTask:
    """In-flight Sans-I/O generator task driven via ``state.io.submit``."""

    kind: str  # "command" | "command_doc" | "tick" | "event" | "startup" | "worker_remind"
    gen: Any
    workflow_name: str = ""
    hold_readiness: str | None = None


@dataclass
class WorkflowRuntimeState:
    """Strictly ephemeral per-session runtime state and cached snapshots (never serialized)."""

    cached_names: tuple[str, ...] = ()
    cached_views: dict[str, str] = field(default_factory=dict)
    cached_worker_context: dict[str, Any] | None = None
    pending_tasks: dict[OpId, _WorkflowTask] = field(default_factory=dict)
    tick_timer_op: OpId | None = None
    tick_active: bool = False
    adopted_instance: str = ""
    instances_by_id: dict[str, tuple[str, str]] = field(default_factory=dict)
    in_flight_agents: set[tuple[str, str]] = field(default_factory=set)
    notices: collections.deque = field(default_factory=collections.deque)
    notified: set = field(default_factory=set)
    errors: dict[str, str] = field(default_factory=dict)
    worker_reminded_entry: int = -1
    me: dict[str, str] | None = None
    launch: dict[str, Any] | None = None


def _wf_durable(state: Any) -> WorkflowDurableState:
    if state is None:
        return WorkflowDurableState()
    durable_fn = getattr(state, "durable", None)
    if callable(durable_fn):
        return durable_fn(WORKFLOWS_DURABLE_SPEC)
    existing = getattr(state, "_wf_durable_fallback", None)
    if not isinstance(existing, WorkflowDurableState):
        existing = WorkflowDurableState()
        try:
            setattr(state, "_wf_durable_fallback", existing)
        except Exception:
            pass
    return existing


def _wf_runtime(state: Any) -> WorkflowRuntimeState:
    if state is None:
        return WorkflowRuntimeState()
    runtime_fn = getattr(state, "runtime", None)
    if callable(runtime_fn):
        return runtime_fn("workflows", WorkflowRuntimeState)
    existing = getattr(state, "_wf_runtime_fallback", None)
    if not isinstance(existing, WorkflowRuntimeState):
        existing = WorkflowRuntimeState()
        try:
            setattr(state, "_wf_runtime_fallback", existing)
        except Exception:
            pass
    return existing


def _session_meta(state: Any):
    meta = session_state(state)
    if state is not None and not isinstance(state, ZooState):
        d = getattr(state, "__dict__", None)
        if isinstance(d, dict):
            for attr, field_name in (
                ("_session_id", "session_id"),
                ("_instance_id", "instance_id"),
                ("_project_name", "project_name"),
                ("_session_kind", "kind"),
            ):
                if attr in d and d[attr]:
                    setattr(meta, field_name, d[attr])
    return meta


def get_session_cwd(state) -> str:
    """Resolve the active working directory purely from ``state.cwd`` (with legacy fallbacks)."""
    cwd = (
        getattr(state, "cwd", "")
        or getattr(state, "session_cwd", "")
        or getattr(state, "terminal_cwd", "")
    )
    return str(cwd) if cwd else ""


def worker_identity(state):
    """Workers are identified by their durable session kind, never by environment."""
    return WORKER_KIND.fullmatch(str(_session_meta(state).kind or ""))


def _apply_snapshot(state: Any, snapshot: dict[str, Any] | None) -> None:
    if not isinstance(snapshot, dict):
        return
    rt = _wf_runtime(state)
    if "names" in snapshot:
        rt.cached_names = tuple(snapshot["names"])
    if isinstance(snapshot.get("views"), dict):
        rt.cached_views = dict(snapshot["views"])
    if "worker_context" in snapshot and snapshot["worker_context"] is not None:
        rt.cached_worker_context = dict(snapshot["worker_context"])


# ---------------------------------------------------------------------------
# Host/Driver-only helpers (NEVER called from Sans-I/O pipeline code)
# ---------------------------------------------------------------------------

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
    """Saved config layers only: called ONCE at controller build/init time, never in pipeline steps."""
    path = ((config or {}).get("_runtime_launch", {}) or {}).get("config_path")
    try:
        if path and Path(path).resolve() == CONFIG_PATH.resolve():
            path = None
        return load_layered_config(state_root=resolve_state_root(), config_path=path or None)
    except Exception:
        LOG.warning("Cannot read saved Agent Zoo model settings", exc_info=True)
        return None


def _default_model_from_sources(saved_cfg: dict | None, config: dict | None) -> str:
    for source in (saved_cfg, config):
        model = str(((source or {}).get("llm", {}) or {}).get("default") or "")
        if model:
            return model
    return ""


def saved_default_model(config) -> str:
    return _default_model_from_sources(saved_model_config(config), config)


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


def _pick_live_row(disc: Any) -> LiveSessionInfo | None:
    """Return the freshest live, non-terminal registry row with a websocket URL."""
    rows = disc.rows if isinstance(disc, DiscoverSessionsResult) else ()
    live = [
        row for row in rows
        if isinstance(row, LiveSessionInfo)
        and not row.is_terminal
        and row.process_state != "dead"
        and row.websocket_url
    ]
    if not live:
        return None
    return max(live, key=lambda r: (r.updated_at, r.started_at, r.instance_id))


# ---------------------------------------------------------------------------
# Transcript rendering (host/driver-only)
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
# Instructions (driver-side rendering)
# ---------------------------------------------------------------------------

def render_instructions(store: WorkflowStore, group: dict, agent: dict, items: list[dict],
                        viewer_session_id: str = "") -> str:
    text = group["instructions"]
    item_text = "\n\n".join(item_block(store, i) for i in items) or "(this agent did not take an item)"
    if items and "{{item}}" not in text:
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
# Commit-Class WorkflowStoreRequest & WorkflowStoreDriver
# ---------------------------------------------------------------------------

@dataclass
class WorkflowStoreRequest(IoRequest):
    """Typed commit-class request executed exclusively by ``WorkflowStoreDriver``."""

    op: str = ""
    workflow_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)


class WorkflowStoreDriver(QueuedWorkerDriver):
    """Commit-class Sans-I/O driver owning ALL ``WorkflowStore`` disk/lock/checkpoint I/O."""

    request_types = (WorkflowStoreRequest,)
    durability = "commit"
    worker_name = "agent-zoo-workflow-store"

    def __init__(self, *, root: Path | None = None) -> None:
        super().__init__()
        self._root = root

    def register_sim(self, sim, *, delay: float = 0.0):
        """Register deterministic simulation handler on a ``SimulationRuntime``."""
        from agent_utils.runtime.simulation import SimOutcome

        def _handler(_op_id, req, _sim):
            try:
                return SimOutcome(value=self.execute(req), delay=delay)
            except Exception as exc:
                return SimOutcome(error=exc, delay=delay)

        sim.register_handler(WorkflowStoreRequest, _handler)
        return self

    def _store(self, name: str) -> WorkflowStore:
        return WorkflowStore(name, self._root)

    def _build_snapshot(
        self,
        *,
        viewer_session_id: str = "",
        default_model: str = "",
        worker_workflow: str = "",
        worker_agent: str = "",
        include_transcripts: bool = False,
    ) -> dict[str, Any]:
        names = list_workflow_names(self._root)
        views: dict[str, str] = {"wf:index": render_index(self._root, viewer_session_id)}
        for wf_name in names:
            store = self._store(wf_name)
            try:
                control = store.control()
                views[f"wf:{wf_name}"] = render_overview(store, viewer_session_id, default_model)
                views[f"wf:{wf_name}:definition"] = store.definition_text()
                if control.get("state") == "archived" and wf_name != worker_workflow:
                    continue
                views[f"wf:{wf_name}:history"] = render_history(store)
                views[f"wf:{wf_name}:log"] = render_log(store)
                for ver in range(1, store.version + 1):
                    try:
                        views[f"wf:{wf_name}:def_{ver}"] = store.definition_text(ver)
                    except WorkflowError:
                        pass
                defn = store.definition()
                items = store.items()
                agents = store.agents()
                queue_ids = set(defn["queues"]) | {TRASH} | {i["queue"] for i in items}
                for q in queue_ids:
                    views[f"wf:{wf_name}:{q}"] = render_queue(store, q)
                group_ids = set(defn["agents"]) | {a["group"] for a in agents}
                for g in group_ids:
                    views[f"wf:{wf_name}:{g}"] = render_group(store, g, default_model)
                for item in items:
                    views[f"wf:{wf_name}:{item['id']}"] = render_item(store, item["id"])
                for agent in agents:
                    transcript_fn = None
                    if include_transcripts:
                        def _transcript(sid, proj=agent.get("project") or "default"):
                            return saved_transcript(proj, sid)
                        transcript_fn = _transcript
                    views[f"wf:{wf_name}:{agent['id']}"] = render_agent(store, agent["id"], transcript_fn)
            except Exception:
                LOG.debug("Skipping damaged workflow %s in snapshot", wf_name, exc_info=True)
        worker_ctx = None
        if worker_workflow and worker_agent:
            store = self._store(worker_workflow)
            try:
                worker_ctx = store.agent(worker_agent)
            except WorkflowError:
                worker_ctx = {"id": worker_agent, "group": "", "sends_to": [], "items": [], "state": "unknown"}
            worker_ctx["workflow"] = worker_workflow
        return {"names": tuple(names), "views": views, "worker_context": worker_ctx}

    @staticmethod
    def _taker_hint(store: WorkflowStore, queue: str) -> str:
        if queue == TRASH:
            return " Move them back out of q_trash to restore them."
        groups = takers(store.definition(), queue)
        if not groups:
            return f" No agent group takes from {queue}."
        if store.control().get("state") != "running":
            return f" {', '.join(groups)} will take them when {store.name} runs."
        return f" {', '.join(groups)} will take them."

    @staticmethod
    def _idle_reason(store: WorkflowStore, group: str | None) -> str:
        definition = store.definition()
        reasons = []
        for gname, settings in definition["agents"].items():
            if group and gname != group:
                continue
            if settings["count"] == 0:
                reasons.append(f"{gname} has count 0")
            else:
                for queue, size in (settings["takes_from"] or {}).items():
                    waiting = sum(i["state"] == "waiting" for i in store.items(queue))
                    if size == 1 and not waiting:
                        reasons.append(f"{queue} is empty, so {gname} has nothing to take")
                    elif waiting < size:
                        reasons.append(
                            f"{gname} takes {size} items from {queue} at a time and "
                            f"{queue} has {waiting} waiting"
                        )
        return "; ".join(reasons) + "." if reasons else "all groups are at their count."

    @staticmethod
    def _take_control_locked(store: WorkflowStore, me: dict, launch: dict) -> tuple[dict | None, int, int]:
        with store.lock():
            store.require()
            control = store.control()
            if control.get("state") == "archived":
                raise WorkflowError(
                    f"{store.name} is archived; restore it with /workflow unarchive {store.name} first"
                )
            previous = control.get("controller")
            epoch = int(control.get("epoch", 0)) + 1
            store.update_control(controller=dict(me, project=launch["project"]), epoch=epoch)
            if not control.get("default_folder") and launch.get("cwd"):
                store.update_control(default_folder=launch["cwd"])
            store.log("control_taken", by=me["session_id"], epoch=epoch)
        watermark = store.counters().get("item", 0)
        return previous, epoch, watermark

    def execute(self, req: WorkflowStoreRequest) -> Any:
        op = req.op
        p = req.params
        viewer = str(p.get("viewer_session_id", "") or "")
        default_model = str(p.get("default_model", "") or "")

        if op == "snapshot":
            return self._build_snapshot(
                viewer_session_id=viewer,
                default_model=default_model,
                worker_workflow=str(p.get("worker_workflow", "") or ""),
                worker_agent=str(p.get("worker_agent", "") or ""),
            )

        if op == "create_workflow":
            store = self._store(req.workflow_name)
            text = str(p["text"])
            check_definition_models(parse_definition(split_version(text)[1]), p.get("config"))
            definition = store.create(text, by=str(p["by"]))
            cwd = str(p.get("cwd") or "") or os.getcwd()
            store.update_control(default_folder=cwd)
            groups = "\n".join(f"  {g}: {describe_group(s)}" for g, s in definition["agents"].items())
            queues = ", ".join(definition["queues"]) or "none"
            msg = (
                f"Created workflow {store.name} (def_1). Nothing is running yet.\n"
                f"Queues: {queues}\nAgent groups:\n{groups}\n"
                f"Add starting items with add_items, then ask the user before running it with "
                f'command_workflow(workflow_name="{store.name}", command="run"). The user can also type '
                f"/workflow run {store.name}. See wf:{store.name}."
            )
            return {"text": msg, "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model)}

        if op == "update_workflow":
            store = self._store(req.workflow_name).require()
            reason = str(p.get("reason") or "").strip()
            if not reason:
                raise WorkflowError("reason is required: one sentence saying why the definition changed")
            text = str(p["text"])
            check_definition_models(parse_definition(split_version(text)[1]), p.get("config"))
            preview_only = bool(p.get("preview_only", False))
            result = store.update(text, reason=reason, by=str(p["by"]), preview_only=preview_only)
            effects = "\n".join(f"- {e}" for e in result["effects"])
            if not result["changed"]:
                msg = f"The definition is unchanged (still def_{result['version']})."
            elif preview_only:
                msg = (
                    f"Preview only; nothing changed. Applying it would create def_{result['from_version'] + 1}."
                    f"\n\nDiff:\n{result['diff']}\nEffects:\n{effects}"
                )
            else:
                msg = (
                    f"Updated {store.name} from def_{result['from_version']} to def_{result['version']}."
                    f"\n\nDiff:\n{result['diff']}\nEffects:\n{effects}"
                )
            return {"text": msg, "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model)}

        if op == "add_items":
            store = self._store(req.workflow_name).require()
            queue = str(p["queue"])
            texts = list(p["texts"])
            note = p.get("note")
            worker_agent = str(p.get("worker_agent") or "")
            if worker_agent:
                context = store.agent(worker_agent)
                allowed = set(context.get("sends_to") or []) | {TRASH}
                created = [
                    store.add_item(
                        queue, text, by=context["id"], note=note,
                        from_items=agent_items(context), allowed=allowed,
                    )
                    for text in texts
                ]
                ids = [m["id"] for m in created]
                record = store.agent(context["id"])
                store.update_agent(context["id"], outputs=record.get("outputs", []) + ids)
                msg = f"Added {', '.join(ids)} to {queue}. Keep working, and call finish_task when your task is done."
                snap = self._build_snapshot(
                    viewer_session_id=viewer, default_model=default_model,
                    worker_workflow=store.name, worker_agent=worker_agent,
                )
            else:
                known = set(store.definition()["queues"]) | {TRASH}
                check_queue_id(queue, known)
                created = [store.add_item(queue, text, by="user", note=note) for text in texts]
                msg = f"Added {', '.join(m['id'] for m in created)} to {queue}." + self._taker_hint(store, queue)
                snap = self._build_snapshot(viewer_session_id=viewer, default_model=default_model)
            return {"text": msg, "snapshot": snap}

        if op == "move_items":
            store = self._store(req.workflow_name).require()
            ids = list(p["item_ids"])
            to_queue = str(p["to_queue"])
            note = p.get("note")
            worker_agent = str(p.get("worker_agent") or "")
            if worker_agent:
                context = store.agent(worker_agent)
                allowed = set(context.get("sends_to") or []) | {TRASH}
                moved = store.move_items(ids, to_queue, by=context["id"], note=note, allowed=allowed)
                msg = f"Moved {', '.join(m['id'] for m in moved)} to {to_queue}."
                snap = self._build_snapshot(
                    viewer_session_id=viewer, default_model=default_model,
                    worker_workflow=store.name, worker_agent=worker_agent,
                )
            else:
                known = set(store.definition()["queues"]) | {TRASH} | {i["queue"] for i in store.items()}
                check_queue_id(to_queue, known)
                moved = store.move_items(ids, to_queue, by="user", note=note)
                msg = f"Moved {', '.join(m['id'] for m in moved)} to {to_queue}." + self._taker_hint(store, to_queue)
                snap = self._build_snapshot(viewer_session_id=viewer, default_model=default_model)
            return {"text": msg, "snapshot": snap}

        if op == "finish_agent":
            store = self._store(req.workflow_name).require()
            agent_id = str(p["agent_id"])
            summary = str(p.get("summary") or "")
            pairs = list(p.get("outputs") or [])
            result = store.finish_agent(agent_id, summary=summary, outputs=pairs)
            created = ", ".join(f"{m['id']} in {m['queue']}" for m in result["created"])
            msg = "Task finished" + (f"; created {created}" if created else "") + ". This session is ending."
            snap = self._build_snapshot(
                viewer_session_id=viewer, default_model=default_model,
                worker_workflow=store.name, worker_agent=agent_id,
            )
            return {"text": msg, "result": result, "snapshot": snap}

        if op == "run_prepare":
            store = self._store(req.workflow_name).require()
            definition = store.definition()
            group = p.get("group")
            if group is not None:
                check_group_id(group, definition["agents"])
            me = dict(p["me"])
            launch = dict(p["launch"])
            previous, epoch, watermark = self._take_control_locked(store, me, launch)
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
                store.log("run", group=group, by=me["session_id"])
            paused_agents = [a for a in store.agents(group) if a["state"] == "paused"]
            return {
                "previous": previous,
                "epoch": epoch,
                "watermark": watermark,
                "version": store.version,
                "paused_agents": paused_agents,
                "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
            }

        if op == "idle_reason":
            store = self._store(req.workflow_name).require()
            return self._idle_reason(store, p.get("group"))

        if op == "pause_prepare":
            store = self._store(req.workflow_name).require()
            definition = store.definition()
            group = p.get("group")
            if group is not None:
                check_group_id(group, definition["agents"])
            with store.lock():
                control = store.control()
                if control.get("state") == "archived":
                    raise WorkflowError(f"{store.name} is archived")
                if group is None:
                    store.update_control(state="paused")
                else:
                    paused = set(control.get("paused_groups") or []) | {group}
                    store.update_control(paused_groups=sorted(paused))
                store.log("pause", group=group, by=str(p["by"]))
            return {
                "agents": store.agents(group),
                "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
            }

        if op == "update_agent":
            store = self._store(req.workflow_name)
            updated = store.update_agent(str(p["agent_id"]), **dict(p["changes"]))
            return {
                "agent": updated,
                "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
            }

        if op == "read_agent":
            store = self._store(req.workflow_name)
            return {"agent": store.agent(str(p["agent_id"])), "control": store.control()}

        if op == "archive_check":
            store = self._store(req.workflow_name).require()
            if store.control().get("state") == "archived":
                raise WorkflowError(f"{store.name} is already archived")
            return {"ok": True}

        if op == "archive_list_agents":
            store = self._store(req.workflow_name).require()
            return [a for a in store.agents() if a["state"] in ACTIVE_STATES | {"attention"}]

        if op == "archive_return_agent":
            store = self._store(req.workflow_name)
            agent = dict(p["agent"])
            with store.lock():
                for item_id in agent_items(agent):
                    store.return_item(item_id, agent["id"], by="archive")
            store.update_agent(agent["id"], state="stopped", problem="workflow archived")
            return {"ok": True}

        if op == "archive_finalize":
            store = self._store(req.workflow_name)
            with store.lock():
                store.update_control(state="archived", controller=None)
                store.log("archived", by=str(p["by"]))
            return {"snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model)}

        if op == "unarchive":
            store = self._store(req.workflow_name).require()
            with store.lock():
                if store.control().get("state") != "archived":
                    raise WorkflowError(f"{store.name} is not archived")
                store.update_control(state="paused", paused_groups=[])
                store.log("unarchived", by=str(p["by"]))
            return {
                "text": f"Restored {store.name}. It is paused; start it with /workflow run {store.name}.",
                "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
            }

        if op == "rename":
            store = self._store(req.workflow_name).require()
            new_name = str(p["new_name"])
            if self._store(new_name).exists():
                raise WorkflowError(f'a workflow named "{new_name}" already exists; choose another name')
            renamed = store.rename(new_name, by=str(p["by"]))
            return {
                "new_name": renamed.name,
                "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
            }

        if op == "set_count":
            store = self._store(req.workflow_name).require()
            result = store.set_count(str(p["agent_group"]), int(p["count"]), by=str(p["by"]))
            return {
                "result": result,
                "control": store.control(),
                "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
            }

        if op == "set_setting":
            store = self._store(req.workflow_name).require()
            cfg = p.get("config")
            check_fn = (lambda prof: check_model(cfg, prof)) if cfg is not None else None
            change = prepare_setting(
                store, p["kind"], p.get("target"), p["setting"], p["raw"],
                check_model=check_fn, default_model=default_model,
            )
            if change["unchanged"]:
                return {
                    "unchanged": True,
                    "change": change,
                    "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
                }
            result = store.set_setting(
                change["section"], change["name"], change["key"], change["value"],
                reason=change["reason"], by="user",
            )
            definition = store.definition()
            active = [a for a in store.agents() if a["state"] in ACTIVE_STATES]
            return {
                "unchanged": False,
                "change": change,
                "result": result,
                "definition": definition,
                "active": active,
                "control": store.control(),
                "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
            }

        if op == "render_command_doc":
            action = str(p["action"])
            if action in {"list", "archived"}:
                archived = action == "archived"
                subtitle, body = render_list(self._root, viewer, archived=archived)
                title = "Archived workflows" if archived else "Workflows"
                return {
                    "title": title, "subtitle": subtitle, "body": body,
                    "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
                }
            store = self._store(req.workflow_name).require()
            kind, target = p["kind"], p.get("target")
            if action == "show":
                if kind == "workflow":
                    subtitle, body = render_overview_md(store, viewer, default_model)
                elif kind == "group":
                    subtitle, body = render_group_md(store, target, default_model)
                else:
                    subtitle, body = render_queue_md(store, target)
                title = target_label(store, kind, target)
            elif action == "settings":
                body = render_settings(store, kind, target, default_model)
                title = f"Settings: {target_label(store, kind, target)}"
                subtitle = ""
            else:
                raise WorkflowError(f"unknown document action {action!r}")
            return {
                "title": title, "subtitle": subtitle, "body": body,
                "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
            }

        if op == "tick_read":
            store = self._store(req.workflow_name)
            if not store.exists():
                return {"exists": False}
            return {
                "exists": True,
                "control": store.control(),
                "definition": store.definition(),
                "agents": store.agents(),
            }

        if op == "report_notices":
            store = self._store(req.workflow_name)
            if not store.exists():
                return {"attention": [], "new_items": [], "watermark": p.get("watermark", 0)}
            definition = store.definition()
            attention = [
                {
                    "text": (
                        f"Workflow {store.name}: {a['id']} needs attention: {a.get('problem')}. "
                        f"See wf:{store.name}:{a['id']}."
                    ),
                    "key": (store.name, a["id"], a.get("problem")),
                }
                for a in store.agents()
                if a["state"] == "attention"
            ]
            watermark = int(p.get("watermark", 0))
            latest = store.counters().get("item", 0)
            new_items: list[str] = []
            if latest > watermark:
                watched = {q for q, s in definition["queues"].items() if s["notify_on_new"]}
                if watched:
                    fresh = [i for i in store.items() if int(i["id"][5:]) > watermark and i["queue"] in watched]
                    by_queue: dict[str, list[str]] = collections.defaultdict(list)
                    for item in fresh:
                        by_queue[item["queue"]].append(f"{item['id']} ({item['title']})")
                    for queue, entries in by_queue.items():
                        new_items.append(
                            f"Workflow {store.name}: new items in {queue}: {', '.join(entries)}. "
                            f"See wf:{store.name}:{queue}."
                        )
            return {
                "attention": attention,
                "new_items": new_items,
                "watermark": max(watermark, latest),
                "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
            }

        if op == "plan_start_agents":
            store = self._store(req.workflow_name)
            expected_epoch = p.get("epoch")
            expected_iid = p.get("instance_id")
            default_cwd = str(p.get("cwd") or "") or os.getcwd()
            planned = []
            with store.lock():
                control = store.control()
                if (
                    expected_epoch is None
                    or control.get("epoch") != expected_epoch
                    or (control.get("controller") or {}).get("instance_id") != expected_iid
                    or control.get("state") != "running"
                ):
                    return {"planned": [], "definition": store.definition()}
                definition = store.definition()
                paused_groups = set(control.get("paused_groups") or [])
                last_started = dict(control.get("last_started") or {})
                for group, settings in definition["agents"].items():
                    if group in paused_groups or settings["count"] <= 0:
                        continue
                    active = sum(a["state"] in ACTIVE_STATES for a in store.agents(group))
                    room = settings["count"] - active
                    while room > 0:
                        items = []
                        if settings["takes_from"]:
                            number = store.counters().get("agents", {}).get(group, 0) + 1
                            provisional = f"agt_{group_short(group)}_{number}"
                            items = store.take_batch(settings["takes_from"], provisional)
                            if not items:
                                break
                        elif settings["every"]:
                            last = float(last_started.get(group) or 0)
                            if time.time() - last < settings["every"]:
                                break
                        folder_template = (
                            settings["folder"] or definition["folder"]
                            or control.get("default_folder") or default_cwd
                        )
                        agent = store.new_agent(
                            group, version=store.version, item_ids=[i["id"] for i in items],
                            sends_to=list(settings["sends_to"]), folder="", model=None,
                        )
                        agent["folder"] = render_folder(store, folder_template, agent)
                        store.save_agent(agent)
                        last_started[group] = time.time()
                        planned.append({"agent": agent, "settings": settings, "items": items})
                        room -= 1
                store.update_control(last_started=last_started)
            return {"planned": planned, "definition": definition}

        if op == "prepare_spawn":
            store = self._store(req.workflow_name)
            agent_id = str(p["agent_id"])
            if p.get("model"):
                store.update_agent(agent_id, model=p["model"])
            agent = store.agent(agent_id)
            message = str(p.get("message") or "")
            if p.get("render_instructions"):
                message = render_instructions(
                    store, dict(p["settings"]), agent, list(p.get("items") or []),
                    viewer_session_id=viewer,
                )
            folder = agent.get("folder") or ""
            if folder:
                Path(folder).mkdir(parents=True, exist_ok=True)
            session_id = agent.get("session_id") or str(p["session_id"])
            instance_id = str(p["instance_id"])
            agent = store.update_agent(
                agent_id,
                session_id=session_id,
                instance_id=instance_id,
                state="starting",
                project=str(p["project"]),
                launch_started=time.time(),
                websocket_url=None,
                process_identity=None,
                release="resume" if p.get("is_resume") else None,
                recovery_action=p.get("recovery_action"),
            )
            return {
                "agent": agent,
                "message": message,
                "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
            }

        if op == "log_agent_started":
            store = self._store(req.workflow_name)
            store.log("agent_started", agent=str(p["agent_id"]), items=list(p.get("items") or []))
            return {"snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model)}

        if op == "launch_failed":
            store = self._store(req.workflow_name)
            agent = dict(p["agent"])
            items = list(p.get("items") or [])
            error_text = str(p["error"])
            group = agent["group"]
            with store.lock():
                for item in items:
                    store.return_item(item["id"], agent["id"], by="scheduler")
                control = store.control()
                paused = set(control.get("paused_groups") or []) | {group}
                store.update_control(paused_groups=sorted(paused))
            store.update_agent(agent["id"], state="stopped", problem=f"could not start: {error_text}")
            store.log("agent_start_failed", agent=agent["id"], error=error_text)
            return {"snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model)}

        if op == "claim_resume":
            store = self._store(req.workflow_name)
            agent = dict(p["agent"])
            expected_epoch = p.get("epoch")
            with store.lock():
                current = store.agent(agent["id"])
                if expected_epoch is not None and int(store.control().get("epoch", 0)) != expected_epoch:
                    return {"status": "problem", "problem": f"{store.name} is now controlled by another session"}
                if current.get("instance_id") != agent.get("instance_id") or current["state"] not in {
                    "starting", "running", "paused", "attention"
                }:
                    return {"status": "skip"}
                current["state"] = "starting"
                store.save_agent(current)
            project = agent.get("project") or str(p["project"])
            try:
                saved = saved_checkpoint(project, agent["session_id"])
            except FileNotFoundError:
                return {"status": "problem", "problem": f"{agent['id']} has no saved checkpoint to resume from"}
            except Exception as exc:
                return {"status": "problem", "problem": f"{agent['id']}'s checkpoint could not be read: {exc}"}
            action, reason = checkpoint_recovery(saved.document)
            if action in {"uncertain", "done"}:
                return {
                    "status": "problem",
                    "problem": (
                        f"{agent['id']} checkpoint {saved.commit_id}: {reason}. "
                        "External work may already have happened; inspect its transcript and artifacts before retrying"
                    ),
                }
            return {"status": "ready", "commit_id": saved.commit_id, "action": action, "agent": current}

        if op == "mark_attention":
            store = self._store(req.workflow_name)
            agent_id = str(p["agent_id"])
            problem = str(p["problem"])
            store.update_agent(agent_id, state="attention", problem=problem)
            store.log("agent_needs_attention", agent=agent_id, problem=problem)
            return {"snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model)}

        if op == "release_commit":
            store = self._store(req.workflow_name)
            agent_id = str(p["agent_id"])
            expected_iid = p.get("instance_id")
            with store.lock():
                current = store.agent(agent_id)
                if current["state"] in ACTIVE_STATES and current.get("instance_id") == expected_iid:
                    current.update(state="running", release=None)
                    store.save_agent(current)
            return {
                "agent": store.agent(agent_id),
                "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
            }

        if op == "restart_check":
            store = self._store(req.workflow_name).require()
            agent = store.agent(check_agent_id(p["agent_id"]))
            if agent["state"] in {"finished", "restarted"}:
                raise WorkflowError(f"{agent['id']} has already {agent['state']}")
            return {"agent": agent}

        if op == "restart_finalize":
            store = self._store(req.workflow_name).require()
            agent_id = str(p["agent_id"])
            item_ids = list(p.get("item_ids") or [])
            with store.lock():
                for item_id in item_ids:
                    store.return_item(item_id, agent_id, by="restart_agent")
            store.update_agent(agent_id, state="restarted", problem=None)
            store.log("agent_restarted", agent=agent_id)
            first_queue = store.item(item_ids[0])["queue"] if len(item_ids) == 1 else ""
            return {
                "first_queue": first_queue,
                "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
            }

        if op == "message_check":
            store = self._store(req.workflow_name).require()
            group = p.get("group")
            if group is not None:
                check_group_id(group, {a["group"] for a in store.agents()} | set(store.definition()["agents"]))
            return {"agents": store.agents(group)}

        if op == "adopt":
            me = dict(p["me"])
            launch = dict(p["launch"])
            adopted: list[str] = []
            managed_updates: dict[str, dict[str, Any]] = {}
            notices: list[str] = []
            for wf_name in list_workflow_names(self._root):
                store = self._store(wf_name)
                control = store.control()
                holder = control.get("controller") or {}
                if (
                    control.get("state") != "running"
                    or holder.get("session_id") != me["session_id"]
                    or holder.get("instance_id") == me["instance_id"]
                ):
                    continue
                try:
                    _, epoch, watermark = self._take_control_locked(store, me, launch)
                    managed_updates[wf_name] = {"epoch": epoch, "watermark": watermark}
                    adopted.append(wf_name)
                    notices.append(
                        f"Workflow {wf_name} is running again under this session after a restart. See wf:{wf_name}."
                    )
                except Exception as exc:
                    notices.append(f"Workflow {wf_name} could not resume after a restart: {exc}")
            return {
                "adopted": adopted,
                "managed_updates": managed_updates,
                "notices": notices,
                "snapshot": self._build_snapshot(viewer_session_id=me["session_id"], default_model=default_model),
            }

        if op == "rediscover_apply":
            by_session: dict[str, dict[str, Any]] = dict(p.get("by_session") or {})
            adoptions: list[dict[str, Any]] = []
            for wf_name in list(p.get("names") or []):
                store = self._store(wf_name)
                if not store.exists():
                    continue
                for agent in store.agents():
                    row = by_session.get(agent.get("session_id") or "") if agent["state"] in ACTIVE_STATES else None
                    if row is None:
                        continue
                    store.update_agent(
                        agent["id"],
                        instance_id=row["instance_id"],
                        websocket_url=row.get("websocket_url") or agent.get("websocket_url"),
                        process_identity=row.get("process_identity") or agent.get("process_identity"),
                    )
                    adoptions.append({
                        "workflow": store.name,
                        "agent_id": agent["id"],
                        "session_id": row["session_id"],
                        "instance_id": row["instance_id"],
                        "websocket_url": row.get("websocket_url") or "",
                        "process_identity": row.get("process_identity"),
                    })
            return {
                "adoptions": adoptions,
                "snapshot": self._build_snapshot(viewer_session_id=viewer, default_model=default_model),
            }

        if op == "session_event_lookup":
            wf_name = req.workflow_name
            agent_id = str(p.get("agent_id") or "")
            session_id = str(p.get("session_id") or "")
            instance_id = str(p.get("instance_id") or "")
            if wf_name and agent_id:
                store = self._store(wf_name)
                if not store.exists():
                    return {"found": False}
                return {
                    "found": True,
                    "workflow_name": wf_name,
                    "agent": store.agent(agent_id),
                    "control": store.control(),
                }
            for candidate in list(p.get("managed_names") or []):
                store = self._store(candidate)
                if not store.exists():
                    continue
                for agent in store.agents():
                    if (instance_id and agent.get("instance_id") == instance_id) or (
                        session_id and agent.get("session_id") == session_id and agent["state"] in ACTIVE_STATES
                    ):
                        return {
                            "found": True,
                            "workflow_name": candidate,
                            "agent": agent,
                            "control": store.control(),
                        }
            return {"found": False}

        if op == "worker_remind":
            store = self._store(req.workflow_name)
            agent_id = str(p["agent_id"])
            try:
                context = store.agent(agent_id)
            except WorkflowError:
                return {"action": "none", "worker_context": None}
            context["workflow"] = store.name
            if context.get("state") in {"finished", "restarted"}:
                return {"action": "none", "worker_context": context}
            reminders = int(context.get("reminders", 0))
            if reminders == 0:
                updated = store.update_agent(agent_id, reminders=1)
                updated["workflow"] = store.name
                return {"action": "remind", "worker_context": updated}
            updated = store.update_agent(
                agent_id,
                reminders=reminders + 1,
                state="attention",
                problem="ended its turn again without calling finish_task; message it or use restart_agent",
            )
            updated["workflow"] = store.name
            return {"action": "attention", "worker_context": updated}

        raise ValueError(f"Unsupported WorkflowStoreRequest op: {op!r}")


# ---------------------------------------------------------------------------
# Pure Sans-I/O Controller: control, scheduling, live agents
# ---------------------------------------------------------------------------

class WorkflowController:
    """100% pure Sans-I/O workflow controller.

    All persistence/filesystem/lock/checkpoint I/O is yielded as typed
    ``WorkflowStoreRequest``s (``WorkflowStoreDriver``). All external session
    lifecycle, discovery, liveness observation, and WS transport delegate to
    ``AgentZooSessionDriver`` via yielded ``SessionRequest``s and unsolicited
    ``ExternalSession*`` events. Per-session state lives in ``state.durable``
    and ``state.runtime``.
    """

    def __init__(self, config, session=None):
        self.config = config
        self.session = session
        self._saved_config: dict[str, Any] | None = saved_model_config(config)
        self._state: Any = getattr(session, "state", None)
        self._fallback_durable = WorkflowDurableState()
        self._fallback_runtime = WorkflowRuntimeState()

    def _durable(self, state: Any = None) -> WorkflowDurableState:
        target = state if state is not None else self._state
        return _wf_durable(target) if target is not None else self._fallback_durable

    def _runtime(self, state: Any = None) -> WorkflowRuntimeState:
        target = state if state is not None else self._state
        return _wf_runtime(target) if target is not None else self._fallback_runtime

    @property
    def managed(self) -> dict[str, dict[str, Any]]:
        return self._durable().managed

    @property
    def me(self) -> dict[str, str] | None:
        return self._runtime().me

    @me.setter
    def me(self, value: dict[str, str] | None) -> None:
        self._runtime().me = value

    @property
    def launch(self) -> dict[str, Any] | None:
        return self._runtime().launch

    @launch.setter
    def launch(self, value: dict[str, Any] | None) -> None:
        self._runtime().launch = value

    @property
    def notices(self) -> collections.deque:
        return self._runtime().notices

    @property
    def notified(self) -> set:
        return self._runtime().notified

    @property
    def errors(self) -> dict[str, str]:
        return self._runtime().errors

    @property
    def adopted_instance(self) -> str:
        return self._runtime().adopted_instance

    @adopted_instance.setter
    def adopted_instance(self, value: str) -> None:
        self._runtime().adopted_instance = value

    @property
    def worker_record(self) -> dict[str, Any] | None:
        return self._runtime().cached_worker_context

    @worker_record.setter
    def worker_record(self, value: dict[str, Any] | None) -> None:
        self._runtime().cached_worker_context = value

    @property
    def worker_reminded_entry(self) -> int:
        return self._runtime().worker_reminded_entry

    @worker_reminded_entry.setter
    def worker_reminded_entry(self, value: int) -> None:
        self._runtime().worker_reminded_entry = value

    @property
    def instances_by_id(self) -> dict[str, tuple[str, str]]:
        return self._runtime().instances_by_id

    @property
    def in_flight_agents(self) -> set[tuple[str, str]]:
        return self._runtime().in_flight_agents

    # -- identity and launch settings (pure) ---------------------------------

    def bind(self, state) -> None:
        if state is not None:
            self._state = state
        rt = self._runtime(state)
        smeta = _session_meta(state)
        session_id = str(smeta.session_id or "")
        instance_id = str(smeta.instance_id or "")
        if session_id and instance_id:
            rt.me = {"session_id": session_id, "instance_id": instance_id}
        context = dict(getattr(state, "_agent_zoo_context", {}) or {})
        provenance = (self.config or {}).get("_runtime_launch", {}) or {}
        project = str(context.get("project_name") or smeta.project_name or "")
        store_root = str(context.get("project_store_root") or "")
        if not store_root and project:
            store_root = str(Path(resolve_state_root()) / "projects" / project / "coordination")
        if project and store_root:
            local_root = str(getattr(state, "_persistence_local_root", "") or "")
            if not local_root:
                try:
                    local_root = str(resolve_local_root(self.config))
                except Exception:
                    local_root = ""
            rt.launch = {
                "project": project,
                "store_root": store_root,
                "state_root": str(resolve_state_root()),
                "local_state_root": local_root,
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
        if self.launch is None:
            raise WorkflowError("this session cannot launch agents (it has no project or store root)")

    def default_model(self) -> str:
        return _default_model_from_sources(self._saved_config, self.config)

    def check_model(self, profile: str) -> str:
        return check_model(self.config, profile)

    def model_for(self, definition: dict, group: dict) -> str:
        model = group.get("model") or definition.get("model") or self.default_model()
        if not model:
            raise WorkflowError("no model is configured; set one with /workflow set NAME model PROFILE")
        return self.check_model(model)

    def spec_for(self, wf_name: str, agent: dict, *, message: str,
                 checkpoint: tuple[str, str] | None = None) -> SessionSpec:
        launch = self.launch
        return SessionSpec(
            project=launch["project"], workdir=agent["folder"], model=agent["model"], pipeline_key="default",
            env_name=launch["env_name"], config_path=launch["config_path"], config_sets=launch["config_sets"],
            state_root=launch["state_root"], store_root=launch["store_root"],
            local_state_root=launch["local_state_root"], host="127.0.0.1", port=0,
            parent_session_id=self.me["session_id"], kind=f"wf:{wf_name}:{agent['id']}",
            title=f"{wf_name}:{agent['id']}", first_user_message="" if checkpoint else message,
            lifetime="independent", startup_mode="paused" if checkpoint else "normal",
            extra_env=(("AZO_WORKFLOW", wf_name), ("AZO_WORKFLOW_AGENT", agent["id"])),
        )

    def notice(self, text: str, key=None) -> None:
        if key is not None:
            if key in self.notified:
                return
            self.notified.add(key)
        self.notices.append(text)

    def _snap_params(self) -> dict[str, Any]:
        return {
            "viewer_session_id": (self.me or {}).get("session_id", ""),
            "default_model": self.default_model(),
        }

    def _apply_result_snapshot(self, res: Any) -> None:
        if isinstance(res, dict) and "snapshot" in res and self._state is not None:
            _apply_snapshot(self._state, res["snapshot"])

    # -- control ----------------------------------------------------------------

    def run(self, name: str, group: str | None = None):
        self.require_launch()
        prep = yield WorkflowStoreRequest(
            op="run_prepare",
            workflow_name=name,
            params={**self._snap_params(), "group": group, "me": dict(self.me), "launch": dict(self.launch)},
        )
        self._apply_result_snapshot(prep)
        self.managed[name] = {"epoch": prep["epoch"], "watermark": prep["watermark"]}
        previous = prep["previous"]
        resumed, problems = yield from self.resume_paused(name, prep["paused_agents"])
        started = yield from self.tick(name)
        lines = []
        if previous and previous.get("session_id") != self.me["session_id"]:
            lines.append(f"Took control of {name} from session {previous['session_id'][:8]}.")
        scope = f"{group} in {name}" if group else name
        lines.append(f"Running {scope} from this session (def_{prep['version']}).")
        if resumed:
            lines.append("Resumed " + ", ".join(resumed) + " where they left off.")
        if started:
            lines.extend(started)
        if not resumed and not started:
            reason = yield WorkflowStoreRequest(op="idle_reason", workflow_name=name, params={"group": group})
            lines.append("No agents started yet: " + str(reason))
        lines.extend(problems)
        lines.append(f"See wf:{name} for progress.")
        return "\n".join(lines)

    def pause(self, name: str, group: str | None = None):
        prep = yield WorkflowStoreRequest(
            op="pause_prepare",
            workflow_name=name,
            params={**self._snap_params(), "group": group, "by": self.me_label()},
        )
        self._apply_result_snapshot(prep)
        stopped, starting, problems = [], [], []
        for agent in prep["agents"]:
            if agent["state"] == "starting":
                starting.append(agent["id"])
            elif agent["state"] == "running":
                problem = yield from self.pause_one(name, agent)
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

    def pause_one(self, wf_name: str, agent: dict):
        try:
            yield from self.send(wf_name, agent, "interrupt")
        except Exception as exc:
            return f"Could not interrupt {agent['id']}: {exc}"
        res = yield WorkflowStoreRequest(
            op="update_agent",
            workflow_name=wf_name,
            params={**self._snap_params(), "agent_id": agent["id"], "changes": {"state": "paused"}},
        )
        self._apply_result_snapshot(res)
        return None

    def archive(self, name: str):
        yield WorkflowStoreRequest(op="archive_check", workflow_name=name)
        yield from self.pause(name)
        agents = yield WorkflowStoreRequest(op="archive_list_agents", workflow_name=name)
        ended = []
        for agent in agents:
            yield from self.stop_agent(name, agent, reason="workflow")
            yield WorkflowStoreRequest(op="archive_return_agent", workflow_name=name, params={"agent": agent})
            ended.append(agent["id"])
        fin = yield WorkflowStoreRequest(
            op="archive_finalize",
            workflow_name=name,
            params={**self._snap_params(), "by": self.me_label()},
        )
        self._apply_result_snapshot(fin)
        self.managed.pop(name, None)
        text = (
            f"Archived {name}. It no longer appears in /workflow; its definition, items and history "
            f"stay readable at wf:{name}, and /workflow unarchive {name} brings it back."
        )
        if ended:
            text += f" Shut down {', '.join(ended)}."
        return text

    def unarchive(self, name: str):
        res = yield WorkflowStoreRequest(
            op="unarchive",
            workflow_name=name,
            params={**self._snap_params(), "by": self.me_label()},
        )
        self._apply_result_snapshot(res)
        return res["text"]

    def rename(self, name: str, new_name: str):
        res = yield WorkflowStoreRequest(
            op="rename",
            workflow_name=name,
            params={**self._snap_params(), "new_name": new_name, "by": self.me_label()},
        )
        self._apply_result_snapshot(res)
        target_name = res["new_name"]
        if name in self.managed:
            self.managed[target_name] = self.managed.pop(name)
        for iid, pair in list(self.instances_by_id.items()):
            if pair[0] == name:
                self.instances_by_id[iid] = (target_name, pair[1])
        self.errors.pop(name, None)
        return (
            f"Renamed {name} to {target_name}. Open it with /workflow view {target_name}; "
            f"agents read it at wf:{target_name}."
        )

    # -- agents (pure SessionRequests + WorkflowStoreRequests) -----------------

    def _owns_control(self, wf_name: str, control: dict) -> bool:
        mine = self.managed.get(wf_name)
        return (
            mine is not None
            and int(control.get("epoch", 0)) == mine["epoch"]
            and (control.get("controller") or {}).get("instance_id") == (self.me or {}).get("instance_id")
        )

    def _discover_target(self, wf_name: str, agent: dict):
        session_id = agent.get("session_id") or ""
        if not session_id:
            return None
        disc = yield DiscoverSessions(
            store_root=(self.launch or {}).get("store_root") or "",
            session_id=session_id,
            include_stale=False,
            probe_local_processes=True,
        )
        row = _pick_live_row(disc)
        if row is None:
            return None
        identity = row.process_identity or agent.get("process_identity")
        res = yield WorkflowStoreRequest(
            op="update_agent",
            workflow_name=wf_name,
            params={
                **self._snap_params(),
                "agent_id": agent["id"],
                "changes": {
                    "instance_id": row.instance_id,
                    "websocket_url": row.websocket_url,
                    "process_identity": identity,
                },
            },
        )
        self._apply_result_snapshot(res)
        self.instances_by_id[row.instance_id] = (wf_name, agent["id"])
        return {
            "session_id": row.session_id,
            "instance_id": row.instance_id,
            "websocket_url": row.websocket_url,
            "process_identity": identity,
        }

    def _resolve_target(self, wf_name: str, agent: dict):
        if agent.get("websocket_url") and agent.get("session_id"):
            return {
                "session_id": agent["session_id"],
                "instance_id": agent.get("instance_id") or "",
                "websocket_url": agent["websocket_url"],
                "process_identity": agent.get("process_identity"),
            }
        target = yield from self._discover_target(wf_name, agent)
        if target is None:
            raise WorkflowError(f"{agent['id']} is not running")
        return target

    @staticmethod
    def _send_request(target: dict, action: str, message: str = "") -> SendSessionInput:
        if action == "message":
            channel, payload = "user_text", {"text": str(message)}
        elif action in {"interrupt", "continue"}:
            channel, payload = "slash_command", {"raw": f"/{action}"}
        else:
            raise WorkflowError(f"unsupported agent action {action!r}")
        return SendSessionInput(
            session_id=target["session_id"],
            instance_id=target.get("instance_id") or "",
            channel=channel,
            payload=payload,
            client_id=uuid.uuid4().hex,
            websocket_url=target["websocket_url"],
            expected_process_identity=target.get("process_identity"),
            verify_loopback=True,
            timeout_s=30.0,
        )

    def send(self, wf_name: Any, agent: dict, action: str, message: str = ""):
        """Deliver one user message or slash command to a live agent via ``SendSessionInput``."""
        name = wf_name.name if isinstance(wf_name, WorkflowStore) else str(wf_name)
        target = yield from self._resolve_target(name, agent)
        result = yield self._send_request(target, action, message)
        if isinstance(result, SessionDeliveryResult):
            if result.outcome == "sent":
                return
            if result.outcome in {"pre_send_race", "unavailable"}:
                refreshed = yield from self._discover_target(name, agent)
                if refreshed is not None and refreshed["instance_id"] != target["instance_id"]:
                    result = yield self._send_request(refreshed, action, message)
                    if isinstance(result, SessionDeliveryResult) and result.outcome == "sent":
                        return
            raise WorkflowError(result.error or f"could not deliver {action} to {agent['id']} ({result.outcome})")

    def send_data(self, wf_name: Any, agent: dict, artifacts: Sequence[SessionArtifact], *,
                  channel: str = "workflow", message: str = ""):
        """Push typed ``SessionArtifact`` payloads to a live agent via ``SendSessionData``."""
        name = wf_name.name if isinstance(wf_name, WorkflowStore) else str(wf_name)
        target = yield from self._resolve_target(name, agent)
        result = yield SendSessionData(
            session_id=target["session_id"],
            instance_id=target.get("instance_id") or "",
            artifacts=tuple(artifacts),
            channel=channel,
            message=message,
            sender_session_id=(self.me or {}).get("session_id", ""),
            sender_instance_id=(self.me or {}).get("instance_id", ""),
            client_id=uuid.uuid4().hex,
            websocket_url=target["websocket_url"],
            expected_process_identity=target.get("process_identity"),
            verify_loopback=True,
            timeout_s=30.0,
        )
        if isinstance(result, SessionDeliveryResult) and result.outcome != "sent":
            raise WorkflowError(result.error or f"could not deliver data to {agent['id']} ({result.outcome})")

    def stop_agent(self, wf_name: Any, agent: dict, *, reason: str = "workflow"):
        """Request graceful WS shutdown for a live agent via ``StopSession``."""
        del wf_name
        session_id = agent.get("session_id") or ""
        if not session_id:
            return
        try:
            obs = yield ObserveSession(session_id=session_id, instance_id=agent.get("instance_id") or "")
        except Exception as exc:
            LOG.warning("Could not observe %s before shutdown: %s", agent["id"], exc)
            return
        live = obs.live_info if isinstance(obs, SessionObservation) else None
        if isinstance(obs, SessionObservation) and (obs.alive is False or (obs.alive is None and not agent.get("websocket_url"))):
            return
        try:
            yield StopSession(
                session_id=session_id,
                instance_id=(live.instance_id if live and live.instance_id else agent.get("instance_id") or ""),
                mode="graceful_ws", reason=reason, save=True, spawned_policy="leave",
                expected_process_identity=(live.process_identity if live and live.process_identity else agent.get("process_identity")),
                timeout_s=5.0,
            )
        except Exception as exc:
            LOG.warning("Could not shut down %s: %s", agent["id"], exc)

    def spawn_agent(
        self,
        wf_name: str,
        agent: dict,
        message: str = "",
        *,
        checkpoint: tuple[str, str] | None = None,
        recovery_action: str | None = None,
        model: str | None = None,
        render_from_settings: dict | None = None,
        items: list[dict] | None = None,
    ):
        """Start (or resume) one agent's session via ``WorkflowStoreRequest`` + ``SpawnSession``."""
        session_id = agent.get("session_id") or str(uuid.uuid4())
        instance_id = str(uuid.uuid4())
        prep = yield WorkflowStoreRequest(
            op="prepare_spawn",
            workflow_name=wf_name,
            params={
                **self._snap_params(),
                "agent_id": agent["id"],
                "session_id": session_id,
                "instance_id": instance_id,
                "project": self.launch["project"],
                "is_resume": checkpoint is not None,
                "recovery_action": recovery_action,
                "model": model,
                "message": message,
                "render_instructions": render_from_settings is not None,
                "settings": render_from_settings,
                "items": items or [],
            },
        )
        self._apply_result_snapshot(prep)
        agent = prep["agent"]
        rendered_message = prep["message"]
        source = (
            SessionSource(kind="resume", session_id=checkpoint[0], revision_id=checkpoint[1])
            if checkpoint else SessionSource(kind="fresh")
        )
        try:
            spawn_result = yield SpawnSession(
                session_id=session_id,
                instance_id=instance_id,
                source=source,
                spec=self.spec_for(wf_name, agent, message=rendered_message, checkpoint=checkpoint),
                wait_ready=False,
                ready_timeout_s=float(READY_TIMEOUT_SECONDS),
                watch=False,
                result_key=f"wf:{wf_name}:{agent['id']}:{instance_id}",
                launch_metadata={"workflow": wf_name, "agent_id": agent["id"]},
            )
        except Exception:
            yield WorkflowStoreRequest(
                op="update_agent",
                workflow_name=wf_name,
                params={
                    **self._snap_params(),
                    "agent_id": agent["id"],
                    "changes": {"process_identity": None, "problem": "launch did not complete"},
                },
            )
            raise
        is_res = isinstance(spawn_result, SessionSpawnResult)
        resolved_iid = (spawn_result.instance_id if is_res and spawn_result.instance_id else instance_id)
        self.instances_by_id[resolved_iid] = (wf_name, agent["id"])
        updates: dict[str, Any] = {
            "instance_id": resolved_iid,
            "pid": spawn_result.pid if is_res else None,
            "process_identity": spawn_result.process_identity if is_res else None,
        }
        if is_res and spawn_result.ready:
            updates["websocket_url"] = spawn_result.ready.get("websocket_url") or spawn_result.websocket_url or None
        upd = yield WorkflowStoreRequest(
            op="update_agent",
            workflow_name=wf_name,
            params={**self._snap_params(), "agent_id": agent["id"], "changes": updates},
        )
        self._apply_result_snapshot(upd)
        try:
            yield ConfigureSessionLifecycle(spawned_policy="leave", session_ids=(session_id,))
        except Exception:
            LOG.debug("Could not configure independent lifecycle for %s", session_id, exc_info=True)

    def resume_agent(self, wf_name: str, agent: dict):
        """Relaunch a stopped agent from its saved checkpoint. Returns a problem or None."""
        if not agent.get("session_id"):
            return f"{agent['id']} never started, so there is nothing to resume"
        mine = self.managed.get(wf_name)
        claim = yield WorkflowStoreRequest(
            op="claim_resume",
            workflow_name=wf_name,
            params={
                "agent": agent,
                "epoch": mine["epoch"] if mine is not None else None,
                "project": self.launch["project"],
            },
        )
        if claim["status"] == "skip":
            return None
        if claim["status"] == "problem":
            return claim["problem"]
        try:
            yield from self.spawn_agent(
                wf_name,
                claim["agent"],
                "",
                checkpoint=(agent["session_id"], claim["commit_id"]),
                recovery_action=claim["action"],
            )
        except Exception as exc:
            return f"{agent['id']} could not be relaunched: {exc}"
        upd = yield WorkflowStoreRequest(
            op="update_agent",
            workflow_name=wf_name,
            params={
                **self._snap_params(),
                "agent_id": agent["id"],
                "changes": {"resumes": int(agent.get("resumes", 0)) + 1, "last_resume": time.time()},
            },
        )
        self._apply_result_snapshot(upd)
        return None

    def resume_paused(self, wf_name: str, paused_agents: list[dict]):
        resumed, problems = [], []
        for agent in paused_agents:
            if agent["state"] != "paused":
                continue
            obs = yield ObserveSession(
                session_id=agent.get("session_id") or "",
                instance_id=agent.get("instance_id") or "",
            )
            alive = obs.alive if isinstance(obs, SessionObservation) else None
            if alive is True and agent.get("websocket_url"):
                try:
                    if agent.get("release") in {"resume", "continue"}:
                        yield from self.release_ready(wf_name, agent, True)
                    else:
                        yield from self.send(wf_name, agent, "continue")
                        upd = yield WorkflowStoreRequest(
                            op="update_agent",
                            workflow_name=wf_name,
                            params={**self._snap_params(), "agent_id": agent["id"], "changes": {"state": "running"}},
                        )
                        self._apply_result_snapshot(upd)
                    check = yield WorkflowStoreRequest(
                        op="read_agent", workflow_name=wf_name, params={"agent_id": agent["id"]},
                    )
                    if check["agent"]["state"] == "running":
                        resumed.append(agent["id"])
                except Exception as exc:
                    problems.append(f"Could not resume {agent['id']}: {exc}")
            elif alive is True:
                upd = yield WorkflowStoreRequest(
                    op="update_agent",
                    workflow_name=wf_name,
                    params={
                        **self._snap_params(),
                        "agent_id": agent["id"],
                        "changes": {"state": "starting", "release": agent.get("release") or "continue"},
                    },
                )
                self._apply_result_snapshot(upd)
                resumed.append(agent["id"])
            elif alive is False:
                problem = yield from self.resume_agent(wf_name, agent)
                if problem:
                    yield from self.mark_attention(
                        wf_name, agent, problem + "; use restart_agent to give its item to a fresh agent"
                    )
                    problems.append(problem + ".")
                else:
                    resumed.append(agent["id"])
            else:
                problems.append(f"Could not tell whether {agent['id']} is still running; left it paused.")
        return resumed, problems

    def mark_attention(self, wf_name: str, agent: dict, problem: str):
        res = yield WorkflowStoreRequest(
            op="mark_attention",
            workflow_name=wf_name,
            params={**self._snap_params(), "agent_id": agent["id"], "problem": problem},
        )
        self._apply_result_snapshot(res)

    # -- scheduling ---------------------------------------------------------------

    def tick(self, name: str):
        """One pure Sans-I/O scheduling pass. Returns descriptions of agents started."""
        mine = self.managed.get(name)
        if mine is None:
            return []
        info = yield WorkflowStoreRequest(op="tick_read", workflow_name=name)
        if not info["exists"]:
            self.managed.pop(name, None)
            return []
        control = info["control"]
        holder = control.get("controller") or {}
        if (
            holder.get("instance_id") != (self.me or {}).get("instance_id")
            or int(control.get("epoch", 0)) != mine["epoch"]
        ):
            self.managed.pop(name, None)
            if control.get("state") != "archived":
                who = holder.get("session_id", "another session")[:8] if holder else "nobody"
                self.notice(f"Workflow {name} is now controlled by session {who}; this session stopped scheduling it.")
            return []
        definition = info["definition"]
        paused_groups = set(control.get("paused_groups") or [])
        running = control.get("state") == "running"
        for agent in info["agents"]:
            if agent["state"] in {"starting", "running"}:
                yield from self.observe(name, agent, definition, running and agent["group"] not in paused_groups)
        started = []
        if running:
            started = yield from self.start_agents(name, definition, paused_groups)
        yield from self._collect_notices(name)
        return started

    def _collect_notices(self, wf_name: str):
        mine = self.managed.get(wf_name)
        watermark = mine.get("watermark", 0) if mine is not None else 0
        rep = yield WorkflowStoreRequest(
            op="report_notices",
            workflow_name=wf_name,
            params={**self._snap_params(), "watermark": watermark},
        )
        self._apply_result_snapshot(rep)
        for item in rep.get("attention", []):
            self.notice(item["text"], key=item["key"])
        if mine is not None:
            mine["watermark"] = rep["watermark"]
            for text in rep.get("new_items", []):
                self.notice(text)

    def release_ready(self, wf_name: str, agent: dict, should_run: bool):
        """Release a ready process before claiming it is running; pause wins."""
        read = yield WorkflowStoreRequest(
            op="read_agent", workflow_name=wf_name, params={"agent_id": agent["id"]},
        )
        agent, control = read["agent"], read["control"]
        if agent["state"] not in ACTIVE_STATES or not self._owns_control(wf_name, control):
            return
        should_run = (
            should_run
            and control.get("state") == "running"
            and agent["group"] not in control.get("paused_groups", [])
        )
        try:
            release = agent.get("release")
            if not should_run:
                if release in {"resume", "continue"}:
                    upd = yield WorkflowStoreRequest(
                        op="update_agent",
                        workflow_name=wf_name,
                        params={**self._snap_params(), "agent_id": agent["id"], "changes": {"state": "paused"}},
                    )
                    self._apply_result_snapshot(upd)
                else:
                    problem = yield from self.pause_one(wf_name, agent)
                    if problem:
                        raise WorkflowError(problem)
                return
            if release in {"resume", "continue"}:
                action = "continue" if release == "continue" else agent.get("recovery_action")
                if action not in {"continue", "message"}:
                    yield from self.mark_attention(
                        wf_name, agent, "recovery action is missing; checkpoint must be reviewed"
                    )
                    return
                yield from self.send(wf_name, agent, action, RECOVERY_MESSAGE if action == "message" else "")
            committed = yield WorkflowStoreRequest(
                op="release_commit",
                workflow_name=wf_name,
                params={**self._snap_params(), "agent_id": agent["id"], "instance_id": agent.get("instance_id")},
            )
            self._apply_result_snapshot(committed)
        except Exception as exc:
            yield from self.mark_attention(
                wf_name, agent,
                f"could not confirm recovery release: {exc}; inspect the live session before retrying",
            )

    def _handle_agent_dead(self, wf_name: str, agent: dict, should_run: bool):
        read = yield WorkflowStoreRequest(
            op="read_agent", workflow_name=wf_name, params={"agent_id": agent["id"]},
        )
        agent, control = read["agent"], read["control"]
        if agent["state"] not in {"starting", "running"} or not self._owns_control(wf_name, control):
            return
        should_run = (
            should_run
            and control.get("state") == "running"
            and agent["group"] not in control.get("paused_groups", [])
        )
        if not should_run:
            upd = yield WorkflowStoreRequest(
                op="update_agent",
                workflow_name=wf_name,
                params={**self._snap_params(), "agent_id": agent["id"], "changes": {"state": "paused"}},
            )
            self._apply_result_snapshot(upd)
            return
        recent = time.time() - float(agent.get("last_resume") or 0) < 600
        if recent and int(agent.get("resumes", 0)) >= MAX_QUICK_RESUMES:
            yield from self.mark_attention(
                wf_name, agent,
                f"stopped unexpectedly {agent['resumes']} times in a row; "
                "read its transcript, then use restart_agent",
            )
            return
        problem = yield from self.resume_agent(wf_name, agent)
        if problem:
            yield from self.mark_attention(
                wf_name, agent, problem + "; use restart_agent to give its item to a fresh agent"
            )

    def observe(self, wf_name: Any, agent: dict, definition: dict, should_run: bool):
        del definition
        name = wf_name.name if isinstance(wf_name, WorkflowStore) else str(wf_name)
        read = yield WorkflowStoreRequest(
            op="read_agent", workflow_name=name, params={"agent_id": agent["id"]},
        )
        agent = read["agent"]
        key = (name, agent["id"])
        if agent["state"] not in {"starting", "running"} or key in self.in_flight_agents:
            return
        self.in_flight_agents.add(key)
        try:
            obs = yield ObserveSession(
                session_id=agent.get("session_id") or "",
                instance_id=agent.get("instance_id") or "",
            )
            alive = obs.alive if isinstance(obs, SessionObservation) else None
            live_info = obs.live_info if isinstance(obs, SessionObservation) else None
            if agent["state"] == "starting":
                ws_url = (
                    live_info.websocket_url
                    if live_info and not live_info.is_terminal and live_info.process_state != "dead"
                    else ""
                )
                if not ws_url and alive is not False and agent.get("session_id"):
                    disc = yield DiscoverSessions(
                        store_root=(self.launch or {}).get("store_root") or "",
                        session_id=agent["session_id"],
                        include_stale=False,
                        probe_local_processes=True,
                    )
                    discovered = _pick_live_row(disc)
                    if discovered is not None:
                        live_info, ws_url = discovered, discovered.websocket_url
                if ws_url and alive is not False:
                    new_iid = live_info.instance_id if live_info and live_info.instance_id else agent["instance_id"]
                    upd = yield WorkflowStoreRequest(
                        op="update_agent",
                        workflow_name=name,
                        params={
                            **self._snap_params(),
                            "agent_id": agent["id"],
                            "changes": {
                                "websocket_url": ws_url,
                                "instance_id": new_iid,
                                "process_identity": (
                                    live_info.process_identity
                                    if live_info and live_info.process_identity
                                    else agent.get("process_identity")
                                ),
                            },
                        },
                    )
                    self._apply_result_snapshot(upd)
                    agent = upd["agent"]
                    self.instances_by_id[new_iid] = key
                    yield from self.release_ready(name, agent, should_run)
                    return
                if alive is True:
                    if time.time() - float(agent.get("launch_started") or 0) > READY_TIMEOUT_SECONDS:
                        yield from self.mark_attention(name, agent, "did not finish starting within two minutes")
                    return
            if alive is False:
                yield from self._handle_agent_dead(name, agent, should_run)
        finally:
            self.in_flight_agents.discard(key)

    def _handle_session_event(self, event: ExternalSessionReady | ExternalSessionExited):
        pair = self.instances_by_id.get(event.instance_id) if event.instance_id else None
        if isinstance(event, ExternalSessionExited) and event.instance_id:
            self.instances_by_id.pop(event.instance_id, None)
        if pair is not None and (pair[0] not in self.managed or pair in self.in_flight_agents):
            return
        lookup = yield WorkflowStoreRequest(
            op="session_event_lookup",
            workflow_name=pair[0] if pair is not None else "",
            params={
                "agent_id": pair[1] if pair is not None else "",
                "session_id": event.session_id,
                "instance_id": event.instance_id,
                "managed_names": list(self.managed),
            },
        )
        if not lookup.get("found"):
            return
        name = lookup["workflow_name"]
        agent = lookup["agent"]
        agent_id = agent["id"]
        pair = (name, agent_id)
        if name not in self.managed or pair in self.in_flight_agents:
            return
        if agent.get("instance_id") and event.instance_id and agent["instance_id"] != event.instance_id:
            return
        if isinstance(event, ExternalSessionReady) and agent["state"] != "starting":
            return
        self.in_flight_agents.add(pair)
        try:
            control = lookup["control"]
            should_run = control.get("state") == "running" and agent["group"] not in (control.get("paused_groups") or [])
            if isinstance(event, ExternalSessionReady):
                upd = yield WorkflowStoreRequest(
                    op="update_agent",
                    workflow_name=name,
                    params={
                        **self._snap_params(),
                        "agent_id": agent_id,
                        "changes": {
                            "websocket_url": event.websocket_url or agent.get("websocket_url"),
                            "instance_id": event.instance_id or agent.get("instance_id"),
                            "process_identity": event.process_identity or agent.get("process_identity"),
                        },
                    },
                )
                self._apply_result_snapshot(upd)
                agent = upd["agent"]
                if event.instance_id:
                    self.instances_by_id[event.instance_id] = pair
                yield from self.release_ready(name, agent, should_run)
            else:
                yield from self._handle_agent_dead(name, agent, should_run)
                yield from self._collect_notices(name)
        finally:
            self.in_flight_agents.discard(pair)

    def on_session_ready(self, event: ExternalSessionReady):
        """Handle unsolicited ``ExternalSessionReady`` from ``AgentZooSessionDriver``."""
        return self._handle_session_event(event)

    def on_session_exited(self, event: ExternalSessionExited):
        """Handle unsolicited ``ExternalSessionExited`` from ``AgentZooSessionDriver``."""
        return self._handle_session_event(event)

    def start_agents(self, wf_name: Any, definition: dict, paused_groups: set):
        del paused_groups
        name = wf_name.name if isinstance(wf_name, WorkflowStore) else str(wf_name)
        mine = self.managed.get(name)
        if mine is None:
            return []
        plan = yield WorkflowStoreRequest(
            op="plan_start_agents",
            workflow_name=name,
            params={
                "epoch": mine["epoch"],
                "instance_id": (self.me or {}).get("instance_id"),
                "cwd": (self.launch or {}).get("cwd") or "",
            },
        )
        definition = plan["definition"]
        started = []
        for entry in plan["planned"]:
            agent, settings, items = entry["agent"], entry["settings"], entry["items"]
            try:
                model = self.model_for(definition, settings)
                yield from self.spawn_agent(
                    name, agent, "",
                    model=model,
                    render_from_settings=settings,
                    items=items,
                )
                logged = yield WorkflowStoreRequest(
                    op="log_agent_started",
                    workflow_name=name,
                    params={**self._snap_params(), "agent_id": agent["id"], "items": [i["id"] for i in items]},
                )
                self._apply_result_snapshot(logged)
                started.append(f"Started {agent['id']}" + (f" on {batch_text(items)}." if items else "."))
            except Exception as exc:
                yield from self.launch_failed(name, agent, items, settings, exc)
        return started

    def launch_failed(self, wf_name: str, agent: dict, items: list[dict], settings, exc):
        del settings
        group = agent["group"]
        res = yield WorkflowStoreRequest(
            op="launch_failed",
            workflow_name=wf_name,
            params={**self._snap_params(), "agent": agent, "items": items, "error": str(exc)},
        )
        self._apply_result_snapshot(res)
        self.notice(
            f"Workflow {wf_name}: could not start an agent in {group}: {exc}. Paused {group}; "
            f"fix the problem (see wf:{wf_name}:{group}), then run it again."
        )

    # -- other controller operations ---------------------------------------------

    def message_agents(self, name: str, message: str, group: str | None):
        if not message.strip():
            raise WorkflowError("message is empty")
        res = yield WorkflowStoreRequest(op="message_check", workflow_name=name, params={"group": group})
        sent, skipped, problems = [], [], []
        for agent in res["agents"]:
            if agent["state"] == "running":
                try:
                    yield from self.send(name, agent, "message", message)
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

    def restart_agent(self, name: str, agent_id: str):
        checked = yield WorkflowStoreRequest(
            op="restart_check", workflow_name=name, params={"agent_id": agent_id},
        )
        agent = checked["agent"]
        yield from self.stop_agent(name, agent, reason="workflow")
        item_ids = agent_items(agent)
        fin = yield WorkflowStoreRequest(
            op="restart_finalize",
            workflow_name=name,
            params={**self._snap_params(), "agent_id": agent["id"], "item_ids": item_ids},
        )
        self._apply_result_snapshot(fin)
        text = f"Stopped {agent_id} and discarded its progress."
        if len(item_ids) == 1:
            text += f" {item_ids[0]} is back in {fin['first_queue']} for a fresh agent."
        elif item_ids:
            text += f" {', '.join(item_ids)} are back in their queues for a fresh agent."
        if name not in self.managed:
            text += f" This session does not control {name}; run it to start new agents."
        return text

    def change_setting(self, name: str, kind: str, target: str | None, setting: str, raw: str):
        """Apply one /workflow set change and describe its effect."""
        res = yield WorkflowStoreRequest(
            op="set_setting",
            workflow_name=name,
            params={
                **self._snap_params(),
                "kind": kind,
                "target": target,
                "setting": setting,
                "raw": raw,
                "config": self.config,
            },
        )
        self._apply_result_snapshot(res)
        change = res["change"]
        label, setting_name = change["label"], change["setting"]
        if res["unchanged"]:
            return f"{label} {setting_name} is already {change['new_text']}; nothing changed."
        result = res["result"]
        text = (
            f"Set {label} {setting_name} from {change['old_text']} to {change['new_text']} "
            f"(def_{result['version']})."
        )
        definition = res["definition"]
        active = res["active"]
        control = res["control"]
        if setting_name == "count":
            if change["value"] < int(change["old_text"]):
                text += " Extra running agents finish their current item first."
            elif name in self.managed:
                started = yield from self.tick(name)
                text += (" " + " ".join(started)) if started else ""
            elif control.get("state") == "running":
                text += " The session controlling it will start agents shortly."
            else:
                text += f" {name} is paused; agents start when it runs."
        elif setting_name == "model":
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
                text += (
                    f" {', '.join(stale)} keep their current model; restart them with restart_agent "
                    "or let them finish to switch."
                )
        elif setting_name == "folder":
            scope = [a["id"] for a in active if kind == "workflow" or a["group"] == target]
            text += " New agents start there."
            if scope:
                text += f" {', '.join(scope)} keep their current folder."
        elif setting_name == "every":
            text += " It applies from the group's next start."
        elif setting_name == "description":
            text += " It shows in /workflow list."
        elif setting_name == "notify":
            text += (
                f" The controlling agent will be told about new items in {target}."
                if change["value"] else f" New items in {target} no longer notify anyone."
            )
        return text

    # -- startup adoption & reload rediscovery (pure generators) -----------------

    def adopt(self, state):
        """After a restart or reload, keep running workflows this session controlled."""
        self.bind(state)
        if not self.me or self.adopted_instance == self.me["instance_id"]:
            return []
        self.adopted_instance = self.me["instance_id"]
        if self.launch is None:
            snap = yield WorkflowStoreRequest(op="snapshot", params=self._snap_params())
            _apply_snapshot(state, snap)
            return []
        res = yield WorkflowStoreRequest(
            op="adopt",
            params={**self._snap_params(), "me": dict(self.me), "launch": dict(self.launch)},
        )
        _apply_snapshot(state, res.get("snapshot"))
        for wf_name, entry in res.get("managed_updates", {}).items():
            self.managed[wf_name] = entry
        for text in res.get("notices", []):
            self.notice(text)
        return list(res.get("adopted", []))

    def rediscover_workflows(self, names: Sequence[str]):
        """Rediscover live worker sessions after parent reload via ``DiscoverSessions`` + ``AdoptSession``."""
        root = (self.launch or {}).get("store_root") or ""
        if not names or not root:
            return
        disc = yield DiscoverSessions(store_root=root, include_stale=False, probe_local_processes=True)
        by_session: dict[str, dict[str, Any]] = {}
        for row in (disc.rows if isinstance(disc, DiscoverSessionsResult) else ()):
            if row.is_terminal or row.process_state == "dead":
                continue
            existing = by_session.get(row.session_id)
            key_new = (row.updated_at, row.started_at)
            if existing is None or key_new >= existing["_order"]:
                by_session[row.session_id] = {
                    "_order": key_new,
                    "session_id": row.session_id,
                    "instance_id": row.instance_id,
                    "websocket_url": row.websocket_url,
                    "process_identity": row.process_identity,
                }
        for row in by_session.values():
            row.pop("_order", None)
        applied = yield WorkflowStoreRequest(
            op="rediscover_apply",
            params={**self._snap_params(), "names": list(names), "by_session": by_session},
        )
        self._apply_result_snapshot(applied)
        for item in applied.get("adoptions", []):
            self.instances_by_id[item["instance_id"]] = (item["workflow"], item["agent_id"])
            try:
                yield AdoptSession(
                    session_id=item["session_id"],
                    instance_id=item["instance_id"],
                    process_identity=item["process_identity"],
                    websocket_url=item["websocket_url"],
                    result_key=f"wf:{item['workflow']}:{item['agent_id']}:{item['instance_id']}",
                    watch=False,
                    metadata={"workflow": item["workflow"], "agent_id": item["agent_id"]},
                )
            except Exception:
                LOG.debug("Could not adopt worker %s/%s", item["workflow"], item["agent_id"], exc_info=True)


# ---------------------------------------------------------------------------
# Tools (100% pure Sans-I/O)
# ---------------------------------------------------------------------------

def function_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    properties = deepcopy(properties)
    for key, prop in properties.items():
        if key not in required:
            kinds = prop["type"] if isinstance(prop["type"], list) else [prop["type"]]
            prop["type"] = list(dict.fromkeys([*kinds, "null"]))
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


def text_property(description: str) -> dict:
    return {"type": "string", "description": description}


WORKFLOW_NAME_PROPERTY = text_property("Workflow name.")
AGENT_GROUP_PROPERTY = text_property("Single agent group (grp_*), or null for all.")
NOTE_PROPERTY = text_property("Optional note shown to the next agent.")


def buffer_text(state, buffer_id: str) -> str:
    manager = getattr(state, "buffer_manager", None)
    if manager is None:
        raise WorkflowError("buffers are unavailable in this session")
    try:
        return manager.resolve_for_read(state, str(buffer_id)).text
    except KeyError:
        raise WorkflowError(f"there is no buffer {buffer_id}") from None


class WorkflowTool(Tool):
    """A pure Sans-I/O tool with separate behavior (and schema) for controllers and workers."""

    group = WORKER_TOOLS_GROUP
    optional_reads = {"_agent_zoo_context", "buffer_manager", "_persistence_local_root", "cwd", "session_cwd"}
    writes = {"result"}

    def __init__(self, name, *, controller_schema=None, worker_schema=None, on_controller=None, on_worker=None,
                 plugin=None):
        super().__init__(name=name, group=WORKER_TOOLS_GROUP)
        self.controller_schema, self.worker_schema = controller_schema, worker_schema
        self.on_controller, self.on_worker = on_controller, on_worker
        self.plugin = plugin
        static = controller_schema or worker_schema
        if callable(static):
            static = static({"sends_to": []})
        self.schema = static
        self.description = static["function"]["description"]

    def available(self, state) -> bool:
        return bool(self.worker_schema if worker_identity(state) else self.controller_schema)

    def _publish(self, state) -> None:
        worker = worker_identity(state)
        schema = self.worker_schema if worker else self.controller_schema
        if worker and callable(schema):
            schema = schema(self.plugin.worker_context(state))
        state.tool_schemas[self.name] = {"schema": schema or self.schema, "available": bool(schema)}

    def _drive_generator(self, state: State, gen):
        result = yield from gen
        self.plugin.ensure_tick_timer(state)
        return str(result)

    def call(self, state: State, **kwargs):
        worker = worker_identity(state)
        handler = self.on_worker if worker else self.on_controller
        if handler is None:
            raise WorkflowError(f"{self.name} is not available in this session")
        if not worker:
            self.plugin.controller.bind(state)
        result = handler(state, **kwargs)
        if inspect.isgenerator(result):
            return self._drive_generator(state, result)
        return str(result)

    def _finish_ok(self, tc: ToolCall, result, ts: float, emit: EmitFn) -> None:
        super()._finish_ok(tc, result, ts, emit)
        if self.name == "finish_task":
            emit(
                SuspendRequested(
                    reason="suspend",
                    save=True,
                    spawned_policy="terminate",
                    registry_status="completed",
                )
            )

    def _finish_error(self, tc: ToolCall, exc: BaseException, ts: float, emit: EmitFn) -> None:
        if isinstance(exc, (WorkflowError, TypeError, ValueError, OSError)):
            tc.result = f"ERROR: {exc}"
        else:
            LOG.exception("Workflow tool %s failed", self.name)
            tc.result = f"ERROR: {type(exc).__name__}: {exc}"
        tc.error = True
        tc.finished_at = ts
        tc._owner = None
        emit(ToolCallFinished(tc))


class WorkflowSchemaPublisher:
    """Publish dynamic controller/worker schemas onto ``state.tool_schemas`` (pure Sans-I/O)."""

    handles = (RenderRequested, TurnSettled)
    reads = {"tool_schemas"}
    optional_reads = set()
    writes = {"tool_schemas"}

    def __init__(self, tools: list[WorkflowTool]):
        self._tools = tuple(tools)

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        del event, emit
        schemas = getattr(state, "tool_schemas", None)
        if not isinstance(schemas, dict):
            state.tool_schemas = {}
        for tool in self._tools:
            tool._publish(state)


class WorkflowsPlugin:
    """Pure Sans-I/O tool handlers, task driver, and cached worker context for one pipeline."""

    def __init__(self, controller: WorkflowController):
        self.controller = controller

    def ensure_tick_timer(self, state: State) -> OpId | None:
        """Schedule the next periodic scheduling timer when any workflow is managed."""
        durable = _wf_durable(state)
        rt = _wf_runtime(state)
        if (
            not durable.managed
            or rt.tick_active
            or rt.tick_timer_op is not None
            or getattr(state, "shutdown", False)
            or getattr(state, "done", False)
            or not hasattr(state, "io")
        ):
            return rt.tick_timer_op
        rt.tick_timer_op = state.io.submit(StartTimer(delay_s=TICK_SECONDS), tag="workflow_tick")
        return rt.tick_timer_op

    def advance_task(
        self,
        state: State,
        task: _WorkflowTask,
        *,
        first: bool = False,
        send_value: Any = None,
        throw_exc: BaseException | None = None,
    ) -> tuple[OpId | None, bool, Any]:
        """Step a generator task until it yields an ``IoRequest`` or completes."""
        try:
            if first:
                yielded = next(task.gen)
            elif throw_exc is not None:
                yielded = task.gen.throw(throw_exc)
            else:
                yielded = task.gen.send(send_value)
        except StopIteration as stop:
            return None, True, stop.value
        except Exception as exc:
            return None, False, exc
        if not isinstance(yielded, IoRequest):
            try:
                task.gen.close()
            except Exception:
                pass
            return None, False, TypeError(
                f"Workflow task yielded unsupported object {type(yielded).__name__}; expected IoRequest"
            )
        tag = "workflow_command" if task.kind in {"command", "command_doc"} else "workflow_scheduler"
        op_id = state.io.submit(yielded, tag=tag, hold_readiness=task.hold_readiness)
        _wf_runtime(state).pending_tasks[op_id] = task
        return op_id, False, None

    def start_event_task(
        self,
        state: State,
        emit: EmitFn,
        kind: str,
        gen,
        workflow_name: str = "",
        *,
        hold_readiness: str | None = None,
    ) -> OpId | None:
        task = _WorkflowTask(
            kind=kind,
            gen=gen,
            workflow_name=workflow_name,
            hold_readiness=hold_readiness,
        )
        op_id, ok, val = self.advance_task(state, task, first=True)
        if op_id is not None:
            return op_id
        self.finish_task_outcome(state, emit, task, ok, val)
        return None

    def finish_task_outcome(self, state: State, emit: EmitFn, task: _WorkflowTask, ok: bool, val: Any) -> None:
        rt = _wf_runtime(state)
        if task.hold_readiness and emit is not None:
            emit(PacingCheck(reason="workflows_ready"))
        if task.kind == "command_doc":
            if ok and isinstance(val, InspectorDocument):
                if hasattr(state, "io") and state.io is not None:
                    state.io.submit(PublishInspector(document=val), tag="frontend")
                emit(ProjectionRequested())
            else:
                emit(EmitNotice(str(val or "Workflow command failed."), level="warning"))
                emit(ProjectionRequested())
            return
        if task.kind == "command":
            if ok and isinstance(val, dict) and val.get("_notice_only"):
                emit(EmitNotice(str(val["text"]), level="info"))
                emit(ProjectionRequested())
                return
            result = (
                {"ok": True, "text": str(val or "")}
                if ok
                else {"ok": False, "text": f"Workflow {task.workflow_name}: {val}"}
            )
            WorkflowCommand.complete(state, result, emit)
            self.drain_notices(state, emit, wake=False)
            self.ensure_tick_timer(state)
            return
        if task.kind == "worker_remind":
            if ok and isinstance(val, dict) and val.get("action") == "remind":
                pending = getattr(state, "pending_interrupts", None)
                if not isinstance(pending, list):
                    pending = []
                    state.pending_interrupts = pending
                pending.append(
                    "You ended your turn without calling finish_task. If your task is done, call finish_task now "
                    "with your summary and outputs. If it is not done, keep working."
                )
                emit(InterruptQueued(source="workflow_reminder"))
                emit(ProjectionRequested())
                emit(PacingCheck(reason="workflow_reminder"))
            elif not ok:
                LOG.exception("Workflow finish reminder failed: %s", val)
            return
        if task.kind == "tick":
            rt.tick_active = False
            if not ok:
                LOG.exception("Workflow scheduler tick failed: %s", val)
        elif not ok:
            LOG.warning("Workflow %s task failed: %s", task.kind, val)
        self.drain_notices(state, emit, wake=(task.kind != "startup"))
        if task.kind == "startup" and getattr(state, "pending_interrupts", None):
            emit(InterruptQueued(source="workflow_notices"))
        self.ensure_tick_timer(state)

    def drain_notices(self, state: State, emit: EmitFn, *, wake: bool) -> int:
        rt = _wf_runtime(state)
        drained = 0
        while rt.notices:
            pending = getattr(state, "pending_interrupts", None)
            if not isinstance(pending, list):
                pending = []
                state.pending_interrupts = pending
            pending.append(rt.notices.popleft())
            drained += 1
        if drained and wake:
            emit(InterruptQueued(source="workflow_notices"))
            emit(ProjectionRequested())
            emit(PacingCheck(reason="workflow_notices"))
        return drained

    # -- worker context (pure in-memory read) -----------------------------------

    def worker_context(self, state) -> dict:
        identity = worker_identity(state)
        rt = _wf_runtime(state)
        cached = rt.cached_worker_context
        if cached is not None and (not identity or cached.get("id") == identity["agent"]):
            return cached
        if not identity:
            return {"id": "", "workflow": "", "group": "", "sends_to": [], "items": [], "state": "unknown"}
        return {
            "id": identity["agent"],
            "workflow": identity["workflow"],
            "group": "",
            "sends_to": [],
            "items": [],
            "state": "unknown",
        }

    # -- controller handlers (all pure Sans-I/O generators) ---------------------

    def create_workflow(self, state, workflow_name, definition_buffer):
        self.controller.bind(state)
        wf_name = check_workflow_name(workflow_name)
        text = buffer_text(state, definition_buffer)
        check_definition_models(parse_definition(split_version(text)[1]), self.controller.config)
        res = yield WorkflowStoreRequest(
            op="create_workflow",
            workflow_name=wf_name,
            params={
                **self.controller._snap_params(),
                "text": text,
                "by": self.controller.me_label(),
                "cwd": get_session_cwd(state),
                "config": self.controller.config,
            },
        )
        _apply_snapshot(state, res.get("snapshot"))
        return res["text"]

    def update_workflow(self, state, workflow_name, definition_buffer, reason, preview_only=False):
        self.controller.bind(state)
        wf_name = check_workflow_name(workflow_name)
        if not str(reason or "").strip():
            raise WorkflowError("reason is required: one sentence saying why the definition changed")
        text = buffer_text(state, definition_buffer)
        check_definition_models(parse_definition(split_version(text)[1]), self.controller.config)
        res = yield WorkflowStoreRequest(
            op="update_workflow",
            workflow_name=wf_name,
            params={
                **self.controller._snap_params(),
                "text": text,
                "reason": str(reason).strip(),
                "by": self.controller.me_label(),
                "preview_only": bool(preview_only),
                "config": self.controller.config,
            },
        )
        _apply_snapshot(state, res.get("snapshot"))
        return res["text"]

    def command_workflow(self, state, workflow_name, command, agent_group=None):
        self.controller.bind(state)
        wf_name = check_workflow_name(workflow_name)
        command = str(command or "").strip().lower()
        agent_group = None if agent_group in (None, "") else agent_group
        if command == "run":
            return (yield from self.controller.run(wf_name, agent_group))
        if command == "pause":
            return (yield from self.controller.pause(wf_name, agent_group))
        if command == "archive":
            if agent_group:
                raise WorkflowError("archive applies to the whole workflow; omit agent_group")
            return (yield from self.controller.archive(wf_name))
        if command == "unarchive":
            if agent_group:
                raise WorkflowError("unarchive applies to the whole workflow; omit agent_group")
            return (yield from self.controller.unarchive(wf_name))
        raise WorkflowError(f'unknown command "{command}"; use run, pause, archive or unarchive')

    def rename_workflow(self, state, workflow_name, new_name):
        self.controller.bind(state)
        return (yield from self.controller.rename(check_workflow_name(workflow_name), check_workflow_name(new_name)))

    def set_agent_count(self, state, workflow_name, agent_group, count):
        self.controller.bind(state)
        wf_name = check_workflow_name(workflow_name)
        res = yield WorkflowStoreRequest(
            op="set_count",
            workflow_name=wf_name,
            params={
                **self.controller._snap_params(),
                "agent_group": agent_group,
                "count": int(count),
                "by": self.controller.me_label(),
            },
        )
        _apply_snapshot(state, res.get("snapshot"))
        result = res["result"]
        if not result["changed"]:
            return f"{agent_group} already has count {count}."
        text = f"Set {agent_group} count from {result['old']} to {count} (def_{result['version']})."
        if int(count) < result["old"]:
            text += " Extra running agents finish their current item first."
        elif wf_name in self.controller.managed:
            started = yield from self.controller.tick(wf_name)
            if started:
                text += " " + " ".join(started)
        elif res["control"].get("state") == "running":
            text += " The controlling session will start agents shortly."
        return text

    def message_agents(self, state, workflow_name, message, agent_group=None):
        self.controller.bind(state)
        agent_group = None if agent_group in (None, "") else agent_group
        return (yield from self.controller.message_agents(check_workflow_name(workflow_name), str(message), agent_group))

    def restart_agent(self, state, workflow_name, agent):
        self.controller.bind(state)
        return (yield from self.controller.restart_agent(check_workflow_name(workflow_name), agent))

    def add_items(self, state, workflow_name, queue, from_buffers, note=None):
        self.controller.bind(state)
        wf_name = check_workflow_name(workflow_name)
        check_queue_id(queue)
        buffers = [from_buffers] if isinstance(from_buffers, str) else list(from_buffers or [])
        if not buffers:
            raise WorkflowError("from_buffers is empty")
        texts = [buffer_text(state, b) for b in buffers]
        res = yield WorkflowStoreRequest(
            op="add_items",
            workflow_name=wf_name,
            params={**self.controller._snap_params(), "queue": queue, "texts": texts, "note": note},
        )
        _apply_snapshot(state, res.get("snapshot"))
        return res["text"]

    def move_items(self, state, workflow_name, item_ids, to_queue, note=None):
        self.controller.bind(state)
        wf_name = check_workflow_name(workflow_name)
        check_queue_id(to_queue)
        ids = [item_ids] if isinstance(item_ids, str) else list(item_ids or [])
        res = yield WorkflowStoreRequest(
            op="move_items",
            workflow_name=wf_name,
            params={**self.controller._snap_params(), "item_ids": ids, "to_queue": to_queue, "note": note},
        )
        _apply_snapshot(state, res.get("snapshot"))
        return res["text"]

    # -- worker handlers (pure Sans-I/O generators) -----------------------------

    def finish_task(self, state, summary, outputs=None):
        identity = worker_identity(state)
        if not identity:
            raise WorkflowError("finish_task is only available to workflow agents")
        pairs = []
        for entry in outputs or []:
            if not isinstance(entry, dict) or "buffer" not in entry or "queue" not in entry:
                raise WorkflowError('each output must look like {"buffer": "buf_...", "queue": "q_..."}')
            pairs.append((check_queue_id(entry["queue"]), buffer_text(state, entry["buffer"])))
        res = yield WorkflowStoreRequest(
            op="finish_agent",
            workflow_name=identity["workflow"],
            params={
                "agent_id": identity["agent"],
                "summary": str(summary or ""),
                "outputs": pairs,
                "viewer_session_id": str(_session_meta(state).session_id or ""),
            },
        )
        _apply_snapshot(state, res.get("snapshot"))
        state.result = str(summary or "")
        return res["text"]

    def worker_add_items(self, state, queue, from_buffers, note=None):
        identity = worker_identity(state)
        if not identity:
            raise WorkflowError("worker_add_items is only available to workflow agents")
        check_queue_id(queue)
        buffers = [from_buffers] if isinstance(from_buffers, str) else list(from_buffers or [])
        if not buffers:
            raise WorkflowError("from_buffers is empty")
        texts = [buffer_text(state, b) for b in buffers]
        res = yield WorkflowStoreRequest(
            op="add_items",
            workflow_name=identity["workflow"],
            params={
                "queue": queue,
                "texts": texts,
                "note": note,
                "worker_agent": identity["agent"],
                "viewer_session_id": str(_session_meta(state).session_id or ""),
            },
        )
        _apply_snapshot(state, res.get("snapshot"))
        return res["text"]

    def worker_move_items(self, state, item_ids, to_queue, note=None):
        identity = worker_identity(state)
        if not identity:
            raise WorkflowError("worker_move_items is only available to workflow agents")
        check_queue_id(to_queue)
        ids = [item_ids] if isinstance(item_ids, str) else list(item_ids or [])
        res = yield WorkflowStoreRequest(
            op="move_items",
            workflow_name=identity["workflow"],
            params={
                "item_ids": ids,
                "to_queue": to_queue,
                "note": note,
                "worker_agent": identity["agent"],
                "viewer_session_id": str(_session_meta(state).session_id or ""),
            },
        )
        _apply_snapshot(state, res.get("snapshot"))
        return res["text"]


def build_tools(plugin: WorkflowsPlugin) -> list[Tool]:
    outputs_property = {
        "type": "array",
        "description": "Result entries [{buffer, queue}]; each buffer becomes one item.",
        "items": {"type": "object", "properties": {
            "buffer": text_property("Result buffer ID."),
            "queue": text_property("Destination queue."),
        }, "required": ["buffer", "queue"]},
    }

    def allowed_text(context) -> str:
        queues = list(context.get("sends_to") or []) + [TRASH]
        return "You can send to: " + ", ".join(queues) + "."

    def finish_schema(context):
        return function_schema(
            "finish_task",
            "Finish your task and end this session; taken items are marked done. " + allowed_text(context),
            {"summary": text_property("Short summary of what you did and found."),
             "outputs": outputs_property},
            ["summary"])

    def worker_add_schema(context):
        return function_schema(
            "add_items",
            "Add buffer contents as items to a queue before finishing. " + allowed_text(context),
            {"queue": text_property("Target queue (q_*)."),
             "from_buffers": {"type": "array", "items": {"type": "string"},
                              "description": "Buffer IDs (one item each)."},
             "note": NOTE_PROPERTY},
            ["queue", "from_buffers"])

    def worker_move_schema(context):
        return function_schema(
            "move_items",
            "Move waiting or finished items to another queue (not in-progress items). " + allowed_text(context),
            {"item_ids": {"type": "array", "items": {"type": "string"}, "description": "Item IDs (item_*)."},
             "to_queue": text_property("Destination queue."),
             "note": NOTE_PROPERTY},
            ["item_ids", "to_queue"])

    create = function_schema(
        "create_workflow",
        "Create a paused workflow from a YAML definition buffer (queues q_*, groups grp_* with takes_from or standing count). Activate the Workflows skill for full YAML syntax.",
        {"workflow_name": text_property("New name (lowercase letters, digits, underscores)."),
         "definition_buffer": text_property("Buffer with YAML definition.")},
        ["workflow_name", "definition_buffer"])
    update = function_schema(
        "update_workflow",
        "Replace a workflow's YAML definition from a buffer (copy wf:<name>:definition first). Returns diff and effects; running agents keep the old definition.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY,
         "definition_buffer": text_property("Buffer with complete new YAML."),
         "reason": text_property("One-sentence reason for history."),
         "preview_only": {"type": "boolean", "description": "Show diff and effects without saving."}},
        ["workflow_name", "definition_buffer", "reason"])
    command = function_schema(
        "command_workflow",
        "Control a workflow: run (confirm with user first; agents cost money), pause, archive, or unarchive.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY,
         "command": {"type": "string", "enum": ["run", "pause", "archive", "unarchive"], "description": "Action."},
         "agent_group": AGENT_GROUP_PROPERTY},
        ["workflow_name", "command"])
    rename = function_schema(
        "rename_workflow",
        "Rename a paused or archived workflow that has no unfinished agents.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY,
         "new_name": text_property("New workflow name.")},
        ["workflow_name", "new_name"])
    count = function_schema(
        "set_agent_count",
        "Set concurrent agent count (0-64) for a group; records a new version immediately.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY,
         "agent_group": text_property("Agent group (grp_*)."),
         "count": {"type": "integer", "description": "Concurrent agents (0-64)."}},
        ["workflow_name", "agent_group", "count"])
    message = function_schema(
        "message_agents",
        "Send a message to a workflow's running agents.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY, "message": text_property("Message text."),
         "agent_group": AGENT_GROUP_PROPERTY},
        ["workflow_name", "message"])
    restart = function_schema(
        "restart_agent",
        "Stop a stuck agent, discard its progress, and return its items to their queues.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY, "agent": text_property("Agent ID (agt_*).")},
        ["workflow_name", "agent"])
    add = function_schema(
        "add_items",
        "Add buffer contents as items to a workflow queue (first line becomes title).",
        {"workflow_name": WORKFLOW_NAME_PROPERTY, "queue": text_property("Target queue (q_*)."),
         "from_buffers": {"type": "array", "items": {"type": "string"},
                          "description": "Buffer IDs (one item each)."},
         "note": NOTE_PROPERTY},
        ["workflow_name", "queue", "from_buffers"])
    move = function_schema(
        "move_items",
        "Move waiting or finished items to another queue (retry, reroute, or discard to q_trash). Use restart_agent first for taken items.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY,
         "item_ids": {"type": "array", "items": {"type": "string"}, "description": "Item IDs (item_*)."},
         "to_queue": text_property("Destination queue (q_* or q_trash)."),
         "note": NOTE_PROPERTY},
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
# Slash command (100% pure Sans-I/O)
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


def parse_command(raw_args: str, *, known_names: Iterable[str] = ()) -> dict:
    """Pure command parser — never touches the filesystem."""
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
    if len(tokens) == 1 and tokens[0].partition(":")[0] in set(known_names):
        raise WorkflowError(f'unknown command "{tokens[0]}"; to open a workflow, use /workflow view {tokens[0]}')
    raise WorkflowError(f'unknown command "{tokens[0]}"; use one of: {COMMAND_VERBS}. Usage: {USAGE}')


class WorkflowCommand:
    """Pure Sans-I/O ``/workflow`` slash command submitting typed ``WorkflowStoreRequest``s."""

    def __init__(self, plugin: WorkflowsPlugin):
        self.plugin = plugin

    @staticmethod
    def document(title: str, body: str, subtitle: str = "") -> CommandResult:
        return CommandResult(outputs=(InspectorDocument(title=title, subtitle=subtitle, body=body,
                                                        format="markdown"),))

    def _doc_gen(self, state, request: dict):
        controller = self.plugin.controller
        res = yield WorkflowStoreRequest(
            op="render_command_doc",
            workflow_name=request.get("name", ""),
            params={
                **controller._snap_params(),
                "action": request["action"],
                "kind": request.get("kind"),
                "target": request.get("target"),
            },
        )
        _apply_snapshot(state, res.get("snapshot"))
        return InspectorDocument(
            title=res["title"],
            subtitle=res.get("subtitle", ""),
            body=res["body"],
            format="markdown",
        )

    def _set_gen(self, request: dict):
        controller = self.plugin.controller
        text = yield from controller.change_setting(
            request["name"], request["kind"], request["target"], request["setting"], request["value"]
        )
        if text.endswith("nothing changed."):
            return {"_notice_only": True, "text": text}
        return text

    def __call__(self, ctx, *, raw_args=""):
        controller = self.plugin.controller
        rt = _wf_runtime(ctx.state)
        try:
            request = parse_command(raw_args, known_names=rt.cached_names)
            if worker_identity(ctx.state):
                raise WorkflowError("workflow agents cannot control workflows")
            controller.bind(ctx.state)
            if request["action"] == "run":
                controller.require_launch()
        except WorkflowError as exc:
            raise CommandError(str(exc)) from exc

        action = request["action"]
        if action in {"list", "archived", "show", "settings"}:
            gen = self._doc_gen(ctx.state, request)
            task = _WorkflowTask(kind="command_doc", gen=gen, workflow_name=request.get("name", ""))
            op_id, ok, val = self.plugin.advance_task(ctx.state, task, first=True)
            if op_id is not None:
                receipt = (
                    "Loading workflows..." if action in {"list", "archived"}
                    else f"Loading {request.get('name', 'workflow')}..."
                )
                return CommandResult(
                    outputs=(CommandNotice(text=receipt, level="info"),),
                    operation_id=op_id,
                    refresh=False,
                )
            if ok and isinstance(val, InspectorDocument):
                return CommandResult(outputs=(val,), refresh=False)
            raise CommandError(str(val or "Workflow view failed"))

        if action == "set":
            label = request["name"] if request["kind"] == "workflow" else f"{request['name']}:{request['target']}"
            gen = self._set_gen(request)
            receipt = f"Setting {label} {request['setting']} to {request['value']}..."
        elif action == "rename":
            gen = controller.rename(request["name"], request["new_name"])
            receipt = f"Renaming {request['name']} to {request['new_name']}..."
        elif action == "archive":
            gen = controller.archive(request["name"])
            receipt = f"Archiving {request['name']}..."
        elif action == "unarchive":
            gen = controller.unarchive(request["name"])
            receipt = f"Restoring {request['name']}..."
        else:
            method = controller.run if action == "run" else controller.pause
            gen = method(request["name"], request["group"])
            verb = "Running" if action == "run" else "Pausing"
            scope = f"{request['name']}:{request['group']}" if request["group"] else request["name"]
            receipt = f"{verb} {scope}..."

        task = _WorkflowTask(kind="command", gen=gen, workflow_name=request["name"])
        op_id, ok, val = self.plugin.advance_task(ctx.state, task, first=True)
        if op_id is not None:
            return CommandResult(
                outputs=(CommandNotice(text=receipt, level="info"),),
                operation_id=op_id,
                refresh=False,
            )
        result = (
            {"ok": True, "text": str(val or "")}
            if ok
            else {"ok": False, "text": f"Workflow {request['name']}: {val}"}
        )
        self.complete(ctx.state, result, ctx.emit, emit_notice=False)
        self.plugin.drain_notices(ctx.state, ctx.emit, wake=False)
        self.plugin.ensure_tick_timer(ctx.state)
        text = str(result.get("text") or "")
        first_line = text.splitlines()[0] if text else "Workflow command completed."
        return CommandResult(
            outputs=(CommandNotice(text=first_line, level="info" if result["ok"] else "warning"),),
            refresh=False,
        )

    @staticmethod
    def complete(state: State, result: dict, emit: EmitFn, *, emit_notice: bool = True) -> None:
        if result.get("ok"):
            text = str(result.get("text") or "")
            pending = getattr(state, "pending_interrupts", None)
            if not isinstance(pending, list):
                pending = []
                state.pending_interrupts = pending
            pending.append(
                "The user typed a /workflow command. Result (already done; do not repeat it):\n" + text
            )
            emit(InterruptQueued(source="workflow_command"))
            first_line = text.splitlines()[0] if text else "Workflow command completed."
            if emit_notice:
                emit(EmitNotice(first_line, level="info"))
            emit(ProjectionRequested())
            emit(PacingCheck(reason="workflow_command"))
        else:
            if emit_notice:
                emit(EmitNotice(str(result.get("text") or "Workflow command failed."), level="warning"))
            emit(ProjectionRequested())


# ---------------------------------------------------------------------------
# Buffers, prompts, notices (100% pure Sans-I/O)
# ---------------------------------------------------------------------------

class WorkflowBuffers:
    """Pure in-memory ``wf:`` readonly buffer provider backed by cached ``WorkflowStoreDriver`` snapshots."""

    handles = (SessionReady, SessionRestored, RenderRequested, TurnSettled)
    reads = {"buffer_manager"}
    writes = {"buffer_manager"}
    optional_reads = {"_agent_zoo_context"}

    def __init__(self, plugin: "WorkflowsPlugin | None" = None):
        self.plugin = plugin

    def register(self, state) -> None:
        manager = getattr(state, "buffer_manager", None)
        if manager is not None:
            manager.register_special_buffer_namespace("wf:", self.render, replace=True)

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        del event, emit
        self.register(state)

    def render(self, state, buffer_id):
        if not buffer_id.startswith("wf:"):
            return None
        parts = buffer_id.split(":")
        if len(parts) not in (2, 3):
            return None
        rt = _wf_runtime(state)
        text = rt.cached_views.get(buffer_id)
        if text is None:
            if buffer_id == "wf:index":
                text = DEFAULT_EMPTY_INDEX
            else:
                wf_name = check_workflow_name(parts[1])
                if wf_name not in set(rt.cached_names):
                    known = (
                        f"; existing workflows: {', '.join(rt.cached_names)}"
                        if rt.cached_names else "; there are no workflows yet"
                    )
                    raise WorkflowError(f'there is no workflow named "{wf_name}"{known}')
                if len(parts) == 3:
                    view = parts[2]
                    if view not in {"definition", "history", "log"} and not (
                        VERSION_ID.fullmatch(view)
                        or QUEUE_ID.fullmatch(view)
                        or GROUP_ID.fullmatch(view)
                        or AGENT_ID.fullmatch(view)
                        or ITEM_ID.fullmatch(view)
                    ):
                        raise WorkflowError(
                            f'unknown view "{view}"; use definition, history, log, def_N, q_*, grp_*, agt_* or item_N'
                        )
                raise WorkflowError(f"view {buffer_id} is not loaded yet; retry shortly")
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
    """Durable role guidance on the system entry for workers and controllers (pure Sans-I/O)."""

    handles = (RenderRequested,)
    reads = {"entries"}
    optional_reads = set()
    transform_name = "workflow role guidance"

    def __init__(self, plugin: WorkflowsPlugin):
        self.plugin = plugin

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        del emit
        assert isinstance(event, RenderRequested)
        if event.channels is not None and MODEL_RENDER_CHANNEL not in event.channels:
            return
        if not state.entries:
            return
        entry = state.entries[0]
        if self.transform_name not in entry.render_transform_names(MODEL_RENDER_CHANNEL, pending=True):
            entry.append_render_transform(MODEL_RENDER_CHANNEL, self.transform, name=self.transform_name)

    def transform(self, messages, state):
        if not messages or messages[0].get("role") != "system":
            return messages
        identity = worker_identity(state)
        managed = _wf_durable(state).managed
        if identity:
            context = self.plugin.worker_context(state)
            block = WORKER_GUIDANCE.format(
                agent=identity["agent"],
                workflow=identity["workflow"],
                group=context.get("group") or "unknown",
                queues=", ".join(list(context.get("sends_to") or []) + [TRASH]),
            )
        elif managed:
            names = ", ".join(sorted(managed))
            block = (
                f"# Workflows\nThis session controls running workflows: {names}. "
                'Activate skill(name="Workflows") before changing them; read view(buffer="wf:<name>") '
                "for their state."
            )
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


def _preload_workflow_skill() -> tuple[str | None, str | None]:
    """Read the bundled/installed Workflows SKILL.md ONCE at build/init time."""
    relative = Path("skills/workflows/SKILL.md")
    installed = Path(resolve_state_root()) / "plugin-configs" / PLUGIN_NAME / relative
    bundled = Path(__file__).resolve().parent.parent / relative
    package = bundled if bundled.is_file() else installed
    if not package.is_file():
        return None, None
    try:
        return str(package), package.read_text(encoding="utf-8")
    except Exception:
        LOG.warning("Could not preload bundled Workflows skill text", exc_info=True)
        return str(package), None


class WorkflowSkillRegistration(RenderTransformRegistrar):
    """Offer the preloaded bundled Workflows skill to controlling sessions only (pure Sans-I/O)."""

    handles = (SessionReady, SessionRestored, RenderRequested)
    optional_reads = {"skill_registry"}
    writes = {"skill_registry"}

    def __init__(self, *, skill_path: str | None = None, skill_text: str | None = None):
        if skill_path is None and skill_text is None:
            skill_path, skill_text = _preload_workflow_skill()
        self._skill_path = skill_path
        self._skill_text = skill_text
        self.registry = None

    def apply(self, state) -> None:
        registry = getattr(state, "skill_registry", None)
        if registry is None or registry is self.registry:
            return
        if worker_identity(state):
            existing = registry.get("Workflows") if callable(getattr(registry, "get", None)) else None
            if existing is not None:
                if callable(getattr(registry, "add_exclusions", None)):
                    registry.add_exclusions(existing.root)
                else:
                    file_skills = getattr(registry, "_file_skills", None)
                    if isinstance(file_skills, list):
                        registry._file_skills = [
                            s for s in file_skills if s.name.lower() != "workflows"
                        ]
                    dynamic_skills = getattr(registry, "_dynamic_skills", None)
                    if isinstance(dynamic_skills, dict):
                        for key in [k for k in dynamic_skills if k.lower() == "workflows"]:
                            dynamic_skills.pop(key, None)
                    sources = getattr(registry, "_sources", None)
                    if isinstance(sources, list):
                        registry._sources = [
                            s for s in sources if getattr(s, "kind", "") != PLUGIN_NAME
                        ]
                    rebuild = getattr(registry, "_rebuild", None)
                    if callable(rebuild):
                        rebuild()
        elif self._skill_path:
            source = SkillSource(
                self._skill_path, kind=PLUGIN_NAME, scope="bundled",
                precedence=-5000, native=True, always_list=True,
            )
            if callable(getattr(registry, "add_sources", None)):
                registry.add_sources([source])
            if self._skill_text is not None and callable(getattr(registry, "register_skill_text", None)):
                try:
                    registry.register_skill_text(
                        self._skill_path,
                        self._skill_text,
                        source=source,
                    )
                except Exception:
                    LOG.warning("Could not register bundled Workflows skill text", exc_info=True)
        self.registry = registry

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        del event, emit
        self.apply(state)


class WorkflowNotices(InterruptCheck):
    """Pure Sans-I/O scheduler & notice hook: timers, session events, adoption, restore, worker reminders."""

    check_name = "workflow_notices"
    cooldown = 0
    handles = (
        SessionReady,
        SessionRestored,
        TurnSettled,
        TimerFired,
        IoCompleted,
        IoFailed,
        IoReattached,
        IoLost,
        ExternalSessionReady,
        ExternalSessionExited,
        SessionDataReceived,
    )
    reads = {"step"}
    optional_reads = {
        "_agent_zoo_context", "_persistence_local_root", "cwd", "session_cwd", "entries",
        "buffer_manager", "done", "shutdown", "activity", "pending_tool_calls",
        "active_llm", "startup_paused", "interrupted",
    }
    writes = {"pending_interrupts", "_interrupt_state", "buffer_manager"}

    def __init__(self, plugin: WorkflowsPlugin):
        self.plugin = plugin

    def _tick_all_managed(self, state: State):
        controller = self.plugin.controller
        controller.bind(state)
        for name in list(controller.managed):
            try:
                yield from controller.tick(name)
                controller.errors.pop(name, None)
            except Exception as exc:
                LOG.exception("Workflow %s scheduling failed", name)
                message = f"Workflow {name}: scheduling failed: {exc}"
                if controller.errors.get(name) != message:
                    controller.errors[name] = message
                    controller.notice(message)

    def _startup_controller_gen(self, state: State):
        controller = self.plugin.controller
        controller.bind(state)
        adopted: list[str] = []
        if _session_meta(state).instance_id:
            adopted = yield from controller.adopt(state)
        else:
            snap = yield WorkflowStoreRequest(op="snapshot", params=controller._snap_params())
            _apply_snapshot(state, snap)
        if adopted:
            yield from controller.rediscover_workflows(adopted)

    def _startup_worker_gen(self, state: State, identity):
        snap = yield WorkflowStoreRequest(
            op="snapshot",
            params={
                "viewer_session_id": str(_session_meta(state).session_id or ""),
                "worker_workflow": identity["workflow"],
                "worker_agent": identity["agent"],
            },
        )
        _apply_snapshot(state, snap)

    def _worker_remind_gen(self, state: State, identity):
        res = yield WorkflowStoreRequest(
            op="worker_remind",
            workflow_name=identity["workflow"],
            params={"agent_id": identity["agent"]},
        )
        if res.get("worker_context") is not None:
            _wf_runtime(state).cached_worker_context = dict(res["worker_context"])
        return res

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        rt = _wf_runtime(state)

        if isinstance(event, (IoCompleted, IoFailed)):
            task = rt.pending_tasks.pop(event.op_id, None)
            if task is None:
                return
            event.stop()
            io_table = getattr(state, "io", None)
            manual_complete = io_table is not None and io_table.is_active(event.op_id)
            if manual_complete:
                io_table.complete(event.op_id, defer_readiness_release=True)
            try:
                if isinstance(event, IoCompleted):
                    op_id, ok, val = self.plugin.advance_task(state, task, send_value=event.value)
                else:
                    err = event.error
                    exc = err if isinstance(err, BaseException) else RuntimeError(str(err))
                    op_id, ok, val = self.plugin.advance_task(state, task, throw_exc=exc)
                if op_id is None:
                    self.plugin.finish_task_outcome(state, emit, task, ok, val)
            finally:
                if manual_complete:
                    io_table.flush_deferred_readiness_releases()
            return

        if isinstance(event, IoReattached):
            meta = event.handle if isinstance(event.handle, dict) else {}
            wf_name = str(meta.get("workflow") or "")
            agent_id = str(meta.get("agent_id") or "")
            iid = str(meta.get("instance_id") or "")
            if wf_name and agent_id and iid:
                rt.instances_by_id[iid] = (wf_name, agent_id)
            self.plugin.ensure_tick_timer(state)
            return

        if isinstance(event, IoLost):
            task = rt.pending_tasks.pop(event.op_id, None)
            if task is not None:
                event.stop()
                io_table = getattr(state, "io", None)
                if io_table is not None and io_table.is_active(event.op_id):
                    io_table.complete(event.op_id)
                self.plugin.finish_task_outcome(
                    state, emit, task, False, RuntimeError(f"Workflow operation lost: {event.reason}")
                )
                return
            if rt.tick_timer_op is not None and event.op_id == rt.tick_timer_op:
                event.stop()
                rt.tick_timer_op = None
                self.plugin.ensure_tick_timer(state)
            return

        if isinstance(event, TimerFired):
            if rt.tick_timer_op is None or event.op_id != rt.tick_timer_op:
                return
            event.stop()
            rt.tick_timer_op = None
            if (
                worker_identity(state)
                or not _wf_durable(state).managed
                or getattr(state, "shutdown", False)
                or getattr(state, "done", False)
            ):
                return
            rt.tick_active = True
            self.plugin.start_event_task(state, emit, "tick", self._tick_all_managed(state))
            return

        if isinstance(event, ExternalSessionReady):
            if worker_identity(state) or not _wf_durable(state).managed:
                return
            self.plugin.controller.bind(state)
            self.plugin.start_event_task(state, emit, "event", self.plugin.controller.on_session_ready(event))
            return

        if isinstance(event, ExternalSessionExited):
            if worker_identity(state) or not _wf_durable(state).managed:
                return
            self.plugin.controller.bind(state)
            self.plugin.start_event_task(state, emit, "event", self.plugin.controller.on_session_exited(event))
            return

        if isinstance(event, SessionDataReceived):
            if not worker_identity(state) or event.channel != "workflow":
                return
            parts = [event.message] if event.message else []
            for art in event.artifacts:
                body = art.data if isinstance(art.data, str) else json.dumps(art.data, ensure_ascii=False)
                parts.append(f"[{art.name} ({art.kind})]: {body}")
            if parts:
                self._ensure_interrupt_attrs(state)
                state.pending_interrupts.append("\n".join(parts))
                emit(InterruptQueued(source="workflow_data"))
                emit(ProjectionRequested())
                emit(PacingCheck(reason="workflow_data"))
            return

        if getattr(state, "buffer_manager", None) is not None:
            WorkflowBuffers(self.plugin).register(state)

        identity = worker_identity(state)
        if identity:
            if isinstance(event, (SessionReady, SessionRestored)):
                entries = getattr(state, "entries", None) or []
                if entries and rt.worker_reminded_entry < 0:
                    rt.worker_reminded_entry = len(entries) - 1
                if hasattr(state, "io") and state.io is not None:
                    self.plugin.start_event_task(
                        state, emit, "startup",
                        self._startup_worker_gen(state, identity),
                        hold_readiness="workflows",
                    )
            elif isinstance(event, TurnSettled):
                self._schedule_worker_reminder(state, emit, identity)
            return

        if isinstance(event, (SessionReady, SessionRestored)):
            self.plugin.controller.bind(state)
            if hasattr(state, "io") and state.io is not None:
                is_restore = isinstance(event, SessionRestored) or bool(getattr(event, "restored", False))
                self.plugin.start_event_task(
                    state, emit, "startup",
                    self._startup_controller_gen(state),
                    hold_readiness="workflows" if is_restore else None,
                )
            else:
                self.plugin.drain_notices(state, emit, wake=False)
            self.plugin.ensure_tick_timer(state)
            return

        # TurnSettled for controller
        self.plugin.controller.bind(state)
        self.plugin.drain_notices(state, emit, wake=False)
        self.plugin.ensure_tick_timer(state)

    def _schedule_worker_reminder(self, state: State, emit: EmitFn, identity) -> None:
        """A worker that ends its turn without finish_task gets one reminder, then needs attention."""
        if getattr(state, "done", False) or getattr(state, "shutdown", False):
            return
        entries = getattr(state, "entries", None) or []
        if not entries:
            return
        last = entries[-1]
        index = len(entries) - 1
        rt = _wf_runtime(state)
        if rt.worker_reminded_entry < 0:
            rt.worker_reminded_entry = index
            return
        if getattr(last, "role", None) != "assistant" or getattr(last, "tool_calls", None):
            return
        if index <= rt.worker_reminded_entry:
            return
        rt.worker_reminded_entry = index
        if hasattr(state, "io") and state.io is not None:
            self.plugin.start_event_task(
                state, emit, "worker_remind", self._worker_remind_gen(state, identity)
            )


# ---------------------------------------------------------------------------
# Registration (zero module-global controller dicts!)
# ---------------------------------------------------------------------------

def register_features(builder, *, session, config):
    state = getattr(session, "state", None)
    controller = WorkflowController(config, session)
    plugin = WorkflowsPlugin(controller)
    if worker_identity(state) and builder.has("rlm"):
        feature = builder._features["rlm"]
        builder.add(replace(feature, components=[c for c in feature.components if not isinstance(c, Tool)]))
    tools = build_tools(plugin)
    builder.add(Feature(
        "workflows",
        components=[
            WorkflowBuffers(plugin),
            WorkflowNotices(plugin),
            *tools,
            WorkflowSchemaPublisher(tools),
            Command(
                "/workflow",
                "List, show, run, pause, rename, archive or change workflows",
                WorkflowCommand(plugin),
                raw=True,
                usage=USAGE,
                section="Workflows",
            ),
        ],
        order=[
            RegisterSpecialBuffers,
            WorkflowBuffers,
            Tool,
            SchemaPublisher,
            WorkflowSchemaPublisher,
            {RenderPlanInitializer, LLMCaller},
        ],
        drivers=[WorkflowStoreDriver],
        durable_state=WORKFLOWS_DURABLE_SPEC,
    ))
    builder.add(Feature(
        "workflow_guidance",
        components=[
            WorkflowSkillRegistration(),
            WorkflowSystemPrompt(plugin),
        ],
        order=[WorkflowSkillRegistration, WorkflowSystemPrompt, SystemPromptSkillList, MessageRenderer],
    ))
