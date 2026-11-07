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
from agents.system_prompt import SystemPromptSkillList
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
        self.handles: dict[str, object] = {}
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
                "default_model": str(((self.config or {}).get("llm", {}) or {}).get("default") or ""),
                "cwd": get_session_cwd(state),
            }

    def me_label(self) -> str:
        return self.me["session_id"] if self.me else "unknown session"

    def require_launch(self) -> None:
        if not self.me:
            raise WorkflowError("this session has no runtime identity yet; try again in a moment")
        if self.launch is None or self.launcher is None:
            raise WorkflowError("this session cannot launch agents (it has no project or runtime launcher)")

    def model_for(self, definition: dict, group: dict) -> str:
        model = group.get("model") or definition.get("model") or self.launch["default_model"]
        models = ((self.config or {}).get("llm", {}) or {}).get("models", {}) or {}
        if not model:
            raise WorkflowError("no model is configured; set model: in the definition")
        if models and model not in models:
            raise WorkflowError(f'unknown model "{model}"; configured models: {", ".join(sorted(models))}')
        return model

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
                raise WorkflowError(f"{store.name} is archived and cannot run again")
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
            self.pause(name)
            ended = []
            for agent in store.agents():
                if agent["state"] not in ACTIVE_STATES:
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
            text = f"Archived {name}. Its definition, items and history stay readable at wf:{name}."
            if ended:
                text += f" Shut down {', '.join(ended)}."
            return text

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
        handle = self.handles.get(agent["id"])
        if handle is not None and getattr(handle, "ref", None) is not None \
                and handle.ref.instance_id == agent.get("instance_id"):
            status = handle.poll().status
            if status in {"exited"}:
                return "dead"
            if status in {"starting", "ready"}:
                return "alive"
        return process_evidence(agent.get("process_identity"))

    def launch_agent(self, store: WorkflowStore, agent: dict, message: str, checkpoint=None) -> None:
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
                           process_identity=None, release="resume" if checkpoint else None)
        try:
            handle = self.launcher.start(plan)
        except Exception:
            store.update_agent(agent["id"], process_identity=None, problem="launch did not complete")
            raise
        self.handles[agent["id"]] = handle
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
        try:
            self.launch_agent(store, agent, "", checkpoint=CheckpointRef(agent["session_id"], saved.commit_id))
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
                    self.send(store, agent, "continue")
                    store.update_agent(agent["id"], state="running")
                    resumed.append(agent["id"])
                except Exception as exc:
                    problems.append(f"Could not resume {agent['id']}: {exc}")
            elif alive == "alive":
                store.update_agent(agent["id"], state="starting")
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
        for agent_id in list(self.handles):
            if not store.agent_path(agent_id).exists() or store.agent(agent_id)["state"] not in ACTIVE_STATES:
                self.handles.pop(agent_id, None)
        for agent in store.agents():
            if agent["state"] in {"starting", "running"}:
                self.observe(store, agent, definition, running and agent["group"] not in paused_groups)
        self.report_attention(store)
        started = []
        if running:
            started = self.start_agents(store, definition, paused_groups)
        self.report_new_items(store, definition)
        return started

    def observe(self, store: WorkflowStore, agent: dict, definition: dict, should_run: bool) -> None:
        agent = store.agent(agent["id"])
        if agent["state"] not in {"starting", "running"}:
            return
        handle = self.handles.get(agent["id"])
        if agent["state"] == "starting":
            observation = handle.poll() if handle is not None else None
            if observation is not None and observation.status == "ready":
                ready = observation.ready or {}
                agent = store.update_agent(agent["id"], state="running", websocket_url=ready.get("websocket_url"),
                                           instance_id=ready.get("instance_id", agent["instance_id"]),
                                           process_identity=ready.get("process_identity") or agent.get("process_identity"))
                try:
                    if agent.get("release") == "resume":
                        self.send(store, agent, "message",
                                  "Your session was restarted after an interruption. Continue your task where you "
                                  "left off, and call finish_task when it is done.")
                        store.update_agent(agent["id"], release=None)
                    if not should_run:
                        self.pause_one(store, store.agent(agent["id"]))
                except Exception as exc:
                    LOG.warning("Could not release %s: %s", agent["id"], exc)
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
                store.update_agent(agent["id"], state="running")
                if not should_run:
                    self.pause_one(store, store.agent(agent["id"]))
                return
        status = self.liveness(store, agent)
        if status != "dead":
            return
        agent = store.agent(agent["id"])
        if agent["state"] not in {"starting", "running"}:
            return  # It finished (or was changed) while we looked.
        self.handles.pop(agent["id"], None)
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
            self.handles.pop(agent_id, None)
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
        if command == "run":
            return self.controller.run(workflow_name, agent_group)
        if command == "pause":
            return self.controller.pause(workflow_name, agent_group)
        if command == "archive":
            if agent_group:
                raise WorkflowError("archive applies to the whole workflow; omit agent_group")
            return self.controller.archive(workflow_name)
        raise WorkflowError(f'unknown command "{command}"; use run, pause or archive')

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
        "Run, pause or archive a workflow. run: take control in this session, resume paused agents where they "
        "left off, and start new agents up to each group's count (agents cost money; confirm with the user "
        "first). pause: start no new agents and interrupt running ones, keeping their work. archive: stop all "
        "agents and retire the workflow; its history stays readable.",
        {"workflow_name": WORKFLOW_NAME_PROPERTY,
         "command": {"type": "string", "enum": ["run", "pause", "archive"], "description": "What to do."},
         "agent_group": AGENT_GROUP_PROPERTY},
        ["workflow_name", "command"])
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

USAGE = "/workflow | /workflow run NAME [GROUP] | /workflow pause NAME [GROUP]"


def parse_command(raw_args: str) -> dict:
    try:
        tokens = shlex.split(raw_args or "")
    except ValueError as exc:
        raise WorkflowError(f"could not parse the command: {exc}") from None
    if not tokens or tokens == ["list"]:
        return {"action": "list"}
    action = tokens[0].lower()
    if action not in {"run", "pause"}:
        raise WorkflowError(f'unknown action "{tokens[0]}". Usage: {USAGE}')
    if len(tokens) not in (2, 3):
        raise WorkflowError(f"Usage: {USAGE}")
    name = check_workflow_name(tokens[1])
    group = check_group_id(tokens[2]) if len(tokens) == 3 else None
    return {"action": action, "name": name, "group": group}


class WorkflowCommand:
    def __init__(self, plugin: WorkflowsPlugin):
        self.plugin = plugin

    def __call__(self, ctx, *, raw_args=""):
        controller = self.plugin.controller
        try:
            request = parse_command(raw_args)
            if worker_identity(ctx.state):
                raise WorkflowError("workflow agents cannot control workflows")
            controller.bind(ctx.state)
            if request["action"] == "list":
                text = render_index(None, (controller.me or {}).get("session_id", ""))
                return CommandResult(outputs=(InspectorDocument(title="Workflows", body=text, format="text"),))
            WorkflowStore(request["name"]).require()
            if request["action"] == "run":
                controller.require_launch()
        except WorkflowError as exc:
            raise CommandError(str(exc)) from exc
        action = controller.run if request["action"] == "run" else controller.pause

        async def work():
            try:
                return {"ok": True, "text": await asyncio.to_thread(action, request["name"], request["group"])}
            except Exception as exc:
                return {"ok": False, "text": f"Workflow {request['name']}: {exc}"}

        ctx.defer(work, then=self.complete)
        verb = "Running" if request["action"] == "run" else "Pausing"
        scope = f"{request['group']} in {request['name']}" if request["group"] else request["name"]
        return CommandResult.notice(f"{verb} {scope}...")

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

    def __call__(self, state):
        state.buffer_manager.register_special_buffer_namespace("wf:", self.render, replace=True)
        return state

    @staticmethod
    def render(state, buffer_id):
        if not buffer_id.startswith("wf:"):
            return None

        def transcript(session_id):
            parts = buffer_id.split(":")
            agent = WorkflowStore(parts[1]).agent(parts[2])
            return saved_transcript(agent.get("project") or "default", session_id)

        text = render_view(buffer_id, viewer_session_id=str(getattr(state, "_session_id", "") or ""),
                           transcript=transcript)
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
            package = installed if installed.is_file() else Path(__file__).resolve().parent.parent / relative
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
            WorkflowBuffers()(state)  # Paused startups skip ordinary components.
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
        WorkflowBuffers(), WorkflowNotices(plugin), *build_tools(plugin),
        Command("/workflow", "List, run or pause workflows", WorkflowCommand(plugin), raw=True,
                usage=USAGE, section="Workflows"),
    ], order=[RegisterSpecialBuffers, WorkflowBuffers, Command, WorkflowNotices, InterruptDelivery,
              ToolDispatchStart]))
    builder.add(Feature("workflow_guidance", components=[
        WorkflowSkillRegistration(), WorkflowSystemPrompt(plugin),
    ], order=[WorkflowSkillRegistration, WorkflowSystemPrompt, SystemPromptSkillList, MessageRenderer]))
