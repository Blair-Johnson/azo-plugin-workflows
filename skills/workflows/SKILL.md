---
name: Workflows
description: Build and run named multi-agent workflows from queues and agent groups. Use when the user wants several agents working through a stream of tasks (for example, find things, process each one, then act on the results), wants to resume or change a workflow, or mentions /workflow or wf:<name> buffers.
---

# Workflows

A workflow is a named, persistent set of **queues** and **agent groups**. Queues
hold items (text). Agent groups start agents that work on items and send results
to other queues. You design the workflow in YAML; the plugin starts, pauses and
resumes the agents. The scheduler runs inside the session that controls the
workflow, so a workflow only advances while some session has it running.

## IDs

| Object | Form | Example |
|---|---|---|
| Workflow | plain name | `miner` |
| Queue | `q_<name>` | `q_targets` |
| Agent group | `grp_<name>` | `grp_reverse` |
| Agent | `agt_<group>_<number>` | `agt_reverse_3` |
| Item | `item_<number>` | `item_42` |
| Definition version | `def_<number>` | `def_4` |

`q_trash` always exists; moving items there discards them reversibly.

## Read state through buffers

```text
wf:index                    every workflow and which session controls it
wf:miner                    overview: groups, queue sizes, anything needing attention
wf:miner:definition         current YAML (copy this to edit)
wf:miner:history            every definition version with its reason
wf:miner:def_4              a past version
wf:miner:q_targets          the items in a queue
wf:miner:grp_reverse        a group's agents and instructions
wf:miner:agt_reverse_3      one agent: its item, outputs and transcript
wf:miner:item_42            one item: text, notes, where it came from
wf:miner:log                every event, newest first
```

## Writing a definition

```yaml
folder: ~/work/papers            # working directory for agents (optional)
model: claude-sonnet-5-5         # model profile for agents (optional; else the saved Agent Zoo default)
queues:
  q_papers: {}
  q_summaries: { notify_on_new: true }   # tell you when items arrive
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
    folder: ~/work/papers/{{agent}}
    instructions: |
      Summarize this paper in one page.
      {{item}}
```

Group settings: `count` (agents at once, default 1), `takes_from`, `sends_to`,
`every` (only without `takes_from`), `instructions` (required), `folder`, `model`.
Queue settings: `notify_on_new`.

In `instructions` and `folder`: `{{item}}` expands to the item the agent took
(ID, title, notes, origin and text), `{{agent}}` to the agent ID,
`{{workflow_name}}` to the workflow name, and `{{buffer://wf:...}}` to a workflow
view at launch time.

Instructions are free-form. Agents already know how to finish (they call
`finish_task` and send results to queues), so write instructions about the work,
including which queue each kind of result belongs in. There are no schemas: if
an agent can get stuck, give it a queue such as `q_blocked` and say when to use it.

## Creating, running and changing

1. Draft the YAML in a scratch buffer (`create_buffer`, `append_lines`).
2. `create_workflow(workflow_name="papers", definition_buffer="buf_...")`.
   Nothing runs yet. Add starting items with `add_items` if needed.
3. Confirm with the user before starting, since agents cost money. Then
   `command_workflow(workflow_name="papers", command="run")`, or the user types
   `/workflow run papers`.
4. To change it, copy `wf:papers:definition` into a buffer, edit it, and call
   `update_workflow(..., reason="...", preview_only=True)` to see the diff and the
   effects on running agents. Then call it again without `preview_only`.
   Running agents keep the definition they started with.

`set_agent_count` changes one group's count immediately (it records a new
version). `command_workflow` takes `run`, `pause`, `archive` or `unarchive`;
`run` and `pause` can target a single `agent_group`. Omit it or pass `null` for
the whole workflow; an empty string is also accepted. Other optional tool fields
accept `null` for their default. `pause` interrupts running agents but keeps their
work; `run` resumes safe checkpoints.

Archive one-off or test workflows so they stop cluttering the user's `/workflow`
list; `unarchive` brings one back, paused. `rename_workflow(workflow_name, new_name)`
renames a workflow that is paused or archived and has no unfinished agents (agent
sessions are tied to the name). Give workflows you may reuse a descriptive name.

A checkpoint with missing tool results is not automatically resumed: that tool may
already have changed something. The worker needs attention and keeps its item.
Inspect `wf:<name>:<agt_agent>` and existing artifacts/processes before choosing
`restart_agent`, which discards progress and requeues the item. Completed tool
results resume via `/continue`, without adding a new user assignment. Recovery
cannot guarantee exactly-once side effects or completion of background commands.

## Items and agents

- `add_items(workflow_name, queue, from_buffers, note=None)`: each buffer becomes
  one item; its first line is the title.
- `move_items(workflow_name, item_ids, to_queue, note=None)`: retry (move a done
  item back into a queue), reroute, or discard (`q_trash`). Notes are shown to the
  next agent that takes the item.
- `message_agents(workflow_name, message, agent_group=None)`: send a message to
  running agents.
- `restart_agent(workflow_name, agent)`: stop a stuck agent, discard its progress
  and return its item to its queue.

You are only notified about agents that need attention and about new items in
queues with `notify_on_new`. Read `wf:<name>` for everything else.

## The user's commands

```text
/workflow list                                  list workflows (archived ones are hidden)
/workflow list archived                          list archived workflows
/workflow view miner[:grp_reverse | :q_catalog]  show a workflow, group or queue
/workflow run miner[:grp_reverse]                take control in this session and run it
/workflow pause miner[:grp_reverse]              pause it; no work is lost
/workflow rename miner reverse_catalog           rename it (paused, no unfinished agents)
/workflow archive miner                          stop it and hide it from the list
/workflow unarchive miner                        bring it back, paused
/workflow set miner:grp_reverse count 8          change a simple setting (see below)
/workflow set miner:grp_reverse                  list the settings there, with current values
```

Settings the user can change directly: `model` and `folder` on the workflow;
`count`, `model`, `every` and `folder` on an agent group; `notify` on a queue.
`default` removes a model or folder setting, and `every off` removes the interval.
Each change records a new definition version, just like your own edits.

You are told when the user runs one of these. The result is marked as already
done: do not repeat it. If you were editing a copy of the definition, your next
`update_workflow` is rejected with a diff; copy `wf:<name>:definition` again and
reapply your change. Model profiles must exist and be enabled; a bad name is
rejected with the list of enabled profiles. When a workflow runs in a new
session, it continues where it left off.
