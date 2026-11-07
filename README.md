# azo-plugin-workflows

Named, persistent multi-agent workflows for Agent Zoo. Describe a workflow to your
agent and it can stand one up in minutes: queues of items, and groups of agents that
take items from one queue and send results to others. You don't write an
orchestration app. The scheduler runs inside the session that controls the workflow,
and every workflow can be resumed from a later session.

## Model

- **Queues** (`q_*`) hold items. An item is text plus its notes and history.
- **Agent groups** (`grp_*`) start agents.
  - A group with `takes_from` starts one agent per item from that queue.
  - A group without it keeps `count` agents running from the same instructions,
    optionally starting a new one at most once per `every` interval.
- **Wiring** lives on agent groups (`takes_from`, `sends_to`), so rewiring is a
  definition change.
- **Versions:** every change creates a new version (`def_N`) with a reason. Running
  agents keep the version they started with.

IDs: workflow `miner`, queue `q_targets`, group `grp_reverse`, agent `agt_reverse_3`,
item `item_42`, version `def_4`. `q_trash` always exists.

## Commands

```text
/workflow                                   list workflows
/workflow miner[:grp_reverse | :q_catalog]  show a workflow, group or queue, including models and counts
/workflow run miner[:grp_reverse]           run it from this session
/workflow pause miner[:grp_reverse]         pause it; no work is lost
/workflow set miner[:grp_reverse | :q_catalog] SETTING VALUE
/workflow set miner:grp_reverse             list what you can set there, with current values
```

| Target | Setting | Values | Takes effect |
|---|---|---|---|
| `miner` | `model` | a model profile, or `default` | agents started afterward |
| `miner` | `folder` | a path, or `default` | agents started afterward |
| `miner:grp_*` | `count` | 0–64 | immediately |
| `miner:grp_*` | `model` | a model profile, or `default` | agents started afterward |
| `miner:grp_*` | `every` | a duration like `30m`, or `off` | the next start |
| `miner:grp_*` | `folder` | a path, or `default` | agents started afterward |
| `miner:q_*` | `notify` | `on` or `off` | immediately |

Each change records a new definition version. Model profiles must exist in
`llm.models` and be enabled. Without a `model:` setting, agents use your saved
Agent Zoo default (not a session's model change or a `--set` override).

## Agent tools

Controlling session:

```python
create_workflow(workflow_name, definition_buffer)
update_workflow(workflow_name, definition_buffer, reason, preview_only=False)
command_workflow(workflow_name, command, agent_group=None)   # run | pause | archive
set_agent_count(workflow_name, agent_group, count)
message_agents(workflow_name, message, agent_group=None)
restart_agent(workflow_name, agent)
add_items(workflow_name, queue, from_buffers, note=None)
move_items(workflow_name, item_ids, to_queue, note=None)
```

Agents started by a workflow:

```python
finish_task(summary, outputs=[])        # outputs: [{"buffer": "buf_...", "queue": "q_..."}]
add_items(queue, from_buffers, note=None)
move_items(item_ids, to_queue, note=None)
```

Workers only touch their own workflow, and only the queues in their group's
`sends_to`. `finish_task` marks their item done and ends the session. Workers lose
the agent-facing RLM tools.

## Buffers

```text
wf:index  wf:miner  wf:miner:definition  wf:miner:history  wf:miner:def_4
wf:miner:q_targets  wf:miner:grp_reverse  wf:miner:agt_reverse_3  wf:miner:item_42  wf:miner:log
```

## Storage and recovery

Each workflow lives in its own directory,
`<state-home>/plugin-configs/azo-plugin-workflows/workflows/<name>/`. That directory
holds the definition versions, items, agent records, an append-only log, and a
control record naming the controlling session. Writes are atomic and use a lock per
workflow.

- **Taking control:** `/workflow run` takes control in the current session. The
  previous controller notices and stops scheduling.
- **Running agents:** live agents are reattached.
- **Stopped agents:** agents that stopped without finishing resume from their saved
  checkpoints. Agents that can't resume are flagged for attention, and
  `restart_agent` hands their item to a fresh agent.
- **Restarts:** after a reload or restart, a session that was running a workflow
  picks it up again on its own.

## Install and test

```sh
pixi run azo-plugin install ~/Documents/GitHub/azo-plugin-workflows   # from the agent-zoo checkout
pixi run test                                                          # uses ../agent-zoo/pixi.toml by default
```

Set `AZO_HOST_MANIFEST` to test against a different Agent Zoo checkout. After
installing, run `/reload` in the TUI.
