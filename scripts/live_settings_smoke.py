"""Opt-in live command test; requires explicit controller and workflow identity."""
import argparse
import asyncio
import json
from pathlib import Path

import yaml
from agent_zoo.live_session_registry import list_live_sessions
from agent_zoo.wsctl import AzoWs


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--workflow-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.workflow_dir
    name = root.name
    rows = [r for r in list_live_sessions(args.registry)
            if r.session_id == args.session and r.instance_id == args.instance]
    assert len(rows) == 1, "Expected one matching live controller"
    ws = AzoWs.attach(rows[0].websocket_url, terminate_on_close=False)
    await ws.connect()
    try:
        hello = ws.observation().hello
        assert hello.get("session_id") == args.session, hello
        assert hello.get("instance_id") == args.instance, hello

        def control():
            return json.loads((root / "control.json").read_text())

        def definition():
            files = sorted((root / "versions").glob("def_*.yaml"),
                           key=lambda p: int(p.stem.split("_")[1]))
            return yaml.safe_load(files[-1].read_text())

        async def command(raw, predicate):
            await ws.slash(raw, client_id="workflow-live-settings-test")
            for _ in range(100):
                await asyncio.sleep(0.1)
                if predicate():
                    print("PASS", raw, flush=True)
                    return
            raise AssertionError("No persisted effect: " + raw)

        await command(f"/workflow pause {name}", lambda: control()["state"] == "paused")
        original = definition()
        # All changes happen while paused; no new agent/model calls are made.
        cases = [
            (name, "model", "gpt-5.6-luna-short", (), "model", "gpt-5.6-luna-short"),
            (name, "model", "gpt-5.6-luna-max", (), "model", "gpt-5.6-luna-max"),
            (f"{name}:grp_reverse", "model", "gpt-5.6-luna-short", ("agents", "grp_reverse"), "model", "gpt-5.6-luna-short"),
            (f"{name}:grp_reverse", "model", "default", ("agents", "grp_reverse"), "model", None),
            (f"{name}:grp_reverse", "count", "3", ("agents", "grp_reverse"), "count", 3),
            (f"{name}:grp_reverse", "count", "2", ("agents", "grp_reverse"), "count", 2),
            (f"{name}:grp_finder", "every", "2h", ("agents", "grp_finder"), "every", "2h"),
            (f"{name}:grp_finder", "every", "1h", ("agents", "grp_finder"), "every", "1h"),
            (f"{name}:q_results", "notify", "off", ("queues", "q_results"), "notify_on_new", False),
            (f"{name}:q_results", "notify", "on", ("queues", "q_results"), "notify_on_new", True),
        ]
        for target, setting, value, path, key, expected in cases:
            def matches():
                data = definition()
                for part in path:
                    data = data[part]
                return data.get(key) == expected
            await command(f"/workflow set {target} {setting} {value}", matches)
        assert definition() == original, "Settings were not restored"
        assert control()["state"] == "paused"
        print("PASS original definition restored; workflow paused", flush=True)
    finally:
        await ws.close()


if __name__ == "__main__":
    asyncio.run(main())
