"""Measure the Gen 9 randbats mechanic pool against a Showdown checkout.

Usage:
    python tools/mechanics_inventory.py /path/to/pokemon-showdown

Reads data/random-battles/gen9/sets.json plus data/{moves,abilities,items,
conditions}.ts and sim/*.ts, and prints how many moves/abilities/items the
randbats pool uses, how many of those carry event handlers (i.e. need porting
logic, not just data), the handler line counts, and the engine event list.
This is the source of truth for coverage.yaml in the JAX port plan.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path


def toid(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def entries(ts_src: str) -> dict[str, dict]:
    """Split a Showdown data file into top-level `id: {...}` blocks."""
    out: dict[str, dict] = {}
    for m in re.finditer(r"\n\t([a-z0-9]+): \{\n(.*?)\n\t\},", ts_src, re.S):
        body = m.group(2)
        hooks = set(re.findall(r"\b(on[A-Z][A-Za-z]*)\s*[:(]", body))
        out[m.group(1)] = dict(
            hooks=sorted(hooks),
            lines=body.count("\n") + 1,
            cond="condition:" in body,
        )
    return out


def report(name: str, pool: set[str], table: dict[str, dict]) -> dict:
    ids = sorted({toid(x) for x in pool})
    found = [i for i in ids if i in table]
    missing = [i for i in ids if i not in table]
    trivial = [i for i in found if not table[i]["hooks"] and not table[i]["cond"]]
    nontrivial = [i for i in found if i not in trivial]
    lines = sum(table[i]["lines"] for i in found)
    hookc = collections.Counter(h for i in nontrivial for h in table[i]["hooks"])
    print(
        f"\n{name}: pool={len(ids)} found={len(found)} missing={len(missing)} "
        f"data-only={len(trivial)} with-handlers={len(nontrivial)} handler-lines={lines}"
    )
    print("  top hooks:", hookc.most_common(20))
    big = sorted(nontrivial, key=lambda i: -table[i]["lines"])[:15]
    print("  largest:", [(i, table[i]["lines"]) for i in big])
    if missing:
        print("  missing:", missing[:10])
    return {i: table[i] for i in found}


def main(root: Path) -> None:
    sets = json.loads((root / "data/random-battles/gen9/sets.json").read_text())
    species, moves, abilities, teras = set(), set(), set(), set()
    roles: collections.Counter = collections.Counter()
    for sp, d in sets.items():
        species.add(sp)
        for s in d.get("sets", []):
            roles[s.get("role")] += 1
            moves.update(s.get("movepool", []))
            abilities.update(s.get("abilities", []))
            teras.update(s.get("teraTypes", []))
    print(
        f"species={len(species)} sets={sum(roles.values())} moves={len(moves)} "
        f"abilities={len(abilities)} tera_types={len(teras)}"
    )
    print("roles:", dict(roles))

    items_ts = (root / "data/items.ts").read_text()
    teams_ts = (root / "data/random-battles/gen9/teams.ts").read_text()
    item_names = set(re.findall(r'name:\s*"([^"]+)"', items_ts))
    items = {
        i for i in re.findall(r"['\"]([A-Z][A-Za-z' .-]+)['\"]", teams_ts) if i in item_names
    }
    print(f"items referenced by teams.ts: {len(items)}")

    mv = report("MOVES", moves, entries((root / "data/moves.ts").read_text()))
    ab = report("ABILITIES", abilities, entries((root / "data/abilities.ts").read_text()))
    it = report("ITEMS", items, entries(items_ts))

    sim = "".join(
        (root / p).read_text()
        for p in ("sim/battle.ts", "sim/battle-actions.ts", "sim/battle-queue.ts")
    )
    evs = sorted(set(re.findall(
        r"(?:runEvent|singleEvent|priorityEvent|eachEvent|fieldEvent|residualEvent)"
        r"\(\s*['\"]([A-Za-z]+)['\"]", sim)))
    print(f"\nENGINE EVENTS fired: {len(evs)}\n{evs}")
    cond = entries((root / "data/conditions.ts").read_text())
    print(f"\nCONDITIONS: {len(cond)}\n{sorted(cond)}")

    out = root.parent / "mechanics_inventory.json" if not Path("docs").exists() else Path("docs/mechanics_inventory.json")
    out.write_text(json.dumps({"moves": mv, "abilities": ab, "items": it, "events": evs,
                               "conditions": sorted(cond)}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
