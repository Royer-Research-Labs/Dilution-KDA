"""Record the vendored public harness in vendored.json: each file's source path, both sha256s and the exact edits
(old, new). An empty edit list means byte-identical.

    python tools/write_vendored.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "dilution-attention"
EDITS = {
    "src/dilution/train.py": [[
        "    optimizer, fused_optimizer = make_adamw(model.parameters(), training, runtime)",
        "    # Parameters tagged _no_weight_decay (KDA's per-head A_log and dt_bias, mixers.py) are not decayed, as in fla /\n"
        "    # Kimi Linear; no softmax or dilution parameter is tagged, so their optimisation is unchanged.\n"
        "    decayed = [p for p in model.parameters() if not getattr(p, \"_no_weight_decay\", False)]\n"
        "    undecayed = [p for p in model.parameters() if getattr(p, \"_no_weight_decay\", False)]\n"
        "    param_groups = [{\"params\": decayed}] + ([{\"params\": undecayed, \"weight_decay\": 0.0}] if undecayed else [])\n"
        "    optimizer, fused_optimizer = make_adamw(param_groups, training, runtime)",
    ]],
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    commit = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(SRC), "status", "--porcelain", "--", "src", "scripts", "tests"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:  # the source tree has uncommitted changes: say so rather than claim the bare commit
        commit += "+uncommitted"
    files = []
    for rel in sorted([*(f"src/dilution/{p.name}" for p in (SRC / "src/dilution").glob("*.py")),
                       *(f"scripts/{p.name}" for p in (SRC / "scripts").glob("*.py")),
                       *(f"tests/{p.name}" for p in (SRC / "tests").glob("*.py"))]):
        ours, theirs = ROOT / rel, SRC / rel
        edits = EDITS.get(rel, [])
        text = theirs.read_text(encoding="utf-8")
        for old, new in edits:
            assert text.count(old) == 1, (rel, old[:60])
            text = text.replace(old, new)
        assert text == ours.read_text(encoding="utf-8"), f"{rel} differs from the source with its recorded edits"
        files.append({"file": rel, "source": rel, "source_sha256": sha(theirs), "sha256": sha(ours), "edits": edits})
    record = {"source": "https://github.com/Royer-Research-Labs/Dilution-Attention", "commit": commit,
              "note": "Each file is the source file with `edits` (old, new) applied; an empty list means byte-identical. "
                      "The only edit exempts parameters tagged _no_weight_decay (KDA's A_log and dt_bias) from weight decay. "
                      "src/dilution/mixers.py (the KDA layer) is new and not vendored.",
              "files": files}
    (ROOT / "vendored.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"{len(files)} files recorded against {commit[:10]}; edited: {[f['file'] for f in files if f['edits']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
