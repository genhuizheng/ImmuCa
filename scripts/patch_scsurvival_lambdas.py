#!/usr/bin/env python3
"""Expose scSurvival-extend's two hardcoded loss weights as parameters.

Why this is a patch script and not an edit
------------------------------------------
`scSurvival-extend/` is a colleague's code, extracted under `data/` on Vista and
gitignored here. An edit made on Windows cannot travel through git. This script
can: it is tracked, it runs on Vista, it is idempotent, and it reverts.

What it changes
---------------
`scsurvival_core.py` currently hardcodes two of the three new loss weights in the
middle of the training loop:

    lambda_ortho = 5e-3
    lambda_consist = 1e-3
    loss = loss + lambdas[0]*ae_loss + lambda_ortho*... + lambda_consist*...

so the orthogonal-attention and cell-patient-consistency terms cannot be turned
off, tuned, or ablated without editing the source. This promotes both to keyword
arguments of `fit()`, keeping 5e-3 / 1e-3 as defaults, so existing behaviour is
bit-identical unless a caller asks otherwise.

`scsurvival.py` already forwards `**kwargs` to `model.fit()` at every call site,
so no change is needed there -- after this patch,

    scSurvivalRun(..., lambda_ortho=0.0, lambda_consist=0.0, validate_entropy=False)

reproduces the base method, and the ablation becomes a single argument change.

Note the third term, attention entropy, is *already* controllable via the
existing `lambdas[1]` and `entropy_threshold`, so it needs no patch.

Usage
-----
    python scripts/patch_scsurvival_lambdas.py --dry-run     # show, change nothing
    python scripts/patch_scsurvival_lambdas.py               # apply
    python scripts/patch_scsurvival_lambdas.py --revert      # restore the backup
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PKG = REPO_ROOT / "data" / "scSurvival-extend" / "scSurvival" / "scSurvival_e"

# The anchor in fit()'s signature that the new parameters go after.
SIG_ANCHOR = "entropy_threshold=0.7,"
SIG_INSERT = [
    "            lambda_ortho=5e-3,",
    "            lambda_consist=1e-3,",
]

# The hardcoded assignments to remove, matched after stripping whitespace.
DEAD_ASSIGNMENTS = ["lambda_ortho = 5e-3", "lambda_consist = 1e-3"]

# Proof the patch is wired to something real: the line that consumes both.
CONSUMER = "lambda_ortho * atten_ortho_loss"


def patch(text: str) -> tuple[str, list[str]]:
    lines = text.split("\n")
    out: list[str] = []
    notes: list[str] = []
    inserted = removed = 0

    for ln in lines:
        stripped = ln.strip()

        # Drop the hardcoded assignments.
        if stripped in DEAD_ASSIGNMENTS:
            notes.append(f"  removed  {stripped}")
            removed += 1
            continue

        out.append(ln)

        # Insert the new keyword args right after the signature anchor.
        if stripped == SIG_ANCHOR and inserted == 0:
            out.extend(SIG_INSERT)
            notes.append(f"  inserted after '{SIG_ANCHOR}':")
            for s in SIG_INSERT:
                notes.append(f"             {s.strip()}")
            inserted = 1

    if inserted != 1:
        raise SystemExit(
            f"FAILED: expected exactly one '{SIG_ANCHOR}' in fit()'s signature, "
            f"found {inserted}. The file may already differ from the version this "
            "patch was written against; inspect it by hand."
        )
    if removed != 2:
        raise SystemExit(
            f"FAILED: expected to remove 2 hardcoded assignments, removed {removed}."
        )
    return "\n".join(out), notes


def already_patched(text: str) -> bool:
    return "lambda_ortho=5e-3," in text and "lambda_ortho = 5e-3" not in text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--package-dir", type=Path, default=DEFAULT_PKG,
                    help=f"scSurvival_e directory (default: {DEFAULT_PKG})")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    ap.add_argument("--revert", action="store_true", help="restore the .orig backup")
    args = ap.parse_args(argv)

    target = args.package_dir / "scsurvival_core.py"
    backup = args.package_dir / "scsurvival_core.py.orig"

    if not target.is_file():
        print(f"no such file: {target}\n"
              "On Windows this is expected -- the package lives on Vista under\n"
              "$SCRATCH/Tumor_immunity_analysis/data/. Pass --package-dir to override.",
              file=sys.stderr)
        return 1

    if args.revert:
        if not backup.is_file():
            print(f"no backup at {backup}; nothing to revert", file=sys.stderr)
            return 1
        shutil.copy2(backup, target)
        print(f"reverted {target} from {backup.name}")
        return 0

    text = target.read_text(encoding="utf-8")

    if already_patched(text):
        print(f"already patched: {target}")
        print("  lambda_ortho / lambda_consist are keyword arguments of fit()")
        return 0

    if CONSUMER not in text:
        print(f"FAILED: '{CONSUMER}' not found -- this does not look like the "
              "expected scsurvival_core.py", file=sys.stderr)
        return 1

    new_text, notes = patch(text)
    print(f"target: {target}")
    print("\n".join(notes))

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"  backup   {backup.name}")
    target.write_text(new_text, encoding="utf-8")
    print("\npatched.")
    print("""
The ablation is now one argument:

    # extension ON  (unchanged default behaviour)
    scSurvivalRun(adata, 'sample_id', ..., task_type='cox')

    # extension OFF (base method)
    scSurvivalRun(adata, 'sample_id', ..., task_type='cox',
                  lambda_ortho=0.0, lambda_consist=0.0, validate_entropy=False)

Verify the parameters really arrive, before trusting a result:

    python -c "import inspect, sys; sys.path.insert(0,'<...>/scSurvival'); \\
from scSurvival_e.scsurvival_core import scSurvival; \\
p=inspect.signature(scSurvival.fit).parameters; \\
print('lambda_ortho', p['lambda_ortho'].default, '| lambda_consist', p['lambda_consist'].default)"
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
