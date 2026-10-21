"""Watch the fly brain sort: ``python -m flybrain_sort.gui 42 7 19 3``.

Runs a sort while recording every spike in the brain, then writes a
self-contained HTML viewer (``flybrain-sort.html``) and opens it in a browser.
The page replays the whole-brain activity, a cartoon fly reacting to each
comparison, and the sorting network as the items move.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from importlib import resources
from pathlib import Path

HEAD = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
)


def page(replay: dict, standalone: bool = True) -> str:
    """The viewer with ``replay`` embedded.

    ``standalone`` gives a complete HTML document; otherwise the fragment is
    returned for hosts that supply their own document skeleton.
    """
    template = resources.files("flybrain_sort").joinpath("viewer.html").read_text(encoding="utf-8")
    data = json.dumps(replay, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("__REPLAY__", data)
    head, body = html.split("<!--BODY-->", 1)
    if not standalone:
        return head + body
    return HEAD + head + "</head>\n<body>\n" + body + "</body>\n</html>\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flybrain_sort.gui", description=__doc__.splitlines()[0])
    ap.add_argument("values", nargs="*", type=float, help="numbers to sort (default: a demo list)")
    ap.add_argument("-o", "--out", type=Path, default=Path("flybrain-sort.html"), help="HTML file to write")
    ap.add_argument("--replay", type=Path, help="build the page from a saved replay instead of running the brain")
    ap.add_argument("--save-replay", type=Path, help="also save the recorded replay as JSON")
    ap.add_argument("--duration", type=float, default=200.0, help="brain time per comparison, ms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    args = ap.parse_args(argv)

    if args.replay:
        replay = json.loads(args.replay.read_text())
    else:
        from .replay import record
        from .sorter import FlyBrainSorter

        values = args.values or [42, 7, 19, 3, 88, 61, 25, 50]
        values = [int(v) if float(v).is_integer() else v for v in values]
        print(f"sorting {values} with the fly brain (recording every spike) ...")
        replay = record(FlyBrainSorter(duration=args.duration, seed=args.seed), values)
        print(f"fly says: {replay['sorted']}")
        if args.save_replay:
            args.save_replay.write_text(json.dumps(replay, separators=(",", ":")))
    args.out.write_text(page(replay), encoding="utf-8")
    print(f"wrote {args.out}")
    if not args.no_open:
        webbrowser.open(args.out.resolve().as_uri())


if __name__ == "__main__":
    main()
