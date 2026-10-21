import json
import re

from flybrain_sort import FlyBrainSorter
from flybrain_sort.gui import page
from flybrain_sort.replay import record

from test_brain import toy_brain


def test_replay_records_every_comparison_and_embeds_in_the_viewer(tmp_path):
    sorter = FlyBrainSorter(toy_brain(), cache=tmp_path / "c.json", duration=100)
    values = [5, 1, 9, 3]
    replay = record(sorter, values)
    assert replay["sorted"] == sorted(values)
    comps = [c for r in replay["rounds"] for c in r["comparisons"]]
    assert len(comps) == 5  # Batcher network for 4 items
    assert all(len(r["frames"]) == 25 for r in replay["rounds"])
    assert len(replay["shown"]) == 2 * len(replay["roles"])
    shown = len(replay["roles"])
    assert all(0 <= k < shown for r in replay["rounds"] for f in r["frames"] for k in f)

    html = page(replay)
    assert html.startswith("<!doctype html>") and "<!--BODY-->" not in html
    data = re.search(r'<script id="replay" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(data)["sorted"] == sorted(values)
    assert "<!doctype" not in page(replay, standalone=False)
