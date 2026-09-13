"""Compare only FILE fusion using identical cached component predictions."""
import argparse
import csv
import json
from pathlib import Path

from contracts import eer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    args = p.parse_args()
    labels = json.loads(args.labels.read_text())
    tables = {}
    for method in ["mean", "max"]:
        with (args.predictions / f"submission_{method}.csv").open() as handle:
            tables[method] = {row["ID"]: row for row in csv.DictReader(handle)}
    expected = {row["ID"] for row in labels}
    if any(set(table) != expected for table in tables.values()):
        raise ValueError("Prediction IDs do not match labels")
    fields = ["VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"]
    for file_id in expected:
        if any(tables["mean"][file_id][key] != tables["max"][file_id][key] for key in fields):
            raise AssertionError("Fusion comparison changed a component score")
    result = {"identical_component_predictions": True, "files": len(labels),
              "scope": "Small paired external FILE diagnostic; not DACON score; upstream overlap possible"}
    for method, table in tables.items():
        result[method] = {"file_eer": eer([r["label_file_fake"] for r in labels],
                                         [float(table[r["ID"]]["FILE_FAKE_PROB"]) for r in labels])}
    (args.predictions / "fusion_comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
