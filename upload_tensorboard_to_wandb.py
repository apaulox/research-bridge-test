#!/usr/bin/env python
import argparse
from collections import defaultdict
import os

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(
        description="Import scalar data from a TensorBoard event directory into W&B."
    )
    parser.add_argument("log_dirs", nargs="+")
    parser.add_argument("--project", default="DenseSSL")
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backbone", required=True)
    args = parser.parse_args()

    metrics_by_step = defaultdict(dict)
    scalar_tags = set()
    log_dirs = [os.path.abspath(path) for path in args.log_dirs]
    for log_dir in log_dirs:
        accumulator = EventAccumulator(log_dir).Reload()
        directory_tags = accumulator.Tags().get("scalars", [])
        scalar_tags.update(directory_tags)
        for tag in directory_tags:
            for event in accumulator.Scalars(tag):
                metrics_by_step[event.step][tag] = event.value
    scalar_tags = sorted(scalar_tags)
    if not scalar_tags:
        raise RuntimeError("No TensorBoard scalar data found")

    run = wandb.init(
        project=args.project,
        name=args.name,
        id=args.run_id,
        resume="allow",
        job_type="tensorboard-import",
        config={
            "visual_backbone": args.backbone,
            "dataset": "FAIR-Play",
            "split": "unseen1",
            "source": "TensorBoard import",
            "source_log_dirs": log_dirs,
        },
    )
    run.define_metric("global_step")
    run.define_metric("data/*", step_metric="global_step")

    for step in sorted(metrics_by_step):
        run.log({"global_step": step, **metrics_by_step[step]})

    run.summary["imported_scalar_tags"] = scalar_tags
    run.summary["imported_steps"] = len(metrics_by_step)
    run.finish()

    print("Imported %d steps from %d TensorBoard directories" % (
        len(metrics_by_step), len(log_dirs)))


if __name__ == "__main__":
    main()
