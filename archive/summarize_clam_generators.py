#!/usr/bin/env python3
"""Summarize calibrated CLAM and EAT ensemble across FMC generators."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


def eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def join(clam_path, eat_path):
    return pd.read_csv(clam_path).merge(
        pd.read_csv(eat_path), on=["FILE", "LABEL_FAKE"], validate="one_to_one"
    )


def measure(frame, scores):
    labels = frame["LABEL_FAKE"].to_numpy()
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "eer": eer(labels, scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dev = join(
        args.directory / "TTM01_dev_100x2_durations.csv",
        args.directory / "TTM01_dev_100x2_eat.csv",
    )
    calibration = LogisticRegression(C=1.0, solver="lbfgs").fit(
        dev[["LOGIT_90"]].to_numpy(), dev["LABEL_FAKE"].to_numpy()
    )
    coefficient = float(calibration.coef_[0, 0])
    intercept = float(calibration.intercept_[0])

    report = {
        "calibration": {"coefficient": coefficient, "intercept": intercept},
        "generators": {},
    }
    all_frames = []
    for generator in ("TTM01", "TTM02", "TTM03", "TTM04", "TTM05"):
        stem = "TTM01_eval_100x2" if generator == "TTM01" else f"{generator}_eval_50x2"
        frame = join(
            args.directory / f"{stem}_d90.csv",
            args.directory / f"{stem}_eat.csv",
        )
        frame["GENERATOR"] = generator
        all_frames.append(frame)

        eat_prob = np.clip(frame["EAT_PROB"].to_numpy(), 1e-6, 1 - 1e-6)
        eat_logit = logit(eat_prob)
        clam_logit = coefficient * frame["LOGIT_90"].to_numpy() + intercept
        result = {
            "eat": measure(frame, eat_prob),
            "clam_calibrated": measure(frame, clam_logit),
            "ensembles": {},
        }
        for weight in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
            result["ensembles"][f"eat_{weight:.2f}"] = measure(
                frame, expit(weight * eat_logit + (1.0 - weight) * clam_logit)
            )
        report["generators"][generator] = result

    combined = pd.concat(all_frames, ignore_index=True)
    eat_prob = np.clip(combined["EAT_PROB"].to_numpy(), 1e-6, 1 - 1e-6)
    eat_logit = logit(eat_prob)
    clam_logit = coefficient * combined["LOGIT_90"].to_numpy() + intercept
    combined_result = {
        "eat": measure(combined, eat_prob),
        "clam_calibrated": measure(combined, clam_logit),
        "ensembles": {},
    }
    for weight in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        combined_result["ensembles"][f"eat_{weight:.2f}"] = measure(
            combined, expit(weight * eat_logit + (1.0 - weight) * clam_logit)
        )
    report["combined"] = combined_result

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
