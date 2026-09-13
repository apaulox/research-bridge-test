#!/usr/bin/env python3
"""Fit CLAM calibration on FMC dev and evaluate EAT+CLAM on FMC eval."""

import argparse
import json

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


def load_join(clam_path, eat_path):
    clam = pd.read_csv(clam_path)
    eat = pd.read_csv(eat_path)
    frame = clam.merge(eat, on=["FILE", "LABEL_FAKE"], validate="one_to_one")
    return frame


def metrics(frame, scores):
    labels = frame["LABEL_FAKE"].to_numpy()
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "eer": eer(labels, scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-clam", required=True)
    parser.add_argument("--dev-eat", required=True)
    parser.add_argument("--eval-clam", required=True)
    parser.add_argument("--eval-eat", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dev = load_join(args.dev_clam, args.dev_eat)
    evaluation = load_join(args.eval_clam, args.eval_eat)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    calibrator.fit(dev[["LOGIT_90"]].to_numpy(), dev["LABEL_FAKE"].to_numpy())
    coefficient = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])

    report = {
        "calibration": {"coefficient": coefficient, "intercept": intercept},
        "dev": {},
        "eval": {},
    }
    for name, frame in (("dev", dev), ("eval", evaluation)):
        eat_probability = np.clip(frame["EAT_PROB"].to_numpy(), 1e-6, 1 - 1e-6)
        eat_logit = logit(eat_probability)
        clam_raw = frame["LOGIT_90"].to_numpy()
        clam_calibrated_logit = coefficient * clam_raw + intercept
        current = {
            "eat": metrics(frame, eat_probability),
            "clam_official": metrics(frame, clam_raw),
            "clam_calibrated": metrics(frame, clam_calibrated_logit),
            "weights": {},
        }
        for eat_weight in np.arange(0.0, 1.0001, 0.05):
            combined_logit = (
                eat_weight * eat_logit
                + (1.0 - eat_weight) * clam_calibrated_logit
            )
            current["weights"][f"{eat_weight:.2f}"] = metrics(
                frame, expit(combined_logit)
            )
        report[name] = current

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
