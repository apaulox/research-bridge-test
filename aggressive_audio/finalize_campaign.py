"""Select on fixed external development data, then compare fusion once."""
import json
from pathlib import Path
import subprocess
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parent
CANDIDATES = {
    "music": ["music_last2_none_v1", "music_last2_channel_v2",
              "music_head_channel_1024_v1", "music_head_channel_overlay_1024_v1"],
    "speech": ["speech_probe_none_v1", "speech_probe_channel_v1"],
}


def main():
    result = {"selection": "minimum mean(clean EER, fixed-channel stress EER); earliest epoch on ties",
              "scope": "Small external development selection; upstream overlap unresolved", "branches": {}}
    for branch, names in CANDIDATES.items():
        reference = None
        trials = []
        for name in names:
            folder = ROOT / "runs" / name
            if not (folder / "complete.json").exists():
                raise RuntimeError(f"Training incomplete: {name}")
            cfg = json.loads((folder / "config.json").read_text())
            ids = (cfg["selected_training_ids"], cfg["selected_dev_ids"])
            if reference is None:
                reference = ids
            if reference != ids:
                raise AssertionError("Candidate development cohorts differ")
            best = min(json.loads((folder / "history.json").read_text()), key=lambda r: r["selection_metric"])
            manifest = Path(cfg["manifest"]) if cfg["manifest"] else ROOT / "manifests" / f"{branch}.jsonl"
            rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
            selected = set(cfg["selected_training_ids"])
            counts = Counter(f"{r['generator']} / label={r['label_fake']}" for r in rows if r["id"] in selected)
            trials.append({"name": name, "best": best, "adaptation": str(folder / "best_adaptation.pt"),
                           "selected_training_generator_counts": dict(sorted(counts.items()))})
        chosen = min(trials, key=lambda r: r["best"]["selection_metric"])
        result["branches"][branch] = {"trials": trials, "chosen": chosen}
    (ROOT / "records" / "campaign_selection.json").write_text(json.dumps(result, indent=2))
    output = ROOT / "fusion_diagnostic" / "predictions_v1"
    if (output / "inference_contract.json").exists():
        contract = json.loads((output / "inference_contract.json").read_text())
        for branch in CANDIDATES:
            if contract[branch + "_adaptation"] != result["branches"][branch]["chosen"]["adaptation"]:
                raise RuntimeError("Cached predictions use a different adaptation")
    if not (output / "inference_contract.json").exists():
        if output.exists():
            raise RuntimeError(f"Incomplete inference output exists; inspect before retrying: {output}")
        subprocess.run([sys.executable, "-B", str(ROOT / "infer.py"),
                        "--input-dir", str(ROOT / "fusion_diagnostic" / "audio"),
                        "--output-dir", str(output),
                        "--music-adaptation", result["branches"]["music"]["chosen"]["adaptation"],
                        "--speech-adaptation", result["branches"]["speech"]["chosen"]["adaptation"]], check=True)
    subprocess.run([sys.executable, "-B", str(ROOT / "evaluate_fusion.py"),
                    "--labels", str(ROOT / "fusion_diagnostic" / "labels.json"),
                    "--predictions", str(output)], check=True)
    fusion = json.loads((output / "fusion_comparison.json").read_text())
    lines = ["# 증강 대조 실험 결과 · 2026-09-12", "",
             "실제 추가 학습과 로컬 추론을 완료한 결과다. DACON 제출 점수나 독립 SOTA 검증 결과는 아니다.", "",
             "## 조건과 선택", "",
             "각 실행은 학습 1,024개(진위 각 512), 검증 256개(각 128), 2 epochs다. 분기별 후보는 같은 학습·검증 표본을 사용한다. 음악 head와 last2는 학습률도 다르므로 순수한 층 수 효과로 해석하지 않는다.",
             "깨끗한 입력과 고정 음질 변형 입력의 EER 평균이 가장 낮은 epoch를 고른다. 동률이면 먼저 저장한 epoch를 유지한다. 낮을수록 좋다.", ""]
    for branch, values in result["branches"].items():
        lines.extend(["| 실행 | 선택 epoch | Clean EER | Stress EER |", "|---|---:|---:|---:|"])
        for trial in values["trials"]:
            m = trial["best"]
            lines.append(f"| {trial['name']} | {m['epoch']} | {m['clean_eer']:.2%} | {m['stress_eer']:.2%} |")
        lines.extend(["", f"선택한 {branch} 모델: `{values['chosen']['name']}/best_adaptation.pt`", ""])
    lines.extend(["## 결합 진단", "",
                  f"동일 성분 예측을 사용한 외부 합성·단독 음원 64개에서 FILE EER: mean **{fusion['mean']['file_eer']:.2%}**, max **{fusion['max']['file_eer']:.2%}**.",
                  "`mean=(VP×VF+MP×MF)/2`, `max=max(VP×VF,MP×MF)`이다. 음성 분기 내부의 구간 max는 유지했다.",
                  "EER에서 `/2` 상수 배율은 순위를 바꾸지 않는다. 개선 여부는 max 대신 두 분기를 합쳐 순위가 바뀌는 효과로 판단한다.",
                  "원본 32개를 여러 혼합에 재사용해 64개가 독립 표본은 아니다. 이 결과만으로 결합 방식을 확정하거나 DACON 개선을 보장할 수 없다.", "",
                  "## 해석과 한계", "",
                  "- 음악은 EAT+AASIST 한 모델로 구성하고, 음성은 공식 AntiDeepfake XLS-R 1B+linear probe에서 probe만 추가 학습했다. 상세 출처와 데이터 가공은 README.md와 manifests에 보관했다.",
                  "- 증강은 전화망·압축·합성 replay·잡음·리샘플링·음량·비생성형 음질 개선이다. 실제 장비 재녹음이나 모든 복제 시스템을 수집·검증한 것은 아니다.",
                  "- ASV와 CoSG 실제·생성 데이터를 학습에 넣었지만 ASV dev만으로 최신 복제·코덱 생성기에 대한 일반화를 입증하지 못한다. 공개 pretrained 모델의 원래 학습 자료와 검증 자료 간 중복은 불명확하다.",
                  "- DACON 평가 데이터 학습·보정이나 파일 간 통계·attention은 사용하지 않았다. 개별 음원·CoSG 원본 demo 이용조건과 서버 Python 3.11 호환성 확인은 최종 제출 전 남아 있다.",
                  "- 추론 CSV는 로컬 진단용이다. 새 제출 ZIP 제작·업로드·서버 시간 제한 검증을 완료했다고 해석하면 안 된다.", "",
                  "## 재개", "",
                  "continue_campaign.py는 완료 실행을 건너뛰고 미완료 실행의 checkpoint_latest.pt를 복원한다. 이후 finalize_campaign.py로 선택·추론·결합 진단을 이어간다.",
                  "전체 재개 checkpoint는 가중치·optimizer·난수 상태·다음 step을 포함한다. 작은 실제 중단/재개 실험에서 연속 실행과 최종 가중치가 완전히 일치했다. 기록: records/resume_verification.json.",
                  "기존 music_last2_none_v1과 pilot 실행은 가중치만 저장돼 정확한 optimizer 재개가 불가능하다. 새 campaign 실행들은 전체 상태를 저장한다.", ""])
    lines.extend(["## 실제 선택된 학습 데이터", "",
                  "다음은 전체 공개 데이터셋 규모가 아니라 선택 모델의 학습에 실제 사용한 표본 수다. label=1은 fake, label=0은 real이다. 원본 경로·출처·이용조건·split은 manifests의 같은 ID로 연결된다.", ""])
    for branch, values in result["branches"].items():
        lines.extend([f"### {branch}", "", "| 생성기 / 라벨 | 표본 수 |", "|---|---:|"])
        for key, count in values["chosen"]["selected_training_generator_counts"].items():
            lines.append(f"| {key} | {count} |")
        lines.append("")
    (ROOT / "RESULTS_2026-09-12.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
