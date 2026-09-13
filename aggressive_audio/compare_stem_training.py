"""Compare unchanged heads and retrained heads on the same separated development inputs."""
import gc
import json
from pathlib import Path

import torch

from contracts import load_manifest
from infer import apply_adaptation
from models import load_eat, load_speech
from run import validate

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "stem_experiment_medium_v1"


def main():
    torch.set_num_threads(4)
    torch.manual_seed(688)
    result = {}
    for branch, reference in [("music","music_head_channel_1024_v1"),("speech","speech_probe_channel_v1")]:
        current = f"{branch}_demucs_medium_v1"
        assert (ROOT/"runs"/current/"complete.json").exists()
        old = json.loads((ROOT/"runs"/reference/"history.json").read_text())
        new = json.loads((ROOT/"runs"/current/"history.json").read_text())
        old_best = min(old,key=lambda r:r["selection_metric"])
        new_best = min(new,key=lambda r:r["selection_metric"])
        rows = sorted([r for r in load_manifest(WORK/(branch+".jsonl")) if r["split"]=="dev"],key=lambda r:r["id"])
        config = json.loads((ROOT/"runs"/reference/"config.json").read_text())
        assert [r["id"] for r in rows] == config["selected_dev_ids"]
        model = (load_eat() if branch=="music" else load_speech())
        apply_adaptation(model,ROOT/"runs"/reference/"best_adaptation.pt",branch)
        model.cuda().eval()
        clean,clean_rows = validate(model,rows,branch,torch.device("cuda"),2,samples=64000)
        stress,stress_rows = validate(model,rows,branch,torch.device("cuda"),2,stress=True,samples=64000)
        result[branch] = {"reference":reference,"retrained":current,"same_development_ids":True,
                          "A_original_head_original_input":old_best,
                          "B_original_head_demucs_input":{"clean_eer":clean,"stress_eer":stress,"selection_metric":(clean+stress)/2},
                          "C_retrained_head_demucs_input":new_best,
                          "fixed_epoch2_original":old[-1],"fixed_epoch2_demucs":new[-1]}
        (WORK/(branch+"_original_head_on_stems.json")).write_text(json.dumps(clean_rows+stress_rows,indent=2))
        print(json.dumps({"branch":branch,"comparison":result[branch]}),flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (WORK/"comparison.json").write_text(json.dumps(result,indent=2))
    lines = ["# Demucs 적용 대조 실험", "",
             "변경한 것은 학습·검증 입력의 Demucs 분리이다. 음악은 drums+bass+other 합산, 음성은 vocals를 사용한다.",
             "이전 실험과 같은 파일 ID, 초기 공개 가중치, 고정 인코더, seed 688, 2 epochs, lr 1e-4, batch 2, accumulation 8, 64000-sample crop, 채널 증강을 사용했다.", "",
             "| 분기 | 조건 | 일반 EER | 분리 후 음질 변형 EER |", "|---|---|---:|---:|"]
    for branch, values in result.items():
        for key,label in [("A_original_head_original_input","A: 기존 head + 기존 입력"),
                          ("B_original_head_demucs_input","B: 기존 head + Demucs 입력"),
                          ("C_retrained_head_demucs_input","C: 재학습 head + Demucs 입력")]:
            m = values[key]
            lines.append(f"| {branch} | {label} | {m['clean_eer']:.2%} | {m['stress_eer']:.2%} |")
    lines += ["", "A→B는 추론 입력만 분리했을 때의 변화이고, B→C는 분리 입력으로 다시 학습한 효과다. C는 동일한 검증 선택 기준으로 고른 checkpoint이며, 선택 효과를 분리할 수 있도록 epoch 2 고정 결과도 comparison.json에 저장했다.",
              "일반/변형 검증 파일은 분기별 동일한 256개다. 변형 조건은 분리한 stem에 증강을 적용하므로, 원본 mix에 잡음을 넣었을 때의 분리기 성능을 측정한 결과는 아니다. A는 원본 입력에 같은 증강을 적용한 결과다.",
              "공개 모델 사전학습 데이터와의 중복, ASV 위주 음성 검증의 범위 제한이 있다. DACON 점수 또는 독립 일반화 성능으로 해석하지 않는다.",
              "학습 변경 대상은 음악 AASIST+Linear, 음성 Linear probe이며 인코더는 고정했다. 기존 제출 ZIP은 변경하지 않았다.", ""]
    (WORK/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    main()
