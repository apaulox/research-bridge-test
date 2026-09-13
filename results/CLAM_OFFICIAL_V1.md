# CLAM official 단독 제출

- 제출 파일: `submit_clam_official_v1.zip`
- Music: 공식 MoM-CLAM `best_model_triplet_loss_margin_0.2.pth` 그대로.
- Backbone: `m-a-p/MERT-v1-95M` + `m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres`.
- 데이터: 공개 MoM 학습 체크포인트를 사용. 이번 실험에서 추가 학습·FMC 보정·가중치 선택은 하지 않음. 공개 체크포인트의 정확한 학습 파일 목록은 미확인.
- 공식 학습 코드의 라벨은 real=0, fake=1. Music 확률은 `sigmoid(raw_logit)`.
- EAT 결합과 FMC에서 학습한 음수 기울기·절편을 제거.
- 원본 mix를 각 backbone의 샘플레이트로 직접 변환: MERT 24kHz, Wav2Vec2 16kHz. Mono, 앞 90초, 짧으면 뒤에 zero-padding.
- 이전 앙상블의 공통 16kHz 중간 변환도 제거하여 공식 전처리에 맞춤.
- Voice DF-Arena, PANNs, HTDemucs와 `FILE=max(VP*VF, MP*MF)`는 유지. CLAM Music 점수는 FILE에도 반영.
- 의존성은 작동한 baseline/v2와 동일. EAT 모델 및 timm 의존성 없음.

## WSL 실행

```bash
source ~/dacon/activate.sh
python ~/dacon/submissions/clam_official_v1/script.py \
  --test-dir ~/dacon/baseline/data/test \
  --sample-submission ~/dacon/baseline/data/sample_submission.csv \
  --output ~/dacon/clam/validation/official_v1/manual_submission.csv
```

위 명령은 제공된 로컬 샘플의 동작 확인용이다. 실제 별도 평가 데이터가 있으면 `--test-dir`과 `--sample-submission`을 해당 경로로 지정한다.

## 검증 범위

- 공식 repository head와 출력 일치, 공식 전처리와 일치.
- 공개 repository 체크포인트와 제출 체크포인트 SHA-256 일치.
- 샘플 전체 파이프라인 실행, ID/열 순서·확률 범위·공식 sigmoid 방향·FILE MAX 식 검증.
- 제공된 샘플 3개는 SHA-256이 동일한 오디오 복제본이다. 이 검증은 성능 평가가 아니다.
- 모델 출처·해시·검증 결과: `clam_official_v1_manifest.json`.
- 샘플 결과: `clam_official_v1_smoke.csv`.

공식 코드: https://github.com/StarkVision-AI/MoM-CLAM/tree/db7fff06e25415c119e58b8e4c4f48c840a845c8

이번 제출은 공식 모델의 원래 점수 방향을 확인하는 실험이다. 기존 FMC 검증에서는 원래 방향이 역전되어 있었으므로, 이전의 보정된 CLAM 단독 EER 0.2233을 이번 설정의 성능으로 인용하면 안 된다.
