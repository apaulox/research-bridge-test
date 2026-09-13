# CLAM official v2: 공식 배치 16 재현

제출 ZIP: `submit_clam_official_v2.zip`

v1의 단일 파일 head 실행을 공식 학습·평가의 batch size 16에 맞춘 실험이다. 공개 가중치, 공식 점수 방향, 음악 입력은 유지한다. **공식 모델에 있는 파일 간 attention을 제거한 모델은 아니다.**

## 변경 이유와 범위

공식 head는 `batch_first=False`인데 `[B, 1, 768]`을 입력한다. 이에 따라 같은 배치의 파일들을 sequence로 처리한다. 공개 가중치는 이 연산으로 학습됐고 공식 DataLoader는 batch size 16을 사용한다. v1은 B=1로 실행해 평가 조건이 달랐다.

단순히 `batch_first=True`로 바꾸는 수정은 v1의 B=1 출력과 수학적으로 같다. 로컬 가중치 검사에서 단일 파일 logit 차이가 0이었다. 따라서 v2에서는 공개 가중치를 유지하면서 공식 배치 동작을 재현한다. 학습된 파일 간 의존성을 제거한 새 모델의 성능 검증은 별도 과제다.

| 항목 | v2 처리 / 공식 코드 대조 |
|---|---|
| CLAM head | 공개 가중치와 forward 연산 유지, 16개씩 실행 |
| 마지막 배치 | 남은 파일도 처리, padding/drop 없음 |
| 배치 순서 | ID 정렬 후 별도 torch Generator(seed=42)로 shuffle. 공식 평가의 shuffle=True를 재현 가능하게 고정한 의도적 차이 |
| 출력 매핑 | ID로 복원하여 sample_submission의 행 순서 유지 |
| 확률 | 공식 `torch.sigmoid(float32 logit)` 사용. v1의 float64 sigmoid에서 변경; 0/1 근처에서 반올림·동점이 달라질 수 있음 |
| 입력 | 원본 mix, torchaudio native decode, MERT 24kHz / Wav2Vec2 16kHz로 직접 resample |
| 길이·채널 | mono, 앞 90초, 짧으면 뒤쪽 zero-padding |
| 특징 | 13개 hidden state의 시간 평균, MERT/Wav2Vec2 순서와 conv의 첫 채널 선택을 공식 그대로 유지 |
| 모델 모드 | encoder/head eval, float32, gradient 비활성화 |
| 로딩 | 제출 ZIP 내부 로컬 가중치만 사용. 다운로드 없음 |
| 예외 처리 | DACON baseline의 실패 처리 유지. 공식 데이터 준비 코드의 파일 건너뛰기는 복제하지 않음 |
| 학습·보정 | 추가 학습, FMC 보정, 방향 반전, EAT 결합 없음 |
| 의존성 | baseline requirements 그대로. timm 코드·설치·runtime import 없음 |

배치 구성에 따라 Music 점수가 바뀔 수 있으므로 입력 파일 집합을 바꾸면 동일 파일의 점수도 달라질 수 있다. 단순히 CSV 행 순서만 바꾸는 경우에는 내부 ID 정렬로 동일한 배치 구성을 유지한다. seed를 고정해도 원래 공식 평가 당시의 배치 구성과 같다는 뜻은 아니다.

## DACON baseline 보존

원본 `/home/huskypaul/dacon/baseline/script.py`와 직접 대조했다.

- 음성/음악 존재 탐지 PANNs, 4.0375초 구간 처리, HTDemucs, 음성 DF-Arena와 입력·집계 유지.
- `FILE=max(VP*VF, MP*MF)` 유지. CLAM의 새로운 MF는 FILE에 반영.
- 5개 확률 컬럼, ID 매핑, 입력 확장자, CLI 경로 옵션, 출력 반올림 유지.
- 위 로직을 포함한 함수 19개가 AST 기준 동일하다. 실행 연결(main), 음악 점수 연결, CLI 설명만 의도적으로 다르다.
- Music shuffle은 별도 RNG를 사용하므로 baseline의 전역 RNG 상태를 변경하지 않는다.

Voice 코드와 baseline 모델 자산 20개가 동일함을 별도로 확인했다. 원본 baseline 자체의 반복 실행에서도 Voice 확률 차이가 관측됐으며, 같은 모델·같은 입력에서도 변동이 있었다. 따라서 보존 검사는 Voice 입력 파형의 SHA-256 동일성과 코드·가중치 동일성을 기준으로 한다. 실행 간 확률 차이는 manifest에 그대로 기록하며, bitwise 동일한 Voice 확률을 보장한다고 해석하지 않는다.

## 검증

- 공식 head와 배치별 확률 대조: 1, 3, 16, 17, 33개 입력, 나머지 배치 및 ID 순서 검사.
- 실제 공식 extractor 함수와 제출 extractor의 MERT/Wav2Vec2 임베딩 대조.
- 17개 서로 다른 로컬 오디오에서 v2와 원본 baseline 전체 추론을 실행해 Voice·존재 확률 및 FILE 공식을 비교.
- 비교 검사에만 결정적 GPU 연산을 적용하며, 이 설정을 제출 코드에 추가하지 않는다.
- timm 미설치 모사 및 import 차단, 새 Hugging Face 모듈 캐시 사용.
- ZIP CRC 및 복사본 SHA-256 확인. 세부 결과는 `clam_official_v2_manifest.json` 참조.

17개 오디오는 기존 FMC eval 파일에서 동작 검증용으로만 선택했다. 학습·보정·배치 크기 또는 seed 최적화에 사용하지 않았다. 이 검증은 리더보드 성능 평가가 아니다.

## WSL 실행

```bash
source ~/dacon/activate.sh
python ~/dacon/submissions/clam_official_v2/script.py \
  --test-dir ~/dacon/baseline/data/test \
  --sample-submission ~/dacon/baseline/data/sample_submission.csv \
  --output ~/dacon/clam/validation/official_v2/manual_submission.csv
```

제공된 baseline 샘플 3개는 같은 오디오의 복제본이므로 배치 변경의 성능 차이를 판정하는 용도로 사용할 수 없다.

공식 코드 기준: https://github.com/StarkVision-AI/MoM-CLAM/tree/db7fff06e25415c119e58b8e4c4f48c840a845c8
