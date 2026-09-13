# MusicDET-FMC 실험 준비

## 현재 실행 상태 (2026-09-07)

WSL `/home/huskypaul/dacon/musicdet`에서 다음 작업이 백그라운드 실행 중이다.

1. Zenodo의 공식 `FakeMusicCaps.zip` 12.9GB 다운로드 및 MD5 검사.
2. 공식 MusicCaps 분할의 실제 음악 5,493개 구간 다운로드.
3. 완료 후 실패 구간 2회 재시도.
4. 공식 방식의 16kHz mono float WAV 변환 및 분할별 가용률 기록.
5. TTM01 dev의 real/fake를 같은 MusicCaps ID 기준으로 짝맞춰 real-only MusicDET 10 epoch 학습.
6. TTM01~05(MusicGen, MusicLDM, AudioLDM2, Stable Audio Open, Mustango)별 고정 방향 EER 평가.

진행 상태는 Ubuntu에서 다음 명령으로 확인한다.

```bash
bash ~/dacon/musicdet/musicdet_status.sh
```

주 로그는 `/home/huskypaul/dacon/musicdet/logs/continuation.log`이다.
최종 학습 폴더는 `/home/huskypaul/dacon/musicdet/runs/musicdet_fmc_real_only_seed688_v1`이다.
공개 원본 `train.py`의 checkpoint 선택 문제를 피하기 위해 모든 epoch를 보존하고,
`real score = -NLL` 방향을 고정한 dev EER 최저 모델을 최종 checkpoint로 선택한다.

확인일: 2026-09-07. 실행 대상: GPU 서버 / Linux.

현재 상태는 **공식 소스 확보 + 원시 점수 추출 코드 준비**다. 학습 완료 가중치,
실제 오디오 추론 결과, DACON 제출 파일은 아직 없다. 서버 접속 정보와 기존 베이스라인
파일도 이 작업 폴더에는 없다. 로컬에는 PyTorch가 없어 신경망 실행 검증은 하지 못했다.

## 1. 어떤 데이터로 학습했는가

| 후보 | 탐지기 학습 데이터 | 확보 상태 |
|---|---|---|
| CLAM | MoM의 real/fake. 논문에는 SONICS 실험도 있음 | 공식 저장소에 약 22MB 탐지기 가중치 존재. FMC 버전은 확인 못 함 |
| MusicDET-FMC, real-only | FMC 평가 설정의 **실제 음악**만으로 학습. 기반 real source는 MusicCaps | 공식 코드 있음. 학습 완료 가중치는 확인 못 함 |
| Class-Conditional MusicDET-FMC | 실제 음악 + FMC 생성 음악. 생성기별 학습 실험을 구별해야 함 | 공식 코드 있음. 학습 완료 가중치는 확인 못 함 |
| CLAM + MusicDET | 두 가중치의 학습 이력을 그대로 사용 | 단순 점수 앙상블은 별도 학습 데이터가 없음. 가중치 조정/보정에 외부 검증셋 필요 |

MoM의 fake 학습/검증에는 Suno v2/v3.5, Udio v1.5, DiffRhythm이 사용된다.
Riffusion, Yue, voice cloning, Suno v3/v4 등은 논문의 OOD 평가 대상이다.
공개 CLAM 파일 자체의 정확한 학습 실행 이력은 메타데이터로 확정하지 못했다.
공개 추출 코드의 backbone은 MERT-v1-95M 및
`m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres`이며, 원본 믹스를 입력한다.
[CLAM 논문](https://arxiv.org/html/2512.00621v1),
[공개 가중치](https://github.com/StarkVision-AI/MoM-CLAM/tree/main/model_wts),
[추출 코드](https://github.com/StarkVision-AI/MoM-CLAM/tree/main/extractors)

FMC fake 생성기는 MusicGen, MusicLDM, AudioLDM2, Stable Audio Open, Mustango다.
MusicDET 논문의 real-only와 class-conditional 결과는 서로 다른 학습 설정이다.
FMC라는 이름만으로 어떤 설정/생성기 subset인지 알 수 없다.
[MusicDET 논문](https://arxiv.org/html/2605.18072v1)

MusicDET의 zero-shot은 **생성 음악 없이 실제 음악만으로 학습**하는 설정을 말한다.
학습되지 않은 모델을 그대로 추론한다는 뜻이 아니다. README의 pretrained MERT/XLS-R은
비교 모델의 backbone이며, 그것만 받아도 MusicDET 탐지기가 완성되는 것은 아니다.
[공식 저장소](https://github.com/Chaolei98/MusicDET),
[릴리스 목록](https://github.com/Chaolei98/MusicDET/releases)

## 2. 첫 단계: 서버 환경 확인

이 폴더의 `server_check.py`를 서버에 복사한다. 기존 베이스라인을 실행하던 Python 환경에서:

```bash
python server_check.py --baseline-dir /실제/베이스라인/폴더
```

위 `/실제/베이스라인/폴더`는 예시이므로 실제 경로로 바꾼다. 출력 결과를 공유하면
GPU 메모리와 설치 버전에 맞춰 다음 설치/학습 단계를 정할 수 있다.
이 스크립트는 패키지를 설치하거나 모델을 변경하지 않는다.

파일을 옮기기 전에도 서버에서 다음 정보를 바로 확인할 수 있다:

```bash
pwd
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python --version
python -m pip show torch torchaudio librosa transformers
```

## 3. 가중치 확보 또는 학습

우선 필요한 것은 학습된 `anti-spoofing_feat_model.pt`와 같은 실행의 `args.json`이다.
외부에서 가중치를 받는 경우 FMC 여부, `only_real`, 생성기 subset, split,
코드 버전도 함께 확인한다. 이 작업에서 작성자에게 메시지를 보내지는 않았다.

직접 학습하려면 아래 재료부터 확보한다:

1. MusicCaps real 오디오.
2. 외부 검증용 FMC fake 오디오. Class-conditional 학습이면 학습용 fake도 필요.
3. 공식 train/dev/eval protocol 또는 독립적으로 만든 split.
4. 별도 확률 보정용 데이터. 학습과 동일한 원곡에서 나온 변형은 다른 split으로 나누지 않는다.

[FMC 공식 배포](https://zenodo.org/records/15063698)는 27,605개 생성 음악을 설명하고
12.9GB ZIP을 제공한다. 이것만으로 real MusicCaps까지 모두 확보됐다고 가정하지 않는다.
MusicDET README가 연결한 [protocol 파일](https://drive.google.com/file/d/1CnC8G6Kp6WfF3XcX0tJI6F6RrfAAY7uI/view)은
아직 다운로드하거나 내용/접근성을 검증하지 않았다.

직접 만든 split이면 '논문 공식 결과 재현'으로 부르지 않고 별도 실험으로 기록한다.
첫 real-only 실험은 FR/RF 합성 학습 없이 진행할 수 있다.
공개 `train.py`는 `--only_real`에서도 real/fake가 섞인 dev의 평균 NLL로 checkpoint를 선택하고,
EER도 출력한다. fake 미사용의 범위를 학습 업데이트와 모델 선택으로 구분해 기록해야 한다.
또한 같은 output 폴더를 지정하면 기존 폴더를 삭제하는 코드가 있으므로,
학습 실행 래퍼를 준비할 때 기존 출력 덮어쓰기를 막아야 한다.
현재 학습은 실행하지 않았다.

## 4. 준비한 점수 추출 코드

서버에 이 폴더 전체를 복사하고, 다음 구조로 가중치를 배치한다:

```text
musicdet_probe.py
vendor/MusicDET/...
model/musicdet_fmc/args.json
model/musicdet_fmc/anti-spoofing_feat_model.pt
```

패키지 목록은 `vendor/MusicDET/requirements.txt`에 있다.
서버 환경을 확인한 다음 별도 환경으로 설치한다. 기존 DACON 환경에서 바로 덮어 설치하지 않는다.
코드가 준비됐다는 것과 해당 버전 조합을 서버에서 검증했다는 것은 다르다.

가중치와 환경이 갖춰진 뒤의 첫 실행:

```bash
python musicdet_probe.py \
  --checkpoint-dir model/musicdet_fmc \
  --audio-dir /실제/로컬검증오디오/폴더 \
  --output output/musicdet_fmc_center_probe.csv \
  --device cuda --mode center --limit 8
```

**이 명령은 현재 가중치가 없으므로 아직 실행할 수 없다.**
생성되는 파일도 submission.csv가 아닌 진단용 원시 점수다.

전처리는 공식 소스의 16kHz mono, 64,600 samples, 가운데 crop/zero-pad,
구간별 RMS 정규화를 따른다. 기본 `center`가 공식 평가 전처리와 대응한다.
`sliding-mean`, `sliding-max`는 긴 파일을 위한 별도 실험 옵션이며 공식 재현으로 표시하지 않는다.
무음은 원시 점수와 `LOW_ENERGY` 표시를 남긴다. 임의의 확률 0으로 바꾸지 않는다.

공식 `test.py`의 `-NLL`은 real 방향 점수다. 이 래퍼는 fake 방향인 `+NLL`을
`MUSIC_ANOMALY_SCORE`로 출력한다. **NLL은 0~1 확률이 아니다.**
외부 라벨 데이터에서 방향을 검증하고, 고정된 보정 함수를 학습한 후 MF로 연결한다.
평가할 1,200개 파일에 맞춰 min-max 범위나 보정 파라미터를 다시 학습하지 않는다.
[공식 추론 코드](https://github.com/Chaolei98/MusicDET/blob/main/test.py),
[공식 전처리](https://github.com/Chaolei98/MusicDET/blob/main/dataset.py)

## 5. 베이스라인 연결 계획

첫 비교는 DF-Arena voice, PANNs, HTDemucs, max 결합식을 고정한다.
MusicDET에는 원본 믹스를 넣는 실험부터 시작한다. 분리 반주 입력은 다음 독립 실험으로 둔다.
이유는 공식 모델 입력이 일반 음악이고, 분리 반주가 학습 분포와 같다는 보장이 없기 때문이다.
가중치 확보와 보정 검증 후 아래 지점을 실제 baseline 코드에 연결한다:

```text
원본 오디오 -> MusicDET -> raw NLL -> 고정된 확률 보정 -> MUSIC_FAKE_PROB
FILE_FAKE_PROB = max(VP * VF, MP * MF)
```

CLAM 점수와 앙상블할 때도 먼저 외부 검증셋에서 각 점수의 방향/스케일을 확인한다.
초기 0.5/0.5 평균은 비교 기준일 뿐 최적이라고 가정하지 않는다.
Music EER, File EER, 실행 시간과 개별 모델이 틀리는 샘플의 차이를 함께 본다.

## 6. 실험 해석에서 구분할 것

- 11번은 EAT + AASIST + FMC 조합의 유효성을 뒷받침한다.
  구조와 학습 데이터가 동시에 바뀌었으므로 FMC 단독 효과나 MusicDET 개선을 보장하지 않는다.
- 높은 CPS는 존재 판정 성능이다. Demucs 분리 품질이나 성분 진위 독립 판정을 증명하지 않는다.
  혼합 데이터 학습은 뒤로 미뤄도, 가짜 보컬 때문에 MF가 올라가는지 확인하는 검증은 남겨둔다.
- MF를 바꾸면 max 결합의 FILE도 바뀐다. 총점 하나로 Music/File EER 두 개를 정확히 역산할 수 없다.
  다른 항이 고정이면 `delta score = -0.27 * delta MusicEER - 0.45 * delta FileEER`다.
- 데이터군별 점수 변화는 가설 비교에 쓸 수 있지만 숨은 1,200개 구성비를 유일하게 결정하지 못한다.
  생성기, 보컬 유무, 순수 반주, 코덱, 전화 대역, 길이를 나눈 외부 검증으로 가설을 함께 확인한다.
- 입력을 바꾸는 전처리 실험과 학습 데이터를 바꾸는 실험을 분리하고 한 번에 하나만 변경한다.
- DACON 규칙 페이지는 이번 세션에서 내용을 확인하지 못했다. 제출 전 외부 데이터/가중치와
  리더보드 이용 관련 규칙 원문을 확인해야 하며, 여기서는 대회 제출이나 평가 데이터 학습을 하지 않았다.

## 7. 검증 및 출처 고정

MusicDET 원본 코드: `087a758955a638dca68ef0dcbb81107fce48d60f`.
CLAM 조사 코드: `db7fff06e25415c119e58b8e4c4f48c840a845c8`.
`vendor/MusicDET`는 공식 파일을 보존했다. `research/clam`은 조사에 필요한 코드 일부뿐이며
CLAM 실행 패키지가 아니다.

로컬 검증 범위: Python 문법, CLI help, 환경 확인 스크립트 실행,
누락/잘못된 가중치 설정 거부와 오디오 구간 경계 검사.
검증하지 못한 범위: trained checkpoint 로딩, CPU/CUDA forward,
실제 오디오 디코딩, 예측 성능, 확률 보정, 베이스라인 통합, 제출 시간/용량.
