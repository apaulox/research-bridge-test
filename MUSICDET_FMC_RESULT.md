# MusicDET-FMC real-only 결과

실행일: 2026-09-08  
환경: WSL Ubuntu 24.04, `dacon-cu128`, CUDA, RTX 5080 16GB

## 데이터

- FakeMusicCaps fake: protocol에서 요구한 27,465개 전부 확보
- MusicCaps real: 5,493개 중 5,147개 확보
- 변환된 16kHz mono float WAV: 32,612개
- TTM01 paired dev: real 510개 + fake 510개
- 생성기: TTM01 MusicGen, TTM02 MusicLDM, TTM03 AudioLDM2,
  TTM04 Stable Audio Open, TTM05 Mustango

## 학습

- 모델: MusicDET SpecNF, `K=2`, `L=1`, `R=5`
- 학습 입력: MusicCaps real만 사용
- 학습 길이: 64,600 samples, 16kHz
- seed: 688
- epoch: 10
- checkpoint 선택: fake 방향을 고정한 paired TTM01 dev EER 최솟값
- 최적 epoch: 4
- 최적 dev EER: 0.3490196078

체크포인트:

```text
/home/huskypaul/dacon/musicdet/runs/musicdet_fmc_real_only_seed688_v1
```

## FMC eval

| 생성기 | EER | Fake AUC |
|---|---:|---:|
| MusicGen | 0.3580 | 0.6835 |
| MusicLDM | 0.4508 | 0.5618 |
| AudioLDM2 | 0.5895 | 0.3782 |
| Stable Audio Open | 0.3168 | 0.7473 |
| Mustango | 0.4696 | 0.5567 |
| 평균 EER | 0.4369 | - |

AudioLDM2에서는 점수 방향이 반대로 나타났다. 현재 모델 하나로 모든 생성기를
안정적으로 일반화한다고 보기 어렵다.

## 확률 보정

FMC TTM01~TTM05 dev에서 NLL을 `P(fake)`로 변환하는 로지스틱 보정을 학습했다.
공유 real은 생성기별로 반복하고 `class_weight=balanced`를 사용했다.

```text
P(fake) = sigmoid(4.038366120662576 * NLL - 5.510506263478469)
```

손대지 않은 전체 eval의 pooled Fake AUC는 0.5855002638이다. 이 보정은 제출용
확률 범위를 정하기 위한 고정 변환이며 생성기 일반화 문제를 해결하지는 않는다.

## DACON 베이스라인 통합

WSL 베이스라인에 다음 파일을 별도로 배치했다.

```text
/home/huskypaul/dacon/baseline/script_musicdet_fmc.py
/home/huskypaul/dacon/baseline/model/musicdet_fmc/
```

voice branch, PANNs, HTDemucs, MAX fusion은 유지한다. music branch만 원본 믹스를
MusicDET에 입력하고 고정 로지스틱 보정을 거쳐 `MUSIC_FAKE_PROB`를 만든다.
기본 전처리는 공식 평가와 같은 center crop/zero-pad다.

실행:

```bash
cd ~/dacon/baseline
conda activate dacon-cu128
python script_musicdet_fmc.py --musicdet-mode center
```

3개 샘플 smoke test가 통과했고, 확률 범위·열 순서·MAX fusion 식을 검증했다.
기존 `script.py`와 `output/submission.csv`는 덮어쓰지 않았다.
