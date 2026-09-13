# DACON 증강 실험 · 재개 안내

> 최신 Demucs 두 분기 실험은 [재개 안내](RESUME_DEMUCS_MEDIUM.md)와
> [비교 결과](stem_experiment_medium_v1/RESULTS.md)를 보세요.
> 아래 본문은 2026-09-12의 원본 mix 기반 실험 기록입니다.
> 전체 폴더 안내는 [저장소 첫 화면](../README.md)에 있습니다.

현재 작업: EAT 단독 음악 분기, AntiDeepfake XLS-R-1B(wav2vec 2.0 계열)+linear probe 음성 분기, 동일 성분 예측을 사용한 mean/max FILE 비교.

2026-09-12 상태: 계획한 대조 학습과 64개 외부 파일의 전체 로컬 추론을 완료했다. 선택 모델은 `music_head_channel_1024_v1`과 `speech_probe_channel_v1`이다. mean/max FILE EER는 모두 17.0833%로, 이번 진단에서 mean 우위는 관찰되지 않았다. 상세 결과는 `RESULTS_2026-09-12.md`에 있다. 실행 중인 학습은 없으며 다음 작업은 더 독립적인 검증과 제출 환경·이용조건 확인이다.

## 재개

Windows PowerShell에서 다음 명령으로 이 실험 묶음의 미완료 실행을 이어간다. 완료된 실행은 건너뛰며 기존 결과를 덮어쓰지 않는다.

```powershell
wsl -d Ubuntu-24.04 -- /home/huskypaul/miniforge3/envs/dacon-cu128/bin/python -B /mnt/c/Users/husky_gp7j99y/Documents/ChatGPT/DACON/aggressive_audio/continue_campaign.py
```

`runs/<실험명>/checkpoint_latest.pt`는 학습한 가중치, AdamW optimizer, Python/NumPy/PyTorch/CUDA 난수 상태, 다음 epoch/step, 학습 중 loss, 증강 횟수, 검증 이력, 설정·데이터 명단을 저장한다. optimizer 갱신 경계에서 주기적으로 atomic 저장한다. `resume_status.json`으로 재개 지점을 읽을 수 있다.

가중치는 학습 대상 부분과 관련 buffer의 delta이다. 고정된 공식 base model 전체를 중복 저장하지 않으므로 재개 시 기존 EAT runtime/base, `assets/speech_xlsr1b`, 원본 데이터 경로와 같은 환경이 필요하다. 다른 노트북으로 `.pt` 하나만 옮기면 재개할 수 없다.

학습 묶음이 완료되면 다음 명령으로 같은 검증 표본에 기반한 모델 선택과 64개 외부 음원의 mean/max 추론 비교를 실행한다. 결과는 `RESULTS_2026-09-12.md`, `records/campaign_selection.json`, `fusion_diagnostic/predictions_v1`에 저장된다.

```powershell
wsl -d Ubuntu-24.04 -- /home/huskypaul/miniforge3/envs/dacon-cu128/bin/python -B /mnt/c/Users/husky_gp7j99y/Documents/ChatGPT/DACON/aggressive_audio/finalize_campaign.py
```

안전하게 멈추려면 실행 중인 run 폴더에 `STOP` 파일을 만든다. 다음 optimizer 갱신 경계에서 저장하고 종료한다. 전체 묶음의 다음 실험 실행까지 막으려면 이 폴더에 `STOP_CAMPAIGN`도 만든다. 다시 진행할 때는 해당 STOP 파일을 `STOP.handled` 등 다른 이름으로 옮긴 뒤 위 명령을 실행한다. 프로세스를 강제 종료하면 마지막 저장 경계 이후의 작업은 재실행된다.

`records/resume_verification.json`: 실제 중단·재개한 작은 EAT 실행과 연속 실행의 최종 가중치가 bitwise identical임을 확인했다. 이것은 기능 검사이며 성능 근거가 아니다.

기존 `*_pilot_v1`과 `music_last2_none_v1`에는 가중치만 담긴 `best_adaptation.pt`/`last_adaptation.pt`가 있다. optimizer가 없으므로 이 파일을 정확한 학습 재개용으로 표현하면 안 된다. `music_last2_channel_v1`은 epoch 1 중간에 중단됐고, 저장된 것은 epoch 0이다. 같은 설정으로 시작한 `music_last2_channel_v2`가 후속 실행이다.

## 모델·데이터

- 음악: DeepFense/FakeMusicCaps_EAT_AASIST_NoAug_Seed2의 공식 공개 가중치. 기존 로컬 EAT runtime을 재사용하며 CLAM/MusicDET 앙상블은 없다. 특징 추출기 고정+분류부 학습 및 마지막 2개 transformer 층 추가 학습을 각각 비교한다.
- 음성: nii-yamagishilab/xls-r-1b-anti-deepfake, revision `60c96543210117bdb256636e7961e16463f99291`. encoder 고정, temporal mean+Linear(1280,2) probe 학습. 공식 checkpoint는 이미 RawBoost (1)+(2)로 후속학습된 모델이다. 여기서 `none`은 우리의 추가 학습에서 증강을 끈다는 뜻이다. 두 모델 모두 출력 0번이 fake이다.
- FMC/MusicCaps: 로컬 실제 5,147개, 생성 27,465개. 원본 ID를 기준으로 분할해 생성기별 변형이 같은 split에 속한다. 공개 EAT 제작자의 정확한 학습 목록을 복원한 것은 아니다. upstream checkpoint와의 중복은 불명확하다.
- ASVspoof 2019 LA: 첫 실험용 train real/fake 각 1,024개, dev 각 256개. 공식 speaker split 유지. PA의 replay label을 fake로 바꾸지 않는다.
- CoSG: 공식 CodecFake+의 실제·생성 음성 1,797개, 17개 source-model 그룹. CoRS 코덱 round-trip은 제외한다. 원본 화자·prompt 연결이 완전하지 않아 CoSG 전체를 train 쪽에만 넣고, 임의 분할을 독립 검증이라고 주장하지 않는다. ASV-only dev는 현대 코덱·복제에 대한 일반화를 입증하지 못한다.

각 run에서 실제로 선택한 파일 ID는 `config.json`, 전체 출처·라벨·분할은 `manifests/*.jsonl`에 있다. 범주 균형 표본을 사용하므로 전체 명단의 모든 파일을 각 run이 사용한 것은 아니다.

## 증강과 라벨

전화망 시뮬레이션, MP3/Opus, 합성 impulse response 기반 replay 시뮬레이션, 잡음, 리샘플링, 음량, 비생성형 spectral attenuation을 구현했다. clean 25%, 나머지는 1~3개 변형을 무작위 적용한다. 실제 장비로 재생·재녹음한 데이터나 생성형 음질 개선 모델을 사용한 것으로 표현하면 안 된다.

음성 overlay는 별도 대조 실험이다. 음악 진위 라벨은 유지하고, 배경 노래의 보컬 라벨을 추정하지 않는다. TTS/VC/복제/코덱 생성물은 공개 데이터의 출처·생성 라벨로 들여오며, 필터를 적용했다고 real을 fake로 바꾸지 않는다.

## 결합과 검증 범위

`infer.py`는 기존 PANNs·Demucs 경로를 유지하며 음악은 원본 mix의 처음 4초, 음성은 vocals stem의 기존 구간/max 집계로 처리한다. FILE만 아래처럼 비교한다.

```text
mean = (VP * VF + MP * MF) / 2
max  = max(VP * VF, MP * MF)
```

두 CSV의 네 성분 확률은 동일한 캐시를 사용한다. 존재 확률로 나누는 가중평균이나 성분의 raw 평균으로 바꾼 것이 아니다. max는 수학적으로 대칭이며 mean은 한쪽 분기의 오탐뿐 아니라 실제 fake 신호도 낮출 수 있다.

EER는 순위와 임계값 이동에 기반하므로 평균의 `/2` 자체는 EER를 개선하지 않는다. 여기서 확인하는 효과는 두 분기의 합이 max와 달리 파일 순위를 바꾸는지이다. 확률이 낮아졌다는 사실만으로 오탐 개선을 선언하지 않는다.

현재 `infer.py`는 64개 외부 파일의 전체 추론을 통과한 로컬 실험 runner이다. 별도 portable ZIP 패키징과 서버 검증 전에는 DACON 제출 가능 파일이라고 표현하지 않는다. 실제 DACON 비공개 평가 데이터 학습·보정, 파일 간 attention·통계 보정은 사용하지 않는다.

## 출처·이용조건

- 규칙: https://dacon.io/competitions/official/236749/overview/rules
- 라벨: https://dacon.io/competitions/official/236749/overview/description
- EAT: https://huggingface.co/DeepFense/FakeMusicCaps_EAT_AASIST_NoAug_Seed2
- 음성 모델: https://huggingface.co/nii-yamagishilab/xls-r-1b-anti-deepfake (CC-BY-NC-SA-4.0)
- FMC: https://zenodo.org/records/15063698 (CC-BY-NC-4.0, 공식 API 확인)
- MusicCaps: https://huggingface.co/datasets/google/MusicCaps (metadata CC-BY-SA-4.0가 개별 YouTube 음원 권리를 포괄하지 않음)
- ASVspoof: https://zenodo.org/records/6906306 (ODC-By; README와 라이선스 보관)
- CoSG: https://huggingface.co/datasets/CodecFake/CodecFake_Plus_Dataset (공식 카드 MIT; 원본 demo 출처도 유지)

FMC 초기 파일 목록의 CC-BY 표기는 CC-BY-NC로 정정했다. 음원과 실험 결과는 바뀌지 않았다. `records/music_manifest_license_correction.json`에 기록했다. MusicCaps 개별 음원과 CoSG 원본 demo의 이용조건은 최종 제출 자료를 만들 때까지 정리해야 하며, 현재 기록만으로 모든 데이터의 재배포 권한이 확인됐다고 선언하지 않는다.

ASVspoof 미러 ZIP 전체 MD5는 공식 ZIP과 다르다. 공식 ZIP의 122,316개 공통 파일 CRC·크기는 모두 같았으며, 원본과 대응되는 파일만 추출한다. 추가 Mac 메타데이터를 데이터로 사용하지 않는다. `records/audit-mirror.json`에 공식 member index가 있다.

공식 음성 구현은 연구팀이 지정한 fairseq commit `862efab86f649c04ea31545ce28d13c59560113d`를 사용한다. 기존 PyTorch 2.7.1+cu128을 유지했다. 구버전 fairseq의 NumPy builtin alias 호환 보정은 `models.py`에 명시되어 있다. 이 학습 환경의 Python은 3.10이며 DACON 서버 Python 3.11 호환성은 별도 검증 대상이다.
