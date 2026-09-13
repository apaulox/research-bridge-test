# Demucs 두 분기 Medium 실험 재개

현재 상태: 두 분기 2-epoch 학습과 A/B/C 비교까지 완료. 결과는 `stem_experiment_medium_v1/RESULTS.md` 및 `comparison.json`에 저장했다. 완료된 학습을 다시 시작할 필요는 없다. 아래 명령은 복구·재현용이다.

최신 사용자 결정: 원본 mix를 음악 모델에 넣지 않는다. Demucs vocals는 음성 분기, drums+bass+other 합산은 음악 분기로 들어간다.

- 분리: HTDemucs, 44.1kHz stereo, mono 평균·표준편차로 표준화, shifts=0, overlap=0.25, 역표준화 후 각 분기를 16kHz mono로 변환.
- 음악: EAT 고정, AASIST+Linear 학습. 분리된 반주 4초 입력.
- 음성: XLS-R 1B 고정, Linear probe 학습. 이전 실험과 똑같이 분리된 vocals 64,000 samples 학습 입력.
- 각 분기: real/fake 각 512개 학습, 각 128개 검증, 2 epochs. 공식 원래 모델에서 시작.
- 최신 추가 지시: Demucs 유무만 비교한다. 이전 music_head_channel_1024_v1 / speech_probe_channel_v1의 학습·검증 ID와 seed, 초기 모델, 인코더 고정, lr=1e-4, batch=2, accumulation=8, augmentation, epoch를 유지한다. 코드에서 일치를 assertion으로 확인한다. 음성 64,600-sample 학습 변경안과 Light 실험은 실행하지 않는다.
- 증강: 기존 channel 설정, 25% clean / 나머지 1–3개, 분리된 파형에 적용. 검증 stress도 분리 후 증강이므로 잡음 상황에서 Demucs 자체의 견고성을 측정하는 것은 아니다.
- CPS ResNet38과 mean 결합은 유지. 이번 요청 범위는 재학습 실험이며 제출 ZIP을 자동으로 덮어쓰지 않는다.

## 재개 명령

먼저 같은 stem_campaign.py 프로세스가 아직 실행 중인지 확인한다. 실행 중이면 중복 시작하지 않는다.

```powershell
wsl -d Ubuntu-24.04 -- /home/huskypaul/miniforge3/envs/dacon-cu128/bin/python -B /mnt/c/Users/husky_gp7j99y/Documents/ChatGPT/DACON/aggressive_audio/stem_campaign.py
```

`stem_experiment_medium_v1/progress.json`에 진행 상태, 같은 폴더의 로그에 학습 진행이 남는다.
분리된 WAV와 원본·체크포인트 fingerprint는 `stem_experiment_v1/music`, `stem_experiment_v1/speech`에 공유 캐시로 저장된다. 재개 시 유효한 파일은 건너뛴다.

학습 체크포인트:

- `runs/music_demucs_medium_v1/checkpoint_latest.pt`
- `runs/speech_demucs_medium_v1/checkpoint_latest.pt`

가중치 delta, optimizer, Python/NumPy/Torch/CUDA 난수 상태, 다음 epoch/step을 저장한다. 고정된 공식 base 모델과 데이터 캐시도 재개에 필요하다. 중단 시 마지막 저장 경계 이후 작업만 다시 수행한다.

분리 단계에서 안전하게 멈추려면 `stem_experiment_medium_v1/STOP` 파일을 만든다. 학습 단계에서는 현재 run 폴더에도 `STOP`을 만들어 다음 optimizer 갱신 경계에서 저장하고 종료한다. 다시 시작할 때 해당 STOP 파일을 다른 이름으로 바꾼다.

`stem_experiment_v1/STOP`은 이전 큰 실험을 Light로 전환할 때 남긴 파일로, 현재 Medium 실행의 정지 신호가 아니다. Light 학습은 사용자가 실행을 취소해 시작되지 않았다.

완료 결과는 `stem_experiment_medium_v1/training_results.json`에 저장된다. 원본 mix와 분리된 반주는 입력 분포가 다르므로 이전 검증 수치와 단순 비교하지 말고, 같은 분리 입력에서 모델들을 비교한다.

학습 완료 후 아래 명령으로 A(기존 모델·기존 입력), B(기존 모델·분리 입력), C(재학습 모델·분리 입력)를 비교한다. 결과는 같은 폴더의 `comparison.json`, `RESULTS.md`에 저장된다.

```powershell
wsl -d Ubuntu-24.04 -- /home/huskypaul/miniforge3/envs/dacon-cu128/bin/python -B /mnt/c/Users/husky_gp7j99y/Documents/ChatGPT/DACON/aggressive_audio/compare_stem_training.py
```
