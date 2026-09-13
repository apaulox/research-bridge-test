# EAT75 + CLAM25 FMC v3

- 비교 기준: `submit_eat75_clam25_fmc_v2.zip`.
- 변경: CLAM head를 파일별 추론에서 배치 16으로 전환. ID 정렬 후 seed 42로 그룹을 고정하고 마지막 잔여 배치도 유지.
- 보정: 새 배치 조건의 출력으로 TTM01 dev 100 real / 100 fake에서 로지스틱 보정을 다시 적합. 모델 가중치 추가 학습은 없음.
- 유지: EAT 가중치·전처리·75:25 logit 결합·Voice·존재 탐지·Demucs·FILE MAX·제출 인터페이스·requirements.
- 입력: v2의 원본 mix → 16 kHz 공통 로딩 유지. MERT는 여기서 24 kHz로 재표본화. 각 CLAM 입력은 첫 90초 사용 / 짧으면 뒤에 0을 채움.
- 공식 구현과 차이: 공식 native-rate 로딩 대신 기존 v2 입력 경로를 유지하고, 배치 구성을 재현 가능하게 고정. FMC 점수 보정과 EAT 결합은 DACON용 변경.
- 한계: 공식 head의 파일 간 attention은 남아 있음. 배치 축 버그를 제거하거나 재학습한 버전이 아님. 같은 파일도 함께 들어오는 파일 집합이 달라지면 CLAM 점수가 달라질 수 있음.
- 데이터: 기존 MusicCaps real / FakeMusicCaps fake 음악 파일을 보정·평가에 사용. 분리 stem 아님. 사용 subset의 vocal 포함 여부와 성분별 voice/music 생성 라벨은 이 검증에서 확인하지 않았음.
- 평가: TTM01 eval 100×2 및 TTM02–05 eval 각각 50×2. dev와 파일명 중복 없음. EAT는 변경이 없어 기존 점수를 재사용. eval로 결합 비중·seed를 탐색하지 않음.
- FILE 공식이 같아도 Music 출력 변경은 FILE 결과에 반영될 수 있음. 로컬 Music EER로 DACON File/Voice/Music EER을 확정할 수 없음.

최종 보정 계수·로컬 수치·ZIP SHA256은 `eat75_clam25_fmc_v3_manifest.json`에 저장.
실행 smoke 출력은 `eat75_clam25_fmc_v3_smoke.csv`에 저장. 실제 DACON 제출 점수는 미확인.

로컬 결과:

- 새 보정: `sigmoid(0.4315278838792132 × raw_logit − 6.293195255530049)`.
- 기존 보정 계수 −0.829808에서 양수로 바뀜. 배치 조건을 바꾼 뒤 기존 반전 보정을 재사용하지 않은 이유.
- 별도 eval CLAM EER: TTM01 0.34 / TTM02 0.32 / TTM03 0.22 / TTM04 0.40 / TTM05 0.18.
- EAT와 v3 결합의 Music EER은 위 eval 모두 0.00. 이 subset에서는 EAT 대비 개선을 확인할 수 없음.
- 위 값은 로컬 음악 파일 라벨 기준이며, DACON 제출 EER이 아님.

WSL 환경 진입:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'source /home/huskypaul/miniforge3/etc/profile.d/conda.sh; conda activate dacon-cu128; cd /home/huskypaul/dacon; exec bash'
```

완성된 v3를 WSL에서 실행할 때:

```bash
cd /home/huskypaul/dacon/submissions/eat75_clam25_fmc_v3
python script.py --test-dir /실제/오디오/폴더 --sample-submission /실제/sample_submission.csv --output output/submission.csv
```
