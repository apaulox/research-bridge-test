# DACON 음원 진위 판별 실험

현재 작업은 **EAT + XLS-R에 HTDemucs 두 분기를 적용한 실험**입니다.
아래에서 필요한 작업의 폴더만 열면 됩니다.

| 폴더 | 언제 보나요? |
|---|---|
| [aggressive_audio](aggressive_audio/) | 현재 모델 학습, 증강 변경, Demucs 비교, 제출 패키징 |
| [common](common/) | 공통 오디오 처리, 데이터 다운로드, ZIP 제작 |
| [environment](environment/) | WSL·CUDA·패키지 설정, 다른 PC로 환경 이전 |
| [archive](archive/) | 이전 CLAM·MusicDET·앙상블 실험 확인 |
| [results](results/) | 이전 제출 검증·보고서·ZIP 해시 확인 |

## 자주 하는 작업

| 작업 | 파일 |
|---|---|
| Demucs 분리 후 재학습 | [stem_campaign.py](aggressive_audio/stem_campaign.py) |
| 학습 조건 / 증강 변경 | [run.py](aggressive_audio/run.py) / [augment.py](aggressive_audio/augment.py) |
| Demucs 효과 비교 | [compare_stem_training.py](aggressive_audio/compare_stem_training.py) |
| 최신 결과 보기 | [RESULTS.md](aggressive_audio/stem_experiment_medium_v1/RESULTS.md) |
| 중단 후 재개 | [RESUME_DEMUCS_MEDIUM.md](aggressive_audio/RESUME_DEMUCS_MEDIUM.md) |
| 제출 패키지 구성 | [package_demucs.py](aggressive_audio/package_demucs.py) |
| ZIP 만들기 | [build_dacon_submission_zip.py](common/build_dacon_submission_zip.py) |
| ZIP 추론 검증 | [verify_demucs_zip.py](aggressive_audio/verify_demucs_zip.py) |

현재 실험의 결과 기록은 재개 경로를 유지하기 위해 `aggressive_audio/` 안에 함께 둡니다.
모델·음원·학습 체크포인트·ZIP은 Git에 포함되지 않습니다.
패키징에는 로컬 base 모델, 학습 결과와 기존 제출 폴더가 필요합니다.
다른 PC에서 실행하려면 [환경 이전 안내](environment/GIT_ENVIRONMENT.md)를 확인하세요.

과거 파일의 이동 위치는 [file_moves.json](results/file_moves.json)에 기록했습니다.
기존 제출 ZIP 및 모델 가중치는 이번 폴더 정리로 변경하지 않았습니다.
