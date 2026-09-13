# 이전 실험

현재 EAT+XLS-R 실험을 실행할 때 이 폴더의 파일들을 차례로 실행할 필요는 없습니다.

| 이름 | 용도 |
|---|---|
| `script_clam_*`, `prepare_clam_*`, `verify_clam_*` | CLAM 이전 제출 버전 |
| `script_eat75_clam25*`, `build_ensemble_v3.py` | EAT+CLAM 앙상블 |
| `*musicdet*` | MusicDET 데이터 준비·학습·추론·평가 |
| `audit_*`, `diagnose_*`, `inspect_*` | 이전 오류 및 동작 분석 |
| `research/clam/` | CLAM 연구 코드 |
| `vendor/MusicDET/` | MusicDET 원본 코드 및 라이선스 |

보고서와 제출 검사 결과는 `results/`에 있습니다.
과거 실험에는 당시의 WSL 데이터·모델·패키지 경로가 필요합니다.
이번 정리는 과거 실험 전체를 다시 실행해 재현했다는 의미는 아닙니다.
