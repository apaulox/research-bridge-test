# 공통 코드

| 파일 | 용도 |
|---|---|
| `script_eat75_musicdet25_fmc.py` | 현재 실험에서 재사용하는 기존 오디오 읽기·분할·Demucs 함수. 파일 전체의 추론 방식은 과거 EAT+MusicDET 버전입니다. |
| `build_dacon_submission_zip.py` | 제출 ZIP 제작 및 무결성 검사 |
| `download_fmc.py`, `download_musiccaps_real.py` | 음악 데이터 다운로드 |
| `run_with_timm_blocked.py`, `timm_block/` | timm 의존성 차단 검사 |
| `eat_architecture_no_timm/` | timm 의존성을 제거한 EAT 코드 |
| `make_submission_zip.py` | 이전 ZIP 제작 도구 |

최신 두 분기 추론 코드는 `aggressive_audio/package_demucs.py`가 제출 폴더에 구성합니다.
