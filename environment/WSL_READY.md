# DACON WSL CUDA 환경 사용법

2026-09-07 설치 및 실행 검증 완료.

| 항목 | 실제 설치/확인 결과 |
|---|---|
| 리눅스 | WSL2 · Ubuntu-24.04 (Ubuntu 24.04.4 LTS) |
| 리눅스 사용자 | huskypaul |
| GPU | NVIDIA GeForce RTX 5080 · 16GB · sm_120 |
| Windows GPU 드라이버 | 591.86 |
| Conda 환경 | dacon-cu128 |
| Python | 3.10.21 |
| PyTorch / torchaudio | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| PyTorch CUDA 런타임 | 12.8 |
| librosa / transformers | 0.10.2.post1 / 4.46.3 |
| JupyterLab | 4.6.3 |

환경 위치는 `/home/huskypaul/miniforge3/envs/dacon-cu128`이다.
기존 DCASE Conda 환경과 별도로 만들었다.
PyTorch의 Blackwell 지원 CUDA 12.8 빌드를 사용했고,
WSL에서는 기존 Windows GPU 드라이버를 사용한다.
[PyTorch 공식 안내](https://pytorch.org/blog/pytorch-2-7/),
[NVIDIA WSL 안내](https://docs.nvidia.com/cuda/archive/13.0.0/wsl-user-guide/index.html)

## 터미널에서 시작하기

Windows PowerShell에서:

```powershell
wsl -d Ubuntu-24.04
```

열린 Ubuntu에서:

```bash
source ~/dacon/activate.sh
```

이 한 줄이 `dacon-cu128` 환경을 켜고 베이스라인 폴더로 이동한다.
이후 확인 또는 실행:

```bash
pwd
python --version
python script.py
```

베이스라인 결과는 `/home/huskypaul/dacon/baseline/output/submission.csv`에 저장된다.
다시 실행하면 기존 CSV를 갱신한다. 중요한 실험 결과는 실행 전에 다른 이름으로 보관한다.

## Jupyter로 시작하기

Windows PowerShell에서 다음 한 줄을 실행한다:

```powershell
wsl -d Ubuntu-24.04 -- bash /home/huskypaul/dacon/start_jupyter.sh
```

터미널에 나타나는 `http://127.0.0.1:8888/lab?token=...` 형식의 주소를 브라우저에서 연다.
포트가 사용 중이면 표시된 실제 주소/포트를 사용한다. 해당 터미널은 사용 중 열어둔다.
종료할 때는 그 터미널에서 Ctrl+C를 누른다.

1. 왼쪽 파일 목록에서 `baseline` 폴더를 연다.
2. `00_start_here.ipynb`를 연다.
3. 커널은 **Python (DACON CUDA 12.8)** 을 선택한다.
4. 셀을 위에서 아래로 **Shift+Enter**로 실행한다.

Jupyter 서버는 상시 실행해두지 않았다. 위 명령으로 필요할 때 실행한다.
로그인 토큰을 사용하며 로컬 주소에만 바인딩한다.

## 폴더 구성

```text
/home/huskypaul/dacon/
├── activate.sh
├── start_jupyter.sh
├── baseline/
│   ├── script.py                  # ZIP의 원본 실행 코드
│   ├── model/                     # 제공된 PANNs, HTDemucs, DF-Arena 가중치
│   ├── data/                      # 제공된 3개 오디오와 제출 양식
│   ├── output/submission.csv      # 검증에서 생성한 결과
│   ├── 00_start_here.ipynb        # 간단한 시작 노트북
│   └── 01_original_baseline.ipynb # 제공된 노트북의 실행용 복사본
├── notebooks/                    # 제공된 원본 노트북 보관
├── downloads/baseline_submit.zip # 외부 ZIP에서 추출한 원본 제출 ZIP
├── musicdet/                     # 이전에 준비한 MusicDET 코드
├── setup/                        # 설치 명세, 전체 패키지 버전 고정 목록
└── logs/                         # 환경·CUDA·베이스라인 검증 기록
```

Windows 탐색기 주소창에서 `\\wsl.localhost\Ubuntu-24.04\home\huskypaul\dacon`으로 접근할 수 있다.
Windows Downloads의 `open.zip`과 원본 노트북은 수정하지 않았다.

## 검증 범위

- CUDA GPU 행렬 연산 및 torchaudio GPU 리샘플링 성공.
- MusicDET SpecNF의 임의 가중치 + 합성 입력으로 순전파, 역전파, optimizer step, eval 성공.
- 베이스라인 PANNs → HTDemucs → DF-Arena 오프라인 추론으로 3개 파일의 CSV 생성 성공.
- CSV ID 순서, 5개 확률의 유한성/0~1 범위, MAX 결합식 검사 통과.
- 제공된 3개 모델 가중치의 SHA-256 값이 ZIP의 명세와 일치.
- pip check에서 의존성 충돌 없음.

샘플 3개의 결과는 설치 동작 확인용이다. 대회 EER·정확도나 1,200개 처리 시간을 검증한 결과가 아니다.
MusicDET의 **FMC 학습 완료 가중치는 아직 없다**. MusicDET 학습을 실행하지 않았으며,
베이스라인의 음악 분기도 아직 DF-Arena 그대로다.

Windows 작업 폴더의 `wsl_environment/`에 환경 기록, 설치 버전 목록,
`baseline_smoke_submission.csv`를 복사해 두었다.
