# DenseSSL

## 🛠️ Environment Setup (환경 설정 안내)

이 프로젝트는 다양한 GPU 환경(예: RTX 30xx, 40xx, 구형 GPU 등)에서의 원활한 협업 및 재현(Reproduction)을 위해 **PyTorch를 각자의 하드웨어에 맞게 수동으로 먼저 설치**하는 방식을 채택하고 있습니다.

아래의 순서대로 환경을 세팅해 주세요.

### Step 1. 가상환경 생성 (권장)
다른 프로젝트와의 패키지 충돌을 막기 위해 가상환경을 만들어주세요.
```bash
conda create -n densessl python=3.10 -y
conda activate densessl
```

### Step 2. PyTorch 설치 (매우 중요 ⭐)
본인 PC의 GPU 및 CUDA 버전에 호환되는 PyTorch 버전을 설치합니다.
(정확한 설치 명령어는 [PyTorch 공식 홈페이지](https://pytorch.org/get-started/locally/)에서 직접 확인하는 것을 권장합니다.)

**[참고용 설치 명령어 예시]**
- **NVIDIA GPU (CUDA 11.8 지원)**: 폭넓은 호환성을 원할 때
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- **NVIDIA GPU (CUDA 12.1 지원)**: 최신 그래픽카드(RTX 40xx 등) 사용 시
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **Mac (Apple Silicon - M1/M2/M3)**:
  ```bash
  pip install torch torchvision torchaudio
  ```

### Step 3. 나머지 필수 패키지 설치
PyTorch가 오류 없이 설치된 것을 확인한 후, 프로젝트 실행에 필요한 나머지 라이브러리들을 설치합니다.
```bash
pip install -r requirements.txt
```

---
## 🚀 How to Run (실행 방법)
환경 세팅이 완료되었다면 아래 명령어로 코드를 실행할 수 있습니다.

Project Root
```bash
cd .../DenseSSL
```
Make sure that you set your dataset directory correctly on @base_options.py

Train
```bash
./train.sh CHECKPOINT_NAME
```

Infernce
```bash
./test.sh CHECKPOINT_NAME
```

Evaluate
```bash
./evaluate.sh CHECKPOINT_NAME
```