# DenseSSL_new single-head 구조 기록

기록일: 2026-09-30
대상: seed 42 결과를 확인한 뒤 수정한, 다음 학습용 코드 구조

## 영상 경로

1. 입력 영상 `[B, 3, 224, 448]`을 frozen DINOv3 ViT-B/16에 통과시킨다.
2. 보간 없이 원래 patch map `[B, 768, 14, 28]`을 사용한다.
3. 학습 가능한 공통 visual 층을 통과한다. 출력 크기는 `[B, 768, 14, 28]`이다.
4. 공통층 출력에서 두 경로로 나뉜다.
   - Contrastive: `768 → 384 → 384` projection, 위치별 채널 L2 정규화, HW max pooling으로 `[B, 384]` embedding 생성.
   - Decoder: adaptive average pooling으로 `[B, 768, 7, 14]`, 1×1 conv `768 → 8`로 `[B, 8, 7, 14]`, flatten으로 `[B, 784]` 생성. `[16, 32]`로의 interpolation은 없다.

## Mono 오디오 경로

1. 입력은 L+R mono의 복소 스펙트로그램 `[B, 2, 257, 64]`이다. 두 채널은 실수부와 허수부다.
2. Mono U-Net 인코더의 bottleneck 출력은 `[B, 512, 8, 2]`이다.
3. 이 출력에서 두 경로로 나뉜다.
   - Contrastive: `512 → 512 → 384` projection, 위치별 채널 L2 정규화, FT average pooling으로 `[B, 384]` embedding 생성.
   - Decoder: 512채널 bottleneck feature와 skip connection을 사용한다.

## Loss와 decoder

- `[B, 384]` 영상 embedding과 `[B, 384]` mono embedding을 pooling **후** 내적하여 `[B, B]` similarity를 계산한다. 영상–mono 단일 양방향 InfoNCE를 사용한다.
- Decoder는 visual 784채널을 mono bottleneck의 `8×2` 크기로 펼쳐 mono 512채널과 결합한다. 첫 up-convolution 입력은 1296채널이다.
- Decoder가 예측한 complex mask를 mono 스펙트로그램에 적용해 L−R 스펙트로그램을 만들고, 정답 L−R과 MSE를 계산한다.
- 학습 목표는 MSE + contrastive loss다. 공통 visual 층과 mono 인코더는 두 loss의 gradient를 모두 받는다.
- Stereo spatial 인코더, `aspa` 경로, 두 번째 contrastive head 및 별도 spatial contrastive loss는 없다.

## 비교 해석

Ours는 영상 semantic/spatial 두 head와 mono/stereo 두 오디오 경로를 사용한다. 여기서는 영상–mono 한 쌍만 contrastive 학습한다. 이 실험은 pooling 순서와 decoder visual 입력도 Ours와 다르므로, 성능 차이를 head 수 하나만의 효과로 해석하지 않는다.

이 문서는 현재 구조 설명만 기록한다. 학습 시작, checkpoint 변경 또는 추가 코드 수정은 포함하지 않는다.
