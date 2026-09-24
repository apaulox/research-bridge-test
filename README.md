# DenseSSL Aggregation Ablation

DINOv3를 고정하고 contrastive head 구성과 pooling 순서를 비교한다.
현재 실험은 **single-head + pool → product**다.

## 기존 방식과의 차이

| 항목 | 기존 Ours | Single-head + Pool → Product |
| --- | --- | --- |
| 영상 projection | Semantic·spatial 각각 384차원 | 768 → 512차원 projection 하나를 공유 |
| 오디오 projection | Semantic·spatial 각각 384차원 | 512 → 512차원 projection 하나를 공유 |
| 유사도 계산 | Token 간 내적 → 영상 max → 오디오 평균 | 영상 max·오디오 평균 → 내적 |

영상은 공통 projection의 768차원 출력을 single projection에 넣고,
같은 특징을 semantic·spatial loss에 사용한다.

오디오는 기존 mono encoder와 spatial encoder를 유지한다.
두 encoder의 출력에 같은 projection을 적용하며, BatchNorm 통계도 공유한다.
Single-head는 encoder를 하나로 합친다는 뜻이 아니다.

## Aggregation

기존 방식은 오디오 위치마다 가장 유사한 영상 패치를 찾은 뒤 점수를 평균낸다.

```text
[HW, D] × [FT, D] → [HW, FT] 유사도 → HW max → FT 평균 → 점수
```

이번 방식은 영상과 오디오를 먼저 하나의 벡터로 요약한 뒤 내적한다.
영상 max는 채널별 최댓값을 취한다.

```text
영상 [HW, 512] → HW max  → [512] ─┐
                                 ├─ 내적 → 점수
오디오 [FT, 512] → FT 평균 → [512] ─┘
```

두 방식 모두 token별 L2 정규화를 먼저 적용한다. Pooling 후 추가 정규화는 없다.
선택한 aggregation을 semantic·spatial loss 양쪽에 사용한다.

## 유지하는 조건

- Frozen DINOv3 ViT-B/16과 기존 audio encoder·binaural decoder
- Decoder에 들어가는 기존 384차원 visual feature
- `MSE + semantic loss + spatial loss`와 각 loss의 가중치
- 데이터 split, mask 범위, FFT 복원 방식, 학습률 일정

Attention pooling은 사용하지 않는다. 기존 실험은 `DenseSSL`과
`DenseSSL_attention`에 보존하고, 새 checkpoint와 로그는 이 폴더에 저장한다.

## 실행

```bash
cd /home/huskypaul/DenseSSL_aggregation
./train_aggregation.sh densessl_unseen1_dinov3_single_poolprod_sd3 single pool_then_product
```

`head_layout: single`, `aggregation_order: pool_then_product`를 확인한다.
`semantic_pool: avg`는 attention을 사용하지 않는 설정이다.

```bash
./test.sh densessl_unseen1_dinov3_single_poolprod_sd3 dinov3_vitb16 single
./evaluate.sh densessl_unseen1_dinov3_single_poolprod_sd3
```

## 비교 기준

Single/multi는 같은 pooling 방식끼리, pooling 순서는 같은 head 구성끼리 비교한다.
Single 512차원과 multi 384×2는 파라미터 수까지 동일한 조건은 아니다.

새 실험의 seed는 기존 DINOv3 체크포인트에서 확인한 PyTorch seed
`16655445212791328023`을 사용한다. Python·CUDA에도 같은 값을 적용하고,
NumPy에는 `seed % 2**32`를 사용한다. 기존 실행의 Python·NumPy 초기 seed는
확인되지 않아 완전한 재현은 아니며, Ours 기준군도 같은 새 설정으로 다시 학습한다.

세부 설정과 검증 내용은 [AGGREGATION_ABLATION.md](AGGREGATION_ABLATION.md)에 정리했다.

## Single 512 / Pool → Product 결과

FAIR-Play unseen1, seed `16655445212791328023`, 1000 epochs.
Validation loss가 가장 낮은 checkpoint는 epoch 291에서 저장됐다.
동일한 104개 test clip의 평가 결과는 다음과 같다.

| 지표 ↓ | 평균 | 표준편차 | 표준오차 |
| --- | ---: | ---: | ---: |
| STFT L2 | 0.822451 | 0.681647 | 0.066841 |
| Envelope | 0.129860 | 0.056370 | 0.005528 |

원본 평가 출력은 [evaluation_single_poolprod_sd3.txt](evaluation_single_poolprod_sd3.txt)에 있다.
Best 모델 파일과 학습 기록은 [W&B run](https://wandb.ai/stringwoo22-hanyang-university/DenseSSL/runs/pool-densessl_unseen1_dinov3_single_poolprod_sd3)에 연결된
`densessl_unseen1_dinov3_single_poolprod_sd3-best:v1` artifact에 보관했다.
대용량 checkpoint와 FAIR-Play 데이터는 Git에 포함하지 않는다.
