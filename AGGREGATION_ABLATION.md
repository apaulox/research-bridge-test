# DINOv3 head and aggregation ablation

This checkout is isolated at /home/huskypaul/DenseSSL_aggregation, on branch
codex/head-aggregation-ablation, copied from attention commit b2d3de6.
Original DenseSSL and DenseSSL_attention code and checkpoints are unchanged.
Data and pretrained DINOv3 files remain shared read-only inputs.

## Design

| Variant | Contrastive projection | Aggregation |
| --- | --- | --- |
| single_poolprod | shared 512 per modality | pool_then_product |
| multi_poolprod | separate semantic/spatial 384 | pool_then_product |
| single_prodpool (partner) | shared 512 per modality | product_then_pool |
| multi_prodpool (ours control) | separate semantic/spatial 384 | product_then_pool |

Both semantic and spatial losses remain active with their original weights,
temperatures and positive/negative sampling. No learned attention is used.
Single visual head projects the full shared 768-channel feature to 512; its
output feeds both losses. The mono and stereo audio encoders remain separate,
but share one 512-channel projection. Their features are concatenated along
the batch dimension for a single projection call, sharing BatchNorm statistics.
The decoder still receives the original 384 spatial channels in every variant.
Single/multi therefore refers to contrastive projections, not audio encoders.
The designs differ in dimensions and parameter counts (512 versus two 384 heads).

Tokens are L2-normalized along channels before aggregation in all variants.
Pool-then-product: channelwise max over visual HW, mean over audio FT, then dot.
Product-then-pool: all visual/audio token dots, max over visual HW, mean over FT.
There is no post-pooling normalization or extra score scaling. Both semantic
and spatial branches use the selected aggregation order.

## Seed and historical baseline

The historical DINOv3 run had no explicit common Python/NumPy/PyTorch seed.
Its saved CPU Torch generator reports initial_seed=16655445212791328023.
The new experiments use this seed for Python, Torch and CUDA, and seed modulo
2**32 for NumPy. seed_text preserves the full integer in W&B JSON/UI.
For new runs, RNGs are reset after model construction so different head sizes
do not shift the initial data-sampling RNG stream. Resume restores saved states.
This is a new consistent protocol, not an exact replay of the old training.
The historical STFT 0.829156 / ENV 0.130505 are reference results; rerun the
multi_prodpool control with the new protocol for the actual comparison table.

## First experiment only

```bash
cd /home/huskypaul/DenseSSL_aggregation
./train_aggregation.sh densessl_unseen1_dinov3_single_poolprod_sd3 single pool_then_product
```

Expected: dinov3_vitb16, head_layout single, aggregation_order pool_then_product,
semantic_pool avg, seed 16655445212791328023. Here avg means attention OFF;
the aggregation_order determines whether features or similarity scores pool.
Training uses 1000 epochs, batchSize 16 plus 5 spatial samples, automatic
epoch-boundary resume, isolated checkpoints/logs, and W&B upload on completion.
W&B publication failure prints a retry command and does not delete checkpoints.

After training, inference requires the single layout explicitly:

```bash
./test.sh densessl_unseen1_dinov3_single_poolprod_sd3 dinov3_vitb16 single
./evaluate.sh densessl_unseen1_dinov3_single_poolprod_sd3
```

The subsequent variant is multi pool_then_product; the control is multi
product_then_pool. Use unique experiment names. Configuration mismatches are
rejected on resume. Neither is started automatically.

## Controlled scope and verification

Keep historical ISTFT behavior, mask range, validation sampling, loss weights,
LR schedule and decoder unchanged. In particular the known 256-bin ISTFT issue
has not been fixed in this experiment, so it does not confound aggregation.
The portable evaluate_bundle.py still supports DINOv2 only; DINOv3 evaluation
uses this checkout, local DINOv3 source and test.sh. Do not assume a DINOv3
artifact is self-contained for another machine.

Tests: test_aggregation.py checks formulas and a counterexample, all-pair
symmetry, both losses updating shared heads, tensor shapes, strict state reload,
inference independence from stereo input, and exact legacy multi architecture
and output equivalence. test_pooling_ablation.py checks the historical pooling,
optimizer and resume guard. smoke_aggregation.py checks one real pretrained
DINOv3 CUDA forward/backward without saving a training checkpoint.

Only train.py and demo_batch.py are wired to the new head layout. Older demo,
retrieval and visualization entry points retain their default multi layout.
