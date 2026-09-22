# Semantic similarity pooling ablation

Baseline snapshot: fa00eaa. Changes apply only in DenseSSL_attention.

The historical score is mean_q(max_p(dot(v_ip, a_jq))). With visual_attention,
normalize tokens as before, compute query_i = mean_p(v_ip), then
alpha_ijq = softmax_q(query_i^T W a_jq), and sum_q(alpha_ijq * max_p(dot(v_ip,a_jq))).
All candidate pairs use the same rule. This pools similarity scores, not audio
embeddings. Spatial pooling and the inference network are unchanged.

W is a zero-initialized 384x384 parameter (147456 parameters), so initial scores
match average pooling. It uses Adam with lr_attention=0.000025 and existing
weight decay/schedule. Contrastive temperatures retain their historical fixed
behavior. Attention is saved in training_latest.pth and criterion_best.pth.

Commands (run sequentially, not simultaneously):

```bash
cd /home/huskypaul/DenseSSL_attention
./train.sh densessl_unseen1_dinov2reg_semsimpool_avg_s42 dinov2_vitb14_reg avg 42
./train.sh densessl_unseen1_dinov2reg_semsimpool_vcap_s42 dinov2_vitb14_reg visual_attention 42
```

Use separate experiment names. Default pooling is avg; seed defaults to 42.
Seed Python, NumPy and PyTorch before data/model creation. cuDNN benchmark is
disabled and deterministic enabled; this is not a guarantee of bitwise equality
across all platforms. Both new runs should be compared; historical scores had
no recorded matching seed. Resuming rejects missing or mismatched ablation
configuration. Existing historical states remain usable in original DenseSSL.

TensorBoard logs are checkpoints/<experiment>/mono2binaural/tensorboard.
Inference uses ./test.sh <experiment> dinov2_vitb14_reg; evaluation uses
./evaluate.sh <experiment>. No attention module runs at inference.

Switch back to avg with a new experiment name to run the baseline. The original
/home/huskypaul/DenseSSL repository and checkpoints have not been changed.

Verification: python test_pooling_ablation.py checks baseline equivalence,
uniform initialization, spatial invariance, all-pair candidate symmetry,
gradient/optimizer registration, and checkpoint restore/configuration guard.
