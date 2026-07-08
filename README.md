# Unseen Link Prediction via SGMF

SGMF (Side-information enhanced Generalized Matrix Factorization) for Internet-scale unseen AS link prediction. The method constructs an AS hop-count matrix from BGP paths, incorporates multi-source AS side information, and applies GMF-based matrix completion to infer missing AS links.

## Quick Start

```bash
# Install dependencies
pip install torch pandas numpy scikit-learn tensorboardX

# Run with sample data (1000 training pairs)
python multitrainmf.py

# Run with full data
python multitrainmf.py --train_file train_hop_matrix_full.csv --valid_file validate_hop_matrix_full.csv

# CPU mode
python multitrainmf.py --no_cuda
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `../sgmf_myx/data/AShopmatrix` | Path to data directory |
| `--train_file` | `train_hop_matrix_sample_1000.csv` | Training CSV file name |
| `--valid_file` | `validate_hop_matrix_sample_1000.csv` | Validation CSV file name |
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `512` | Training batch size |
| `--lr` | `0.001` | Adam learning rate |
| `--latent_dim` | `100` | Latent embedding dimension |
| `--device_id` | `0` | CUDA device ID |
| `--no_cuda` | `False` | Disable CUDA (use CPU) |
| `--num_users` | `77438` | Number of AS nodes |
| `--ixp_pad` | `1125` | appearIXP padding length |
| `--fac_pad` | `4380` | appearFac padding length |

### Feature Cardinalities

| Argument | Default |
|----------|---------|
| `--num_info_type` | `11` |
| `--num_as_tier` | `5` |
| `--num_info_traffic` | `19` |
| `--num_info_ratio` | `6` |
| `--num_info_scope` | `10` |
| `--num_policy_general` | `5` |
| `--num_policy_locations` | `6` |
| `--num_policy_ratio` | `3` |
| `--num_policy_contracts` | `4` |

## Data Format

CSV files with 25 comma-separated columns (no header):

```
userId,itemId,rating,
ASnode1_info_type,ASnode1_AS_tier,ASnode1_info_traffic,ASnode1_info_ratio,
ASnode1_info_scope,ASnode1_policy_general,ASnode1_policy_locations,
ASnode1_policy_ratio,ASnode1_policy_contracts,ASnode1_appearIXP,ASnode1_appearFac,
ASnode2_info_type,ASnode2_AS_tier,ASnode2_info_traffic,ASnode2_info_ratio,
ASnode2_info_scope,ASnode2_policy_general,ASnode2_policy_locations,
ASnode2_policy_ratio,ASnode2_policy_contracts,ASnode2_appearIXP,ASnode2_appearFac
```

- `appearIXP` / `appearFac` columns contain Python list literals (e.g., `"[1, 5, 23]"`).
- `rating` is the AS hop count (regression target).

## Project Structure

```
Unseen_Link_Prediction/
├── data.py              # Data loading: lazy DataLoader, padding, SampleGenerator
├── engine.py            # Training/evaluation engine (batch-wise validation)
├── gmf.py               # GMF model with side-information embeddings & masked mean
├── mlp.py               # MLP model (baseline)
├── neumf.py             # NeuMF model (GMF + MLP fusion)
├── metrics.py           # Evaluation metrics (Hit Ratio, NDCG)
├── utils.py             # Utilities: checkpoint, optimizer, CUDA setup
├── multitrainmf.py      # Main training entry point
└── README.md
```

## Key Optimizations

Compared to the original implementation, this version includes:

1. **Lazy DataLoader** — Side information is processed on-the-fly in `__getitem__` rather than pre-expanded per sample, reducing memory from O(|E|·F) to O(F).
2. **Batch-wise validation** — Validation data is evaluated in batches instead of loading the entire set onto GPU at once, preventing OOM.
3. **Masked mean for padded sequences** — `appearIXP` and `appearFac` embeddings use `padding_idx=0` with a masked average, ignoring padding positions.
4. **Field bug fixes** — `AS_tier` was incorrectly assigned from `info_type`; now reads from the correct column.
5. **Configurable via argparse** — All hyperparameters and paths are exposed as CLI arguments instead of being hardcoded.
