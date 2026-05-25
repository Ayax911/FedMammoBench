# Federated Deployment Guide — RadImageNet ResNet50

End-to-end operational reference for running a real gRPC-based federated
experiment with RadImageNet pretrained weights.

---

## 1 — Topology

```
┌─────────────────────────────────────────────┐
│                  SERVER                     │
│  scripts/run_server.py                      │
│  configs/radimagenet_resnet50_grpc_server.yaml │
│                                             │
│  • Loads RadImageNet-resnet50.pth           │
│  • Builds initial parameters                │
│  • FedAvg aggregation                       │
│  • Federated evaluation (default, no images)│
│  • Centralized evaluation (optional)        │
│  • Listens on 0.0.0.0:8080                  │
└──────────────────┬──────────────────────────┘
                   │ gRPC
         ┌─────────┴──────────┐
         ▼                    ▼
┌────────────────┐  ┌────────────────┐
│  CLIENT 0      │  │  CLIENT 1      │
│  --client-id 0 │  │  --client-id 1 │
│                │  │                │
│  Local data    │  │  Local data    │
│  Trains head   │  │  Trains head   │
│  (rounds 0-2)  │  │  (rounds 0-2)  │
└────────────────┘  └────────────────┘
```

**Freeze schedule** (controlled by server, applied per-client):
- Rounds 0–2: `freeze_backbone=True` → only the classification head trains.
- Round 3+: `unfreeze_at_epoch=3` → server sends `current_round=3` → full
  fine-tune. The server injects `current_round` via `on_fit_config_fn`; the
  client calls `apply_freeze_policy` after loading server weights each round.

**gRPC message size**: ResNet50 state_dict ≈ 100 MB per client (FP32).
Flower's default limit is 512 MB — sufficient for this configuration.

---

## 1.1 — How the Global Model is Validated

The server supports **two evaluation modes**, controlled by the server-side
`data.name` field:

| Mode | `data.name` | Where validation runs | Server needs images? | Output CSV |
|---|---|---|---|---|
| **Pure federated** (default, recommended) | `none` | Every client evaluates the aggregated model on its own local val split; server weighted-averages the metrics | **No** | `server_federated_metrics.csv` |
| **Federated + centralized holdout** (opt-in) | `mammo_bench` / `cbis_ddsm` / `vindr_mammo` / `synthetic` | Same federated path *plus* a centralized eval on a holdout the server operator owns (e.g. a public benchmark) | **Yes** | both `server_metrics.csv` and `server_federated_metrics.csv` |

The federated path is always active when `min_evaluate_clients ≥ 1` and the
clients have `val_fraction > 0`. The centralized path activates only when the
server's `build_dataset` produces a non-empty `test` split.

### Why federated-by-default

Hosting images on the server breaks the FL principle whenever the holdout is
sourced from the clients themselves (data leakage) and it always adds
storage/compliance overhead. Pure federated evaluation keeps every image
inside its hospital of origin while still producing per-round AUC/F1/sens/spec.

### Caveats of weighted-averaged metrics

`AUC_weighted = Σ(AUC_i · n_i) / Σ n_i` is **not** the same as AUC computed
on the pooled predictions. For ranking metrics this is an approximation; for
linear metrics (loss, accuracy, sensitivity, specificity at a fixed threshold)
the weighted mean is exact under the usual i.i.d. assumption inside each
client. If you need the true pooled AUC you must either accept centralized
holdout (and the data-governance cost) or wire a secure aggregation of
predictions (out of scope here).

### Picking a server config

- `configs/radimagenet_resnet50_grpc_server_no_holdout.yaml` — pure federated.
- `configs/fedavg_mammobench_server_no_holdout.yaml` — pure federated, Mammo-Bench.
- `configs/radimagenet_resnet50_grpc_server.yaml` — legacy, hosts a test set.
- `configs/fedavg_mammobench_server.yaml` — legacy, hosts a test set.

---

## 2 — Preparation Checklist

### 2.1 Dependencies

Install the project in a Python 3.11 virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Verify core imports:

```bash
python -c "import torch, flwr, fedmammo; print('OK')"
```

### 2.2 RadImageNet Checkpoint

Download `RadImageNet-resnet50.pth` from the official repository:

> **BMEII-AI/RadImageNet** on GitHub

Place the file in a directory accessible by all machines and set:

```bash
# Option A — environment variable (recommended for shared clusters)
export FEDMAMMO_RADIMAGENET_DIR=/absolute/path/to/ckpt/dir

# Option B — inline in the YAML (per-machine config)
# model:
#   checkpoint_path: /absolute/path/to/RadImageNet-resnet50.pth
```

Verify the checkpoint loads correctly before starting the experiment:

```python
import torch
state = torch.load("/path/to/RadImageNet-resnet50.pth", map_location="cpu")
print(type(state), list(state.keys())[:5] if isinstance(state, dict) else "plain dict")
```

Expected: either a plain state_dict with keys like `conv1.weight`, or a nested
dict with a `state_dict` key. Both formats are handled by `RadImageNetLoader`.

### 2.3 Network — Localhost Smoke Test

For a localhost smoke test (all processes on one machine), no firewall changes
are needed. The server listens on `0.0.0.0:8080`; clients connect to
`127.0.0.1:8080`.

### 2.4 Network — Multi-Machine

- Open TCP port 8080 on the server machine.
- Clients need the server's LAN IP (e.g. `192.168.1.10`).
- No inbound ports needed on client machines.

### 2.5 Data

For a **smoke test** with synthetic data, no real images are needed — both
configs default to `data.name: synthetic`.

For **real data**, on each client node:

1. Pre-partition the dataset using `scripts/partition_mammobench.py` (or
   manually split the manifest CSV).
2. Each node must have its own `nodeN_manifest.csv` pointing to its local
   image files.
3. Pass paths via CLI: `--manifest /path/to/nodeN_manifest.csv --data-dir /path/to/images`.

---

## 3 — Exact Launch Commands

Start processes **in this order**: server first, then clients.

### Terminal 1 — Server

```bash
export FEDMAMMO_RADIMAGENET_DIR=/path/to/radimagenet
python scripts/run_server.py \
    --config configs/radimagenet_resnet50_grpc_server.yaml \
    --output-dir runs/radimagenet_exp1
```

The server blocks and prints:

```
INFO  run_server: Server config loaded. address=0.0.0.0:8080  rounds=5  strategy=fedavg  min_available_clients=2
INFO  run_server: Artifacts will be written to: runs/radimagenet_exp1
INFO  fedmammo.models.weight_loaders.radimagenet: [radimagenet/resnet50] Loaded checkpoint ...
INFO  fedmammo.federated.server: Starting gRPC server on 0.0.0.0:8080 for 5 rounds (strategy=fedavg, min_available_clients=2). Waiting for clients to connect ...
```

### Terminal 2 — Client 0

```bash
export FEDMAMMO_RADIMAGENET_DIR=/path/to/radimagenet
python scripts/run_client.py \
    --config configs/radimagenet_resnet50_grpc_client.yaml \
    --server 127.0.0.1:8080 \
    --client-id 0 \
    --output-dir runs/radimagenet_exp1
```

### Terminal 3 — Client 1

```bash
export FEDMAMMO_RADIMAGENET_DIR=/path/to/radimagenet
python scripts/run_client.py \
    --config configs/radimagenet_resnet50_grpc_client.yaml \
    --server 127.0.0.1:8080 \
    --client-id 1 \
    --output-dir runs/radimagenet_exp1
```

Multi-machine: replace `127.0.0.1` with the server's LAN IP.

Real data: add `--manifest /data/nodeN_manifest.csv --data-dir /data/images`.

---

## 4 — Expected Log Output

### Server after both clients connect

```
INFO  flwr: [ROUND 1] strategy.configure_fit: 2 clients selected
INFO  fedmammo.federated.client: Client 0: apply_freeze_policy current_round=1 → backbone FROZEN
INFO  fedmammo.federated.client: Client 1: apply_freeze_policy current_round=1 → backbone FROZEN
INFO  flwr: [ROUND 1] strategy.aggregate_fit: received 2 results
INFO  fedmammo.federated.server: [server] round 1 centralized: loss=0.71 auc=nan f1=nan ...
```

`auc=nan` and `f1=nan` are **normal for synthetic data** (not enough samples
to compute meaningful metrics). With real data these populate after round 1.

### Round 3 — progressive unfreeze

```
INFO  fedmammo.federated.client: apply_freeze_policy current_round=3 → backbone UNFROZEN (unfreeze_at_epoch=3)
```

Both clients log this at the start of round 3. From this point the full
backbone trains and gradient flow is end-to-end.

### End of experiment

```
INFO  fedmammo.federated.server: gRPC server stopped. Artifacts in runs/radimagenet_exp1
```

Artifacts written:
```
runs/radimagenet_exp1/
├── server.log
├── config.snapshot.yaml
├── server_metrics.csv              # per-round centralized metrics (only if data.name != 'none')
├── server_federated_metrics.csv    # per-round federated metrics (always)
└── tb/                             # TensorBoard logs (server/centralized + server/federated)
```

Per-client:
```
runs/radimagenet_exp1/client_0/
├── client.log
└── config.snapshot.yaml
```

---

## 5 — Validation Checklist (Before Real Training)

Run in this order:

```bash
# 1. Unit tests (no torch needed for the 6 pure-Python tests)
pytest tests/test_radimagenet.py -v

# 2. Smoke — existing configs unbroken (BC validation)
python -c "
from fedmammo.configs import load_config
for f in ['configs/fedavg_synthetic.yaml', 'configs/base.yaml']:
    load_config(f); print(f'OK: {f}')
"

# 3. Smoke — RadImageNet config parses correctly
python -c "
from fedmammo.configs import load_config
cfg = load_config('configs/radimagenet_resnet50_grpc_server.yaml')
print('weight_source:', cfg.model.weight_source)
print('freeze_backbone:', cfg.model.freeze_backbone)
print('unfreeze_at_epoch:', cfg.model.unfreeze_at_epoch)
print('normalize_preset:', cfg.training.augmentation.normalize_preset)
"

# 4. Smoke — checkpoint loads and model forward passes (requires torch + checkpoint)
python -c "
import os, torch
os.environ['FEDMAMMO_RADIMAGENET_DIR'] = '/path/to/radimagenet'
from fedmammo.configs import load_config
from fedmammo.models import build_model
cfg = load_config('configs/radimagenet_resnet50_grpc_server.yaml')
model = build_model(cfg.model).eval()
x = torch.zeros(1, 1, 224, 224)
y = model(x)
print('Forward OK. Output shape:', y.shape)   # expect: torch.Size([1, 2])
"

# 5. Smoke — freeze policy works
python -c "
from fedmammo.configs import load_config
from fedmammo.models import build_model
from fedmammo.models.weight_loaders import apply_freeze_policy
cfg = load_config('configs/radimagenet_resnet50_grpc_server.yaml')
cfg.model.weight_source = 'none'   # skip checkpoint for this test
import torch; model = build_model(cfg.model)
r = apply_freeze_policy(model, cfg.model, current_round=1)
print('Round 1 — trainable:', r['trainable_params'], '/', r['total_params'])
r = apply_freeze_policy(model, cfg.model, current_round=3)
print('Round 3 — trainable:', r['trainable_params'], '/', r['total_params'])
"
```

---

## 6 — Troubleshooting

### Checkpoint not found

```
FileNotFoundError: RadImageNet-resnet50.pth not found.
  Checked: (1) cfg.model.checkpoint_path=None
           (2) $FEDMAMMO_RADIMAGENET_DIR not set or file missing.
  ...
```

Fix: set `FEDMAMMO_RADIMAGENET_DIR` to the directory containing
`RadImageNet-resnet50.pth`, or set `model.checkpoint_path` in the YAML to the
absolute `.pth` path.

---

### gRPC connection refused on client

```
grpc._channel._InactiveRpcError: StatusCode.UNAVAILABLE
  details = "Connection refused"
```

Fix: start the server before the clients. Verify the server is listening:

```bash
# Linux/macOS
ss -tlnp | grep 8080
# Windows
netstat -ano | findstr 8080
```

If the server is on a different machine, replace `127.0.0.1` with the server's
actual IP address.

---

### Server hangs waiting for clients

```
INFO  Waiting for clients to connect ...
```

The server waits until `min_available_clients` (2) have registered. If only
one client starts, the experiment never begins. Start both clients.

---

### Shape mismatch on state_dict load

```
RuntimeError: Error(s) in loading state_dict for ResNet50Classifier:
  size mismatch for backbone.fc.1.weight: copying a param with shape ...
```

The server and client configs have different `num_classes` or `dropout`.
Ensure both YAMLs have identical `model.name`, `model.num_classes`, and
`model.dropout`.

---

### NaN training loss

Possible causes:
- **Too few synthetic samples**: `synthetic_num_samples: 32` with
  `batch_size: 16` gives only 2 batches. Increase to 256+ or use real data.
- **LR too high after unfreeze**: lower `optimizer.lr` to `1.0e-5` for the
  full fine-tune phase if loss spikes at round 3.
- **Single-class batch**: with only 32 samples balanced between 2 classes,
  random splits may produce all-one-class batches. Increase
  `synthetic_num_samples` or set `balance_classes: true`.

---

### Wrong freeze behavior

Verify the server is correctly injecting `current_round` each round:

```bash
grep "current_round" runs/radimagenet_exp1/client_0/client.log
```

Expected output (one line per round):
```
INFO  ... apply_freeze_policy current_round=1 → backbone FROZEN
INFO  ... apply_freeze_policy current_round=2 → backbone FROZEN
INFO  ... apply_freeze_policy current_round=3 → backbone UNFROZEN
```

If `current_round` is always 0, the server config is using an old
`on_fit_config_fn` that does not include the key. Ensure
`src/fedmammo/federated/server.py:_make_on_fit_config_fn` returns both
`"server_round"` and `"current_round"`.

---

### Large gRPC payload / message too large

Default Flower limit is 512 MB. ResNet50 ≈ 100 MB per client — well within
limits. If you switch to InceptionV3 (≈ 108 MB) or add multiple clients, the
aggregate payload stays under 512 MB for up to 4 clients.

If you hit the limit, pass `grpc_max_message_length` to
`fl.server.start_server` and `fl.client.start_numpy_client` in the run scripts.

---

### BN stats drift during frozen training

Known limitation (documented in TRANSFER_LEARNING_GUIDE.md §4). `apply_freeze_policy`
calls `_set_bn_eval` on frozen BatchNorm layers, but `model.train()` inside
`Trainer` resets them to training mode each epoch. The γ/β weights stay frozen
(correct); only running `mean`/`var` buffers may drift slightly.

For strict BN stat freezing, modify `Trainer.train_one_epoch` to call
`apply_freeze_policy` again after each `model.train()` call.

---

## 7 — Scaling Up

| Change | YAML field to update |
|--------|----------------------|
| More rounds | `federated.rounds` (server + client must match) |
| More clients | `federated.num_clients`, `min_*_clients`, `min_available_clients` |
| Real data | `data.name`, `data.manifest_path`, `data.image_root` |
| Longer freeze | `model.unfreeze_at_epoch` (increase; match server + client) |
| Different backbone | `model.name: densenet121` + update normalize_preset |
| Non-IID split | `partitioning.scheme: dirichlet`, `partitioning.alpha: 0.1` |

For DenseNet121 or InceptionV3 (RadImageNet variants), update
`model.name` and ensure `RadImageNet-{arch}.pth` exists in
`$FEDMAMMO_RADIMAGENET_DIR`.
