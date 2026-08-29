# Scripts Directory Documentation

The `scripts/` directory contains all execution entrypoints, CLI scripts, multi-node gRPC launching commands, and experiment runners for centralized baseline training and federated learning workflows.

---

## Python Execution Entrypoints

### `run_centralized.py`
Standalone centralized training pipeline. Trains a single global model using all aggregated client data or a specified dataset split.

**Usage:**
```bash
python scripts/run_centralized.py --config configs/radimagenet_resnet50_centralized.yaml
```

**Key Responsibilities:**
- Loads experiment configuration via Hydra / YAML (`load_config`).
- Builds model backbone and head via `src/models/build.py`.
- Instantiates PyTorch dataloaders from `src/datasets/build.py`.
- Executes centralized training loop with mixed precision, metrics recording, and TensorBoard logging.
- Saves checkpoint artifacts under `runs/<experiment_name>/`.

---

### `run_federated.py`
Simulated federated learning entrypoint utilizing Ray or Flower in-memory simulation.

**Usage:**
```bash
python scripts/run_federated.py --config configs/fedavg_cbis_ddsm.yaml
```

**Key Responsibilities:**
- Initializes local FL simulation environment.
- Spawns virtual client nodes partitioned according to non-IID or IID data distribution schemes.
- Orchestrates federated aggregation rounds (`FedAvg`, `FedProx`, `SCAFFOLD`, etc.).
- Records per-node metrics and global model evaluation after each communication round.

---

### `run_server.py`
Launches a standalone Flower gRPC aggregation server for multi-machine federated training.

**Usage:**
```bash
python scripts/run_server.py --config configs/exp12/server.yaml
```

**Key Responsibilities:**
- Binds to host gRPC port (e.g. `0.0.0.0:8080`).
- Configures aggregation strategy (FedAvg/FedProx) and server-side evaluation holdouts.
- Coordinates client connection handshakes, round starting signals, and global checkpoint snapshots.

---

### `run_client.py`
Launches an individual Flower gRPC client process on a worker node.

**Usage:**
```bash
python scripts/run_client.py \
    --config configs/exp12/client.yaml \
    --server 192.168.1.10:8080 \
    --client-id 1 \
    --manifest manifests/node1_manifest.csv
```

**Key Responsibilities:**
- Connects to the central gRPC server node.
- Constructs local PyTorch dataset and model instance based on node manifest.
- Executes local epoch training upon receiving global model weights.
- Returns updated model parameters, fit metrics, and validation statistics back to the server.

---

### `run_evaluation.py`
Post-hoc model checkpoint evaluation script.

**Usage:**
```bash
python scripts/run_evaluation.py \
    --config configs/radimagenet_resnet50_fedavg.yaml \
    --checkpoint runs/my_experiment/global_model.pt
```

**Key Responsibilities:**
- Loads saved model weight checkpoints (`.pt`).
- Runs inference on designated validation/test sets without updating gradients.
- Outputs clinical evaluation metrics (ROC-AUC, F1-score, Sensitivity, Specificity, Confusion Matrix).

---

## Shell Automation Scripts

- `docker-deploy-federated.sh`: Automated Docker container deployment for FL nodes.
- `run-exp*.sh`: Batch launchers for experimental suites (Exp 01 to Exp 65).
- `eval-exp*.sh`: Automated post-experiment evaluation grid runners.
