# Documentación de Scripts y Entrypoints (`scripts/`)

Esta sección documenta los scripts de ejecución de experimentos centralizados, simulación federada con Flower/Ray y despliegue distribuido cliente-servidor gRPC.

---

## Modos de Ejecución Centralizados y CLI

Los comandos principales instalados mediante el paquete CLI son:

1. **`fedmammobench-centralized --config configs/exp01.yaml`**:
   Ejecuta una corrida centralizada completa a partir del YAML de experimento usando `src/cli.py`.

2. **`fedmammobench-federated --config configs/fedavg_cbis_ddsm.yaml`**:
   Inicia una simulación de aprendizaje federado local utilizando Flower (Ray backend).

3. **`fedmammobench-evaluate --config configs/exp01.yaml --checkpoint runs/exp01/weights/best.pt`**:
   Evalúa un checkpoint guardado en el conjunto de test.

---

## Flujo de Scripts Distribuidos (gRPC Multi-Nodo)

Para corridas distribuidas entre nodos físicos o máquinas virtuales independientes, se utilizan los scripts de cliente y servidor:

### 1. Servidor de Agregación (`run_server.py`)

Inicia el servidor gRPC Flower que espera las conexiones de los clientes y coordina las rondas de agregación.

```bash
python scripts/run_server.py --config configs/exp12/server.yaml
```

### 2. Clientes Distribuidos (`run_client.py`)

Cada nodo participante ejecuta una instancia del cliente Flower conectándose a la IP del servidor.

```bash
python scripts/run_client.py \
    --config configs/exp12/client.yaml \
    --server <SERVER_IP>:8080 \
    --client-id 1 \
    --manifest manifests/node1_manifest.csv \
    --data-dir /data/mammography
```

---

## Ejemplo de Pipeline Integrado en Python

```python
# Ejemplo de invocación programática de la CLI
from src.config import load_config
from src.cli import run

config = load_config("configs/exp08/exp08.yaml")
run(config)
```
