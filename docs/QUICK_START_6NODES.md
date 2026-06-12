# Quick Start: 6 Nodos + Servidor Central en 10 Minutos

## Resumen Ejecutivo

Tienes:
- **Este PC**: Servidor central (puerto 8080)
- **Otros PCs/Contenedores**: 6 nodos cliente (Node0–Node5)

Cada nodo necesita:
1. **Pesos**: `weights/RadImageNet-resnet50.pth` (copia para todos)
2. **CSV**: `manifests/node<ID>_manifest.csv` (específico por nodo)
3. **Imágenes**: `data/mammobench/images/` (específicas por nodo)

---

## PASO 1: Setup del Servidor Central (Este PC)

### 1.1 Generar particiones de datos

```bash
cd /path/to/Federal-Learning

# Esto divide mammo-bench.csv en 6 nodos + servidor
python scripts/partition_mammobench.py \
  --csv data/mammobench/mammo-bench.csv \
  --out manifests/ \
  --nodes 6
```

✓ Crea: `manifests/node0_manifest.csv` ... `manifests/node5_manifest.csv`

### 1.2 Descargar pesos RadImageNet

```bash
mkdir -p weights

# Opción A: Descargar de GitHub
wget https://github.com/BMEII-AI/RadImageNet/releases/download/v1.0/RadImageNet-resnet50.pth \
  -O weights/RadImageNet-resnet50.pth

# Opción B: Si ya tienes el .pth, copiar:
cp /ruta/a/RadImageNet-resnet50.pth weights/
```

### 1.3 Copiar imágenes

```bash
# Crear estructura
mkdir -p data/mammobench/images

# Copiar TODAS las imágenes de mammo-bench a data/mammobench/images/
# Mantener la estructura de subcarpetas: cmmd/, dmid/, ibia/, cdd-cesm/, kau-bcmd/, ddsm/
cp -r /ruta/a/mammo-bench/images/* data/mammobench/images/
```

### 1.4 Verificar setup del servidor

```bash
bash scripts/verify_setup.sh server
```

Expected output:
```
✓ Config servidor (6 nodos)
✓ Config cliente (todos)
✓ Pesos RadImageNet encontrados
✓ Directorio de imágenes
✓ Manifest Node0 ... Node5
✓ Variable $FEDMAMMOBENCH_RADIMAGENET_DIR = /path/to/Federal-Learning/weights
```

---

## PASO 2: Arrancar Servidor Central

```bash
cd /path/to/Federal-Learning
export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights

bash scripts/start_server.sh configs/exp01_fedavg_resnet50_6nodes_server.yaml
```

**Verás:**
```
════════════════════════════════════════════════════════════════════
  Servidor federated learning
  Config  : configs/exp01_fedavg_resnet50_6nodes_server.yaml
  Escucha : 0.0.0.0:8080
  IP LAN  : 192.168.14.184:8080  ← ¡COPIAR ESTA IP!
  Pesos   : /path/to/Federal-Learning/weights
════════════════════════════════════════════════════════════════════
Waiting for 6 clients to be available...
```

📌 **COPIAR LA IP LAN** (ej: `192.168.14.184`). La necesitas en cada nodo.

**NO CERRAR ESTA TERMINAL** — el servidor debe estar corriendo mientras entrenan los nodos.

---

## PASO 3: Setup de CADA NODO (Máquinas Remotas)

En **CADA máquina remota** que quieras usar como nodo, hacer esto:

### 3.1 Clonar/Actualizar repositorio

```bash
# Opción A: Clonar (primera vez)
git clone --branch feature/radimagenet https://github.com/Ayax911/Federal-Learning.git
cd Federal-Learning

# Opción B: Actualizar (si ya existe)
cd Federal-Learning
git fetch origin feature/radimagenet
git checkout feature/radimagenet
git pull --ff-only
```

### 3.2 Instalar dependencias

```bash
# (O usar setup_node.sh si no tienes venv)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 3.3 Crear estructura de directorios

```bash
mkdir -p weights manifests data/mammobench/images runs
```

### 3.4 Copiar pesos desde servidor

```bash
# Desde máquina remota (reemplazar user@servidor):
scp user@servidor:/path/to/Federal-Learning/weights/RadImageNet-resnet50.pth weights/
```

### 3.5 Copiar manifest correspondiente

**PARA CADA NODO**, copiar su manifest específico:

```bash
# Node 0 (en máquina remota asignada a Node0):
scp user@servidor:/path/to/Federal-Learning/manifests/node0_manifest.csv ./manifests/

# Node 1 (en máquina remota asignada a Node1):
scp user@servidor:/path/to/Federal-Learning/manifests/node1_manifest.csv ./manifests/

# ... etc para Node2, Node3, Node4, Node5
```

### 3.6 Copiar imágenes correspondientes

**PARA CADA NODO**, copiar SOLO las imágenes de su dataset:

```bash
# El manifest referencia qué imágenes usar.
# Copiar TODAS las imágenes de mammo-bench, manteniendo subcarpetas.

rsync -avz user@servidor:/ruta/a/mammo-bench/images/ data/mammobench/images/
```

**O si tienes acceso directo a almacenamiento compartido (NFS/SMB):**

```bash
mount -t nfs servidor:/almacenamiento/mammo-bench /mnt/mammo
ln -s /mnt/mammo/images data/mammobench/images
```

### 3.7 Exportar variable de entorno

```bash
# En cada terminal donde ejecutes el nodo:
export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights

# O agregarlo permanentemente a .venv/bin/activate:
echo 'export FEDMAMMOBENCH_RADIMAGENET_DIR=$VIRTUAL_ENV/../weights' >> .venv/bin/activate
```

### 3.8 Verificar setup del nodo

```bash
# Reemplazar:
#   <NODE_ID>: 0, 1, 2, 3, 4, o 5
#   <SERVER_IP>: IP LAN del servidor (ej: 192.168.14.184)

bash scripts/verify_setup.sh <NODE_ID> <SERVER_IP>

# Ejemplo:
bash scripts/verify_setup.sh 0 192.168.14.184
```

Expected output:
```
✓ Config cliente (todos)
✓ Pesos RadImageNet encontrados
✓ Directorio de imágenes
✓ Manifest Node0
✓ Variable $FEDMAMMOBENCH_RADIMAGENET_DIR = /path/to/Federal-Learning/weights

Node0 listo. Ejecutar con:
  bash scripts/start_client.sh 0 192.168.14.184
```

---

## PASO 4: Arrancar Todos los Nodos

Una vez que el **servidor está corriendo** y los **6 nodos están configurados**, arrancar los nodos.

**EN CADA MÁQUINA REMOTA**, abrir una terminal y ejecutar:

```bash
cd /path/to/Federal-Learning
export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights

# Node 0:
bash scripts/start_client.sh 0 192.168.14.184

# Node 1 (en otra máquina):
bash scripts/start_client.sh 1 192.168.14.184

# Node 2:
bash scripts/start_client.sh 2 192.168.14.184

# Node 3:
bash scripts/start_client.sh 3 192.168.14.184

# Node 4:
bash scripts/start_client.sh 4 192.168.14.184

# Node 5:
bash scripts/start_client.sh 5 192.168.14.184
```

**Reemplazar `192.168.14.184` con la IP LAN real del servidor.**

---

## PASO 5: Monitorear Entrenamiento

### En el servidor central:

Deberías ver logs como:

```
Waiting for 6 clients to be available...
[Round 1/30] 6 clients connected
[Round 1/30] Training...
[Round 1/30] Evaluating...
  Aggregated AUC-PR: 0.742
  Aggregated Loss: 0.523
[Round 2/30] 6 clients connected
...
```

### En cada nodo:

Deberías ver logs como:

```
[nodo0] Servidor      : 192.168.14.184:8080
[nodo0] Manifest      : manifests/node0_manifest.csv (5202 filas)
[nodo0] Imágenes      : data/mammobench/images
[Round 1] Training local... loss=0.45
[Round 1] Evaluating local... auc_pr=0.71
```

---

## PASO 6: Resultados

### En el servidor:

Los resultados se guardan en:

```
runs/exp01_fedavg_resnet50_6nodes/
├── server.log              # Logs de agregación
├── config.snapshot.yaml    # Config utilizado
├── model_round_0.pth
├── model_round_1.pth
├── ...
├── model_round_30.pth      # Modelo final
└── metrics_history.json    # Métricas por ronda
```

### En cada nodo:

```
runs/exp01_fedavg_resnet50/node<ID>/
├── client_<ID>.log
├── metrics_history.json
└── predictions_round_<R>.csv
```

---

## TROUBLESHOOTING Rápido

| Problema | Solución |
|----------|----------|
| **"Waiting for 6 clients..." (no arranca)** | Verificar que los 6 nodos están ejecutándose. Esperar 30-60s. |
| **"Connection refused:8080"** | Usar IP LAN correcta, no `localhost`. Hacer: `hostname -I` en servidor. |
| **"No manifest found"** | Ejecutar: `python scripts/partition_mammobench.py --csv data/mammobench/mammo-bench.csv --out manifests/ --nodes 6` |
| **"No RadImageNet weights"** | Descargar: `wget https://github.com/BMEII-AI/RadImageNet/releases/.../RadImageNet-resnet50.pth -O weights/RadImageNet-resnet50.pth` |
| **"No images found"** | Copiar imágenes a: `data/mammobench/images/` (mantener subcarpetas) |
| **Variable $FEDMAMMOBENCH_RADIMAGENET_DIR no seteada** | Ejecutar: `export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights` |

---

## CHEATSHEET

```bash
# SERVIDOR CENTRAL
bash scripts/verify_setup.sh server
export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights
bash scripts/start_server.sh configs/exp01_fedavg_resnet50_6nodes_server.yaml

# CADA NODO (reemplazar NODE_ID y SERVER_IP)
bash scripts/verify_setup.sh <NODE_ID> <SERVER_IP>
export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights
bash scripts/start_client.sh <NODE_ID> <SERVER_IP>
```

---

## SIGUIENTE: Documentación Completa

Para más detalles, opciones avanzadas y troubleshooting profundo, ver:
- **[docs/SETUP_6NODES.md](SETUP_6NODES.md)** — Guía completa
- **[docs/DOCKER.md](DOCKER.md)** — Ejecución con contenedores
- **[.env.example](../.env.example)** — Variables de configuración
