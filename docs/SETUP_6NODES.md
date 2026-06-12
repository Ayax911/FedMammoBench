# Guía Completa: Configuración de 6 Nodos + Servidor Central

Este documento describe cómo configurar y ejecutar un experimento federado con **6 nodos cliente + 1 servidor central** en máquinas separadas.

---

## 1. ARQUITECTURA GENERAL

```
┌─────────────────────────────────────────────┐
│       SERVIDOR CENTRAL (Este PC)            │
│  - Orquesta agregación federada (rondas)    │
│  - Guarda modelos, métricas y logs          │
│  - Pre-entrena opcionalmente con inbreast   │
│  Puerto: 8080                               │
└─────────────────────────────────────────────┘
                       ↑ gRPC
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   ┌────────┐    ┌────────┐    ┌────────┐
   │ Node0  │    │ Node1  │    │ Node2  │
   │  cmmd  │    │  dmid  │    │  ibia  │
   └────────┘    └────────┘    └────────┘
        ↑              ↑              ↑
        └──────────────┼──────────────┘
                       ↑
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   ┌────────┐    ┌────────┐    ┌────────┐
   │ Node3  │    │ Node4  │    │ Node5  │
   │cdd-cesm│    │kau-bcmd│    │  ddsm  │
   └────────┘    └────────┘    └────────┘
```

---

## 2. DISTRIBUCIÓN DE DATOS (PARTICIÓN 6 NODOS)

El dataset Mammo-Bench se divide en **6 nodos cliente + 1 servidor**:

| Entidad | Dataset | País/Fuente | Aprox. Filas | Rol |
|---------|---------|------|-------------|-----|
| **Nodo 0** | rsna | RSNA Screening | 36,460 | Cliente FL |
| **Nodo 1** | cmmd | China | 5,202 | Cliente FL |
| **Nodo 2** | inbreast | Portugal | 410 | Cliente FL |
| **Nodo 3** | cdd-cesm | Egipto | 800 | Cliente FL |
| **Nodo 4** | kau-bcmd | Arabia Saudita | 2,337 | Cliente FL |
| **Nodo 5** | dmid | Desconocido | 757 | Cliente FL |
| **Servidor** | ddsm | USA (CBIS-DDSM) | 10,400 | Pre-entrenamiento |

---

## 3. SETUP INICIAL (UNA SOLA VEZ)

### 3.1 En el SERVIDOR CENTRAL

```bash
cd /path/to/Federal-Learning

# Crear estructura de directorios
mkdir -p data/mammobench/images manifests weights runs

# Generar particiones de datos (divide mammo-bench.csv en 6 nodos)
python scripts/partition_mammobench.py \
  --csv data/mammobench/mammo-bench.csv \
  --out manifests/ \
  --nodes 6
```

**Esto crea:**
- `manifests/node0_manifest.csv` … `manifests/node5_manifest.csv`
- `manifests/server_train_manifest.csv` (opcional, para pre-entrenamiento del servidor)
- `manifests/partition_summary.txt` (verificación de partición)

### 3.2 Copiar pesos RadImageNet

```bash
# En el SERVIDOR:
# Descargar o copiar RadImageNet-resnet50.pth a:
cp /ruta/a/RadImageNet-resnet50.pth weights/
```

---

## 4. CONFIGURACIÓN POR NODO (PARA CADA MÁQUINA CLIENTE)

Cada nodo cliente debe tener:
- **Manifests**: CSV de registro del nodo (generado en servidor)
- **Imágenes**: Datos médicos reales del dataset correspondiente
- **Pesos**: Copia de RadImageNet-resnet50.pth
- **Config**: YAML con parámetros de entrenamiento

### 4.1 Estructura de Directorios en CADA NODO

```
~/fedmammobench/              (o tu directorio de instalación)
├── .venv/                    # Entorno virtual
├── configs/
│   ├── exp01_fedavg_resnet50_client.yaml    # MISMO para todos
│   └── ...
├── data/
│   └── mammobench/
│       └── images/           # Tus imágenes del dataset
│           ├── cmmd/         # (Node0: cmmd images)
│           ├── dmid/         # (Node1: dmid images)
│           ├── ibia/         # (Node2: ibia images)
│           ├── cdd-cesm/     # (Node3: cdd-cesm images)
│           ├── kau-bcmd/     # (Node4: kau-bcmd images)
│           └── ddsm/         # (Node5: ddsm images)
├── manifests/
│   ├── node0_manifest.csv    # (SOLO si este es node0)
│   ├── node1_manifest.csv    # (SOLO si este es node1)
│   └── ...
├── weights/
│   └── RadImageNet-resnet50.pth
├── scripts/
├── runs/                     # Logs y outputs (se genera)
└── requirements.txt
```

### 4.2 QUÉ CAMBIAR EN CADA NODO

#### **VARIABLE DE ENTORNO** (CRÍTICA)

```bash
# En CADA nodo, antes de ejecutar, exportar:
export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights
```

O en el `.venv/bin/activate`:

```bash
# Agregar al final de .venv/bin/activate:
export FEDMAMMOBENCH_RADIMAGENET_DIR="$VIRTUAL_ENV/../weights"
```

#### **MANIFEST CSV** (POR NODO)

Copiar del servidor solo el manifest del nodo correspondiente:

```bash
# En Node0:
cp /servidor/manifests/node0_manifest.csv ./manifests/

# En Node1:
cp /servidor/manifests/node1_manifest.csv ./manifests/

# ... etc para cada nodo
```

#### **DIRECTORIO DE IMÁGENES** (POR NODO)

Cada nodo necesita SOLO las imágenes de su dataset correspondiente:

```bash
# Node0 (cmmd):
# Copiar todas las imágenes de cmmd a: data/mammobench/images/

# Node1 (dmid):
# Copiar todas las imágenes de dmid a: data/mammobench/images/

# ... etc
```

**IMPORTANTE**: El manifest CSV referencia rutas relativas a `data/mammobench/images/`.
Si el manifest dice:
```
image_path,classification,source_dataset,...
cmmd/patient001/image.tif,Malignant,cmmd,...
```

Las imágenes deben estar en: `data/mammobench/images/cmmd/patient001/image.tif`

#### **CONFIG YAML** (IDÉNTICO PARA TODOS)

El archivo `configs/exp01_fedavg_resnet50_client.yaml` **es el MISMO para todos los nodos**.

**NUNCA EDITAR**. Los parámetros dinámicos se pasan por CLI:

```bash
bash scripts/start_client.sh <NODE_ID> <SERVER_IP> [DATA_DIR] [MANIFEST]
```

---

## 5. EJECUCIÓN

### 5.1 Opción A: UNA MÁQUINA CON CONTENEDORES (docker-compose)

En el servidor central, ejecutar todos los servicios:

```bash
cd /path/to/Federal-Learning

# Crear archivo .env (opcional, sobrescribe defaults)
cat > .env << 'EOF'
SERVER_CONFIG=configs/exp01_fedavg_resnet50_6nodes_server.yaml
CLIENT_CONFIG=configs/exp01_fedavg_resnet50_client.yaml
SERVER_ADDRESS=server:8080
SERVER_PORT=8080
NODE_DATA_DIR=data/mammobench/images
EOF

# Lanzar servidor + todos los 6 nodos
docker compose -f docker-compose.6nodes.yml --profile all up --build
```

### 5.2 Opción B: MÁQUINAS SEPARADAS (Script bash nativo)

#### En el SERVIDOR CENTRAL:

```bash
cd /path/to/Federal-Learning
export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights

# Obtener IP LAN del servidor
IP_LAN=$(hostname -I | awk '{print $1}')
echo "Servidor listening en: $IP_LAN:8080"

# Arrancar servidor
bash scripts/start_server.sh configs/exp01_fedavg_resnet50_6nodes_server.yaml
```

El servidor mostrará:
```
════════════════════════════════════════════════════════════════════
  Servidor federated learning
  Config  : configs/exp01_fedavg_resnet50_6nodes_server.yaml
  Escucha : 0.0.0.0:8080
  IP LAN  : 192.168.14.184:8080  ← USAR ESTA IP EN LOS NODOS
  Pesos   : /path/to/Federal-Learning/weights
════════════════════════════════════════════════════════════════════
```

#### En CADA NODO CLIENTE (máquinas separadas):

Reemplazar `192.168.14.184` con la IP LAN real del servidor.

```bash
# Node0 (en máquina remota 1)
cd ~/fedmammobench
export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights
bash scripts/start_client.sh 0 192.168.14.184

# Node1 (en máquina remota 2)
cd ~/fedmammobench
export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights
bash scripts/start_client.sh 1 192.168.14.184

# Node2 (en máquina remota 3)
cd ~/fedmammobench
export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights
bash scripts/start_client.sh 2 192.168.14.184

# ... etc para Node3, Node4, Node5
```

---

## 6. CHECKLIST PRE-LANZAMIENTO

Antes de ejecutar, verificar en CADA nodo:

- [ ] Manifest CSV existe: `manifests/node<ID>_manifest.csv`
- [ ] Imágenes copiadas: `data/mammobench/images/` contiene subcarpetas del dataset
- [ ] Pesos RadImageNet descargados: `weights/RadImageNet-resnet50.pth`
- [ ] Variable de entorno exportada: `echo $FEDMAMMOBENCH_RADIMAGENET_DIR`
- [ ] Venv activado: `which python` apunta a `.venv/bin/python`
- [ ] IP del servidor correcta en comando `start_client.sh`

Script de verificación (ejecutar en cada nodo):

```bash
#!/bin/bash
set -e
NODE_ID=$1
SERVER_IP=$2

echo "=== Verificación Node$NODE_ID ==="
[[ -f "manifests/node${NODE_ID}_manifest.csv" ]] && echo "✓ Manifest encontrado" || echo "✗ Manifest FALTA"
[[ -d "data/mammobench/images" ]] && echo "✓ Directorio imágenes encontrado" || echo "✗ Directorio imágenes FALTA"
[[ -f "weights/RadImageNet-resnet50.pth" ]] && echo "✓ Pesos descargados" || echo "✗ Pesos FALTAN"
[[ -n "$FEDMAMMOBENCH_RADIMAGENET_DIR" ]] && echo "✓ Variable de entorno seteada" || echo "✗ Variable de entorno NO seteada"
echo "  SERVER_IP: $SERVER_IP"
echo "✓ Listo para ejecutar: bash scripts/start_client.sh $NODE_ID $SERVER_IP"
```

---

## 7. MÉTRICAS Y RESULTADOS

### Servidor Central

El servidor guarda en `runs/exp01_fedavg_resnet50_6nodes/`:

```
├── server.log                    # Logs de agregación por ronda
├── config.snapshot.yaml          # Config utilizado
├── model_round_0.pth             # Modelo inicial
├── model_round_1.pth             # Modelo después de ronda 1
├── ...
├── model_round_30.pth            # Modelo final (ronda 30)
├── metrics_history.json          # Historial de métricas por ronda
└── predictions_round_<R>.csv     # (si save_predictions=true)
```

### Clientes

Cada cliente guarda en `runs/exp01_fedavg_resnet50/node<ID>/`:

```
├── client_0.log                  # Logs locales de Node0
├── metrics_history.json          # Métricas locales por ronda
└── predictions_round_<R>.csv     # Predicciones locales
```

### Métricas Guardadas

Por ronda se registran:

**Servidor:**
- `loss_aggregated`: pérdida promedio ponderada de clientes
- `auc_pr`: AUC-PR (métrica principal)
- `accuracy`: precisión
- `n_clients_connected`: número de clientes conectados

**Cada Cliente:**
- `loss`: pérdida local
- `auc_pr`: AUC-PR local
- `accuracy`: precisión local
- `samples_trained`: muestras utilizadas localmente

---

## 8. OPCIONES AVANZADAS

### 8.1 PRE-ENTRENAMIENTO DEL SERVIDOR (opcional)

El servidor central puede pre-entrenarse con el dataset `inbreast` antes de iniciar la federación:

1. Generar manifest de servidor:
```bash
python scripts/partition_mammobench.py \
  --csv data/mammobench/mammo-bench.csv \
  --out manifests/ \
  --nodes 6
# Crea: manifests/server_train_manifest.csv
```

2. Crear config de pre-entrenamiento (centralizado):
```bash
cp configs/radimagenet_resnet50_centralized.yaml configs/server_pretraining.yaml
```

3. Ejecutar pre-entrenamiento:
```bash
python scripts/run_centralized.py \
  --config configs/server_pretraining.yaml \
  --manifest manifests/server_train_manifest.csv
```

4. Copiar modelo pre-entrenado a checkpoint inicial:
```bash
cp runs/centralized_server_pretraining/model_best.pth \
   weights/initial_model.pth
```

5. Actualizar `exp01_fedavg_resnet50_6nodes_server.yaml`:
```yaml
model:
  checkpoint_path: weights/initial_model.pth  # Cargar pre-entrenamiento
```

### 8.2 DOCKER: UNA MÁQUINA REMOTA COMO NODO

Si quieres ejecutar un nodo en contenedor Docker:

```bash
# En máquina remota (Node2):
SERVER_ADDRESS=192.168.14.184:8080 \
NODE_DATA_DIR=data/mammobench/images \
NODE2_MANIFEST=manifests/node2_manifest.csv \
docker compose -f docker-compose.6nodes.yml --profile node2 up --build
```

### 8.3 GPU SUPPORT

Si las máquinas tienen GPU NVIDIA:

```bash
docker compose -f docker-compose.6nodes.yml -f docker-compose.gpu.yml \
  --profile all up --build
```

O en máquinas separadas:

```bash
# Server
docker compose -f docker-compose.6nodes.yml -f docker-compose.gpu.yml \
  --profile server up --build

# Each node
SERVER_ADDRESS=192.168.14.184:8080 \
docker compose -f docker-compose.6nodes.yml -f docker-compose.gpu.yml \
  --profile node0 up --build
```

---

## 9. TROUBLESHOOTING

### "Esperando a que se conecten los clientes..."
```
[server] Waiting for 6 clients to be available...
```

✓ Normal. El servidor espera a que TODOS los 6 nodos se conecten.
Ejecutar los clientes en las máquinas remotas.

### "Error: No RadImageNet weights found"

```bash
# Verificar:
ls -la weights/
# Si vacío:
export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights
# Descargar: https://github.com/BMEII-AI/RadImageNet
```

### "Error: Manifest not found"

```bash
# En cada nodo, copiar desde el servidor:
scp user@servidor:/path/to/Federal-Learning/manifests/node<ID>_manifest.csv ./manifests/
```

### "Error: No images found in data_dir"

```bash
# Verificar que las imágenes existen:
ls data/mammobench/images/
# Si vacío, copiar desde servidor o almacenamiento compartido
```

### Conexión rechazada (Connection refused) 8080

```bash
# Verificar IP del servidor:
hostname -I
# Usar la IP LAN (no localhost o 127.0.0.1)
bash scripts/start_client.sh 0 <IP_LAN_SERVIDOR>
```

---

## 10. REFERENCIAS

- Configuración: `configs/exp01_fedavg_resnet50_6nodes_server.yaml`
- Scripts: `scripts/partition_mammobench.py`, `scripts/start_server.sh`, `scripts/start_client.sh`
- Docker: `docker-compose.6nodes.yml`, `docker-compose.gpu.yml`
- Dataset: Mammo-Bench (6 fuentes geográficas, distribución Non-IID)
- Modelo: ResNet-50 con pesos RadImageNet
- Loss: BCEWithLogitsLoss (binary classification)
- Métrica primaria: AUC-PR
