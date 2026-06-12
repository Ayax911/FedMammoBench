# Matriz de Configuración: Qué Cambiar en Cada Nodo

Esta es la referencia rápida de **exactamente qué debe tener cada nodo cliente**.

---

## RESUMEN: Archivo a Archivo

| Componente | Qué es | Cambio por Nodo | Dónde Obtenerlo |
|-----------|--------|-----------------|-----------------|
| **RadImageNet .pth** | Pesos pre-entrenados | **MISMO para todos** | Descargar URL (abajo) |
| **Manifest CSV** | Registro de imágenes del nodo | **DIFERENTE** (node0.csv, node1.csv, ...) | Generar en servidor |
| **Imágenes** | Dataset médicas reales | **DIFERENTE** (solo las de su dataset) | Copiar del servidor |
| **Config YAML** | Parámetros de entrenamiento | **MISMO para todos** | Incluido en repo |
| **Variable de entorno** | Ruta a pesos | **MISMO** ($PWD/weights) | Exportar en cada terminal |

---

## TABLA DETALLADA POR NODO

### **Nodo 0 (cmmd — China)**

```
┌─────────────────────────────────────────────────────────────┐
│ ENTIDAD: Nodo 0 (cmmd)                                      │
├─────────────────────────────────────────────────────────────┤
│ DATASET:         cmmd (China)                               │
│ APROX. FILAS:    ~5,202                                     │
│ UBICACIÓN FÍSICA: Máquina remota 1 (opcional)               │
├─────────────────────────────────────────────────────────────┤
│ 1. PESOS RadImageNet                                        │
│    Archivo:      weights/RadImageNet-resnet50.pth           │
│    ¿Cambio?      NO — IDÉNTICO para todos                   │
│    Origen:       Descargar de:                              │
│                  https://github.com/BMEII-AI/RadImageNet    │
│    Tamaño aprox: ~500 MB                                    │
├─────────────────────────────────────────────────────────────┤
│ 2. MANIFEST CSV                                             │
│    Archivo:      manifests/node0_manifest.csv               │
│    ¿Cambio?      SÍ — ESPECÍFICO del nodo                   │
│    Origen:       Generar en servidor:                       │
│                  python scripts/partition_mammobench.py \   │
│                    --csv data/mammobench/mammo-bench.csv \  │
│                    --out manifests/ --nodes 6               │
│    Contenido:    2 columnas: image_path, classification     │
│    Ejemplo:      cmmd/patient001/image.tif,Malignant       │
│    Copiar a:     Máquina Node0 en mismo directorio           │
├─────────────────────────────────────────────────────────────┤
│ 3. IMÁGENES                                                 │
│    Directorio:   data/mammobench/images/                    │
│    ¿Cambio?      SÍ — SOLO imágenes del dataset cmmd        │
│    Estructura:   data/mammobench/images/cmmd/patient*/...   │
│    Aproximado:   ~5,202 imágenes (.tif o .jpg)             │
│    Origen:       Copiar del almacenamiento de mammo-bench   │
│    Sincronizar:  rsync, scp, NFS, o similar                 │
├─────────────────────────────────────────────────────────────┤
│ 4. CONFIG YAML                                              │
│    Archivo:      configs/exp01_fedavg_resnet50_client.yaml  │
│    ¿Cambio?      NO — IDÉNTICO para todos                   │
│    Incluido en:   Repositorio (git clone)                    │
│    Parámetros:   learning_rate, batch_size, epochs, etc.    │
│    NOTA:         Coherencia con servidor CRÍTICA            │
├─────────────────────────────────────────────────────────────┤
│ 5. VARIABLE DE ENTORNO                                      │
│    Variable:     FEDMAMMOBENCH_RADIMAGENET_DIR              │
│    ¿Cambio?      NO — MISMO para todos ($PWD/weights)       │
│    Exportar:     export FEDMAMMOBENCH_RADIMAGENET_DIR=\     │
│                    $PWD/weights                             │
│    Verificar:    echo $FEDMAMMOBENCH_RADIMAGENET_DIR         │
├─────────────────────────────────────────────────────────────┤
│ COMANDO FINAL (en máquina Node0):                           │
│ $ bash scripts/start_client.sh 0 192.168.14.184             │
│                                   ↑ IP LAN del servidor     │
└─────────────────────────────────────────────────────────────┘
```

---

### **Nodo 1 (dmid — Desconocido 1)**

| Componente | Archivo/Directorio | ¿Cambio? | Detalles |
|-----------|-------------------|----------|----------|
| **Pesos** | `weights/RadImageNet-resnet50.pth` | **NO** | Mismo archivo |
| **Manifest** | `manifests/node1_manifest.csv` | **SÍ** | De dataset `dmid` |
| **Imágenes** | `data/mammobench/images/dmid/...` | **SÍ** | ~757 imágenes de dmid |
| **Config YAML** | `configs/exp01_fedavg_resnet50_client.yaml` | **NO** | Idéntico |
| **Variable env** | `FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights` | **NO** | Mismo comando |
| **Comando** | `bash scripts/start_client.sh 1 192.168.14.184` | — | Reemplazar IP y node_id |

---

### **Nodo 2 (ibia — Desconocido 2)**

| Componente | Archivo/Directorio | ¿Cambio? | Detalles |
|-----------|-------------------|----------|----------|
| **Pesos** | `weights/RadImageNet-resnet50.pth` | **NO** | Mismo archivo |
| **Manifest** | `manifests/node2_manifest.csv` | **SÍ** | De dataset `ibia` |
| **Imágenes** | `data/mammobench/images/ibia/...` | **SÍ** | ~3,577 imágenes de ibia |
| **Config YAML** | `configs/exp01_fedavg_resnet50_client.yaml` | **NO** | Idéntico |
| **Variable env** | `FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights` | **NO** | Mismo comando |
| **Comando** | `bash scripts/start_client.sh 2 192.168.14.184` | — | Reemplazar IP y node_id |

---

### **Nodo 3 (cdd-cesm — Egipto)**

| Componente | Archivo/Directorio | ¿Cambio? | Detalles |
|-----------|-------------------|----------|----------|
| **Pesos** | `weights/RadImageNet-resnet50.pth` | **NO** | Mismo archivo |
| **Manifest** | `manifests/node3_manifest.csv` | **SÍ** | De dataset `cdd-cesm` |
| **Imágenes** | `data/mammobench/images/cdd-cesm/...` | **SÍ** | ~800 imágenes de cdd-cesm |
| **Config YAML** | `configs/exp01_fedavg_resnet50_client.yaml` | **NO** | Idéntico |
| **Variable env** | `FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights` | **NO** | Mismo comando |
| **Comando** | `bash scripts/start_client.sh 3 192.168.14.184` | — | Reemplazar IP y node_id |

---

### **Nodo 4 (kau-bcmd — Arabia Saudita)**

| Componente | Archivo/Directorio | ¿Cambio? | Detalles |
|-----------|-------------------|----------|----------|
| **Pesos** | `weights/RadImageNet-resnet50.pth` | **NO** | Mismo archivo |
| **Manifest** | `manifests/node4_manifest.csv` | **SÍ** | De dataset `kau-bcmd` |
| **Imágenes** | `data/mammobench/images/kau-bcmd/...` | **SÍ** | ~2,337 imágenes de kau-bcmd |
| **Config YAML** | `configs/exp01_fedavg_resnet50_client.yaml` | **NO** | Idéntico |
| **Variable env** | `FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights` | **NO** | Mismo comando |
| **Comando** | `bash scripts/start_client.sh 4 192.168.14.184` | — | Reemplazar IP y node_id |

---

### **Nodo 5 (ddsm — USA / CBIS-DDSM)**

| Componente | Archivo/Directorio | ¿Cambio? | Detalles |
|-----------|-------------------|----------|----------|
| **Pesos** | `weights/RadImageNet-resnet50.pth` | **NO** | Mismo archivo |
| **Manifest** | `manifests/node5_manifest.csv` | **SÍ** | De dataset `ddsm` |
| **Imágenes** | `data/mammobench/images/ddsm/...` | **SÍ** | ~10,400 imágenes de ddsm |
| **Config YAML** | `configs/exp01_fedavg_resnet50_client.yaml` | **NO** | Idéntico |
| **Variable env** | `FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights` | **NO** | Mismo comando |
| **Comando** | `bash scripts/start_client.sh 5 192.168.14.184` | — | Reemplazar IP y node_id |

---

## CHECKLIST: Instalación en Nueva Máquina (Nodo X)

```bash
# 1. Clonar repositorio
git clone --branch feature/radimagenet https://github.com/Ayax911/Federal-Learning.git
cd Federal-Learning

# 2. Crear estructura
mkdir -p weights manifests data/mammobench/images runs

# 3. Descargar/copiar pesos (IDÉNTICO para todos)
wget https://github.com/BMEII-AI/RadImageNet/releases/.../RadImageNet-resnet50.pth \
  -O weights/RadImageNet-resnet50.pth
# O copiar del servidor:
scp user@servidor:~/Federal-Learning/weights/RadImageNet-resnet50.pth weights/

# 4. Copiar manifest específico del nodo (DIFERENTE)
# Reemplazar X con el node_id (0-5)
scp user@servidor:~/Federal-Learning/manifests/nodeX_manifest.csv manifests/

# 5. Copiar imágenes del dataset correspondiente (DIFERENTE)
# Las imágenes deben ir a data/mammobench/images/
rsync -avz user@servidor:/almacenamiento/mammo-bench/images/ data/mammobench/images/

# 6. Exportar variable de entorno (MISMO comando para todos)
export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights

# 7. Verificar setup (MISMO comando, cambiar X y IP)
bash scripts/verify_setup.sh X 192.168.14.184

# 8. Ejecutar nodo (cambiar X y IP)
bash scripts/start_client.sh X 192.168.14.184
```

---

## GRÁFICO: FLUJO DE DATOS

```
┌─────────────────────────────────────────────────────────────┐
│            SERVIDOR CENTRAL (Este PC)                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ mammo-bench.csv (completo: ~23,700 filas)             │  │
│  │ RadImageNet-resnet50.pth (500 MB)                     │  │
│  │ Almacenamiento de imágenes (todos los datasets)       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │ partition + distribute
         └─────────────────────────────────────┐
                                                │
    ┌───────────────┬───────────────┬───────────┴──────┐
    ↓               ↓               ↓                  ↓
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│  Node0     │ │  Node1     │ │  Node2     │ │  Node3     │
│  cmmd      │ │  dmid      │ │  ibia      │ │  cdd-cesm  │
│            │ │            │ │            │ │            │
│ ✓ node0.   │ │ ✓ node1.   │ │ ✓ node2.   │ │ ✓ node3.   │
│   csv      │ │   csv      │ │   csv      │ │   csv      │
│ ✓ pesos    │ │ ✓ pesos    │ │ ✓ pesos    │ │ ✓ pesos    │
│ ✓ cmmd/    │ │ ✓ dmid/    │ │ ✓ ibia/    │ │ ✓ cdd-cesm/│
│   imgs     │ │   imgs     │ │   imgs     │ │   imgs     │
│ ✓ config   │ │ ✓ config   │ │ ✓ config   │ │ ✓ config   │
└────────────┘ └────────────┘ └────────────┘ └────────────┘

    ┌───────────────┬───────────────┬───────────┐
    │               │               │           │
    ↓               ↓               ↓           ↓
┌────────────┐ ┌────────────┐ ┌────────────┐
│  Node4     │ │  Node5     │ │  Servidor  │
│  kau-bcmd  │ │  ddsm      │ │  inbreast  │
│            │ │            │ │  (opcional)│
│ ✓ node4.   │ │ ✓ node5.   │ │            │
│   csv      │ │   csv      │ │            │
│ ✓ pesos    │ │ ✓ pesos    │ │            │
│ ✓ kau-bcmd/│ │ ✓ ddsm/    │ │            │
│   imgs     │ │   imgs     │ │            │
│ ✓ config   │ │ ✓ config   │ │            │
└────────────┘ └────────────┘ └────────────┘
```

---

## RESUMEN: "QUÉ CAMBIAR"

### ✓ IDÉNTICO para todos los nodos:
1. **RadImageNet-resnet50.pth** — Pesos pre-entrenados (500 MB)
2. **exp01_fedavg_resnet50_client.yaml** — Config de entrenamiento
3. **FEDMAMMOBENCH_RADIMAGENET_DIR** — Variable de entorno

### ✗ DIFERENTE por nodo:
1. **Manifest CSV** — node0.csv, node1.csv, ..., node5.csv
2. **Imágenes** — Copiar solo dataset del nodo correspondiente
3. **node_id** en comando `start_client.sh` — 0, 1, 2, 3, 4, o 5

---

## LÍNEA DE COMANDO RÁPIDA

```bash
# Para cualquier nodo (reemplazar X = 0-5, IP = servidor)
export FEDMAMMOBENCH_RADIMAGENET_DIR=$PWD/weights
bash scripts/start_client.sh X 192.168.14.184
```

¡Eso es todo!
