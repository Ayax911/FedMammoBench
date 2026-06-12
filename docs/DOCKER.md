# Despliegue con Docker Compose

Contenedorización del experimento de aprendizaje federado (Flower gRPC).
El **servidor central** y cada **nodo cliente** corren en contenedores separados
a partir de una misma imagen, diferenciados por el comando de arranque.

## Componentes

| Archivo | Para qué sirve |
|---|---|
| `Dockerfile` | Imagen única `fedmammobench` (CPU por defecto). |
| `.dockerignore` | Evita copiar `data/`, `runs/`, `.venv/`, etc. al build. |
| `docker-compose.yml` | Servicios `server`, `node0`, `node1` (profiles). |
| `docker-compose.gpu.yml` | Override que da acceso a GPU NVIDIA. |
| `.env.example` | Parámetros del experimento (copiar a `.env`). |
| `run.sh` | Automatiza todos los procesos. |

## Datos persistentes (bind mounts)

Nada pesado se hornea en la imagen; entra por volúmenes montados desde el host:

| Host | Contenedor | Modo | Contenido |
|---|---|---|---|
| `./configs` | `/app/configs` | solo lectura | YAML de los experimentos |
| `./runs` | `/app/runs` | lectura/escritura | **resultados, logs, métricas, TensorBoard** |
| `./data` | `/app/data` | solo lectura | imágenes / datasets |
| `./manifests` | `/app/manifests` | solo lectura | particiones por nodo |
| `./checkpoints` | `/app/checkpoints` | solo lectura | pesos RadImageNet |

Editas los YAML en `configs/` desde tu editor y los resultados aparecen en
`runs/<nombre_experimento>/` en el host, sin entrar al contenedor.

## Puesta en marcha

```bash
cp .env.example .env      # ajusta configs, rutas y SERVER_ADDRESS
```

Coloca antes de arrancar:
- Pesos en `checkpoints/radimagenet/RadImageNet-resnet50.pth`.
- Manifests por nodo en `manifests/` (`scripts/partition_mammobench.py`).
- Imágenes en la ruta de `NODE_DATA_DIR`.

### Modo A — una sola máquina (pipeline completo)

Servidor y nodos en la misma red Docker; los nodos llegan al servidor por su
nombre de servicio (`SERVER_ADDRESS=server:8080`, valor por defecto).

```bash
./run.sh up --build          # CPU
./run.sh up --build --gpu    # GPU NVIDIA
```

### Modo B — máquinas separadas (despliegue real)

Mismo repo clonado en cada host. El nodo apunta a la **IP LAN del servidor**.

```bash
# Host del servidor:
./run.sh server --build --gpu -d

# Host de cada nodo (en su .env: SERVER_ADDRESS=192.168.1.10:8080):
./run.sh node 0 --build --gpu
./run.sh node 1 --build --gpu
```

> El cliente Flower reintenta la conexión con backoff, así que no importa el
> orden de arranque: si el servidor aún no escucha, el nodo espera y reintenta.

## Comandos de `run.sh`

```text
build              Construye la imagen.
up                 Levanta server + node0 + node1 (esta máquina).
server             Levanta solo el servidor.
node <0|1>         Levanta solo un nodo.
down               Detiene y elimina contenedores.
logs [servicio]    Logs en vivo (ej: ./run.sh logs server).
ps                 Estado de los contenedores.
shell [servicio]   Shell dentro de la imagen (debug).
clean              down + limpia huérfanos.

Opciones: --gpu  --build  -d/--detach
```

## GPU — requisitos del host

```bash
# Driver NVIDIA + nvidia-container-toolkit. Verificación:
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Con `--gpu`, el config (`device: auto`) selecciona CUDA automáticamente.
Los wheels de PyTorch de PyPI ya incluyen las librerías CUDA: el contenedor
sólo necesita **acceso** a la GPU, que concede `docker-compose.gpu.yml`.

## Resultados

```bash
ls runs/exp01_fedavg_resnet50/        # server.log, *_metrics.csv, config.snapshot.yaml, tb/
tensorboard --logdir runs/            # en el host, fuera del contenedor
```
