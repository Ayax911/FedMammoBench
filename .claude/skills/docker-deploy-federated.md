# /docker-deploy — Desplegar experimento federado en Docker

Lanza un experimento federado en contenedores Docker (1 servidor + 5 clientes gRPC).

## Uso

```bash
/docker-deploy exp14
```

O con parámetros adicionales:

```bash
/docker-deploy exp14 --clean     # Limpiar contenedores previos (default: true)
/docker-deploy exp14 --monitor   # Monitorear hasta Round 5 (default: false)
```

## Qué hace

1. **Valida** que la config del experimento existe
2. **Limpia** contenedores previos (si `--clean`)
3. **Lanza servidor** gRPC en puerto 8080
4. **Lanza 5 clientes** (CMMD, InBreast, CDD-CESM, KAU-BCMD, DMID)
5. **Verifica conectividad** de cada cliente
6. **Monitorea ROUND 1** si `--monitor`

## Requisitos

- Docker con soporte GPU (`--gpus all`)
- Variables de entorno:
  - `REPO`: raíz del proyecto
  - `MAMMO_DATA`: directorio con imágenes
- Imagen: `ayax911/federal-learning:latest`
- Manifests en `manifests/`:
  - `cmmd-split.csv`
  - `inbreast-split.csv`
  - `cdd-cesm-split.csv`
  - `kau-bcmd-split.csv`
  - `dmid-split.csv`

## Salida

Muestra:
- ✅/❌ Estado de cada contenedor
- Muestras cargadas por nodo
- Métricas de ROUND 1 cuando esté disponible

## Ejemplo: Monitorear exp14 hasta Round 5

```bash
/docker-deploy exp14 --monitor
```

Espera a que se complete ROUND 5 y muestra:
```
[ROUND 5]
configure_fit: strategy sampled 5 clients (out of 5)
loss=0.4521 auc=0.7234 f1=0.6891
```

## Detener/Cleanup

```bash
# Ver logs en tiempo real
docker logs -f exp14_server

# Detener todo
docker stop exp14_server exp14_client{1..5}

# Limpiar (borrar contenedores + volúmenes)
docker rm -f exp14_server exp14_client{1..5}
```

## Notas

- Cada cliente toma ~30-60s para cargar datos y conectarse
- ROUND 1 típicamente toma 3-5 minutos (depende del dataset)
- Los logs se guardan en `/app/runs/exp<NN>/` en los volúmenes montados
- GPU: Requiere `nvidia-docker` o Docker con `--gpus all`
