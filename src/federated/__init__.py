"""Capa federada de FedMammoBench sobre Flower (gRPC real, sin simulación).

Un proceso servidor (`python -m src.federated.server`) + un proceso por nodo
(`python -m src.federated.client`), cada uno con su propio YAML. El servidor
NUNCA procesa imágenes: no tiene sección `data`, la evaluación es puramente
federada (cada nodo evalúa el modelo agregado sobre su val local y el
servidor promedia ponderado por muestras).

Este paquete se sienta en la misma capa que `cli.py`/`evaluate.py` en la
cadena de dependencias de CLAUDE.md: importa de `config`, `seed`,
`checkpoint`, `tracking`, `reporting`, `datasets/`, `models/`, `train/` y
`eval_pipeline` — nunca de `cli.py` — y nada fuera de `federated/` importa
de acá. Todos los puntos de contacto con la API de flwr viven en
`server.py` y `client.py`, para que una futura migración (la API
`start_server`/`start_client` está deprecada en la línea 1.x) quede
contenida en dos archivos.

Diseño completo y decisiones vs. el paquete legacy borrado (`ec55408`):
docs/FEDERATED_DESIGN.md. Contratos por método: src/federated/DOCS.md.
"""
