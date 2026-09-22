# Resultados de Implementación y Verificación de Grad-CAM

## 1. Resumen de lo Implementado

Se implementó la funcionalidad de explicabilidad post-hoc mediante **Grad-CAM** para mamografías, respetando rigurosamente todos los contratos y decisiones de arquitectura del repositorio:

1. **[`src/interpretability.py`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/interpretability.py)**:
   - Dataclass `GradCAMResult(heatmap: np.ndarray, target_score: float)`.
   - `compute_gradcam(...)`:
     - Registra forward hook sobre `model[0][target_layer_index]` (default `layer4`, índice 7).
     - Habilita gradientes explícitamente (`torch.enable_grad()`), asegurando que `model.eval()` no impida la propagación.
     - Extrae la probabilidad de malignidad unificada a través de `LossSpec.probs(logits)[0]`, ejecutando `.backward()`.
     - Maneja tensores de entrada con `image.detach().requires_grad_(True)` para garantizar la derivabilidad sin importar si los pesos del backbone están congelados (`requires_grad=False`).
     - Calcula pesos de activación $w_k = \frac{1}{Z}\sum_{i,j} \frac{\partial y}{\partial A^k_{i,j}}$ y el mapa $CAM = \text{ReLU}(\sum_k w_k A^k)$.
     - Aplica upsampling bilineal a la resolución espacial de entrada `(H, W)` y normalización a `[0, 1]` con guardia de división por cero `+ 1e-8`.
     - Trata casos de gradiente muerto o score saturado devolviendo una matriz de ceros sin lanzar excepciones.
   - `overlay_heatmap(...)`:
     - Aplica el mapa de color `'jet'` al heatmap y lo mezcla con la imagen base PIL RGB respetando el factor de transparencia `alpha`.

2. **[`src/gradcam.py`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/gradcam.py)**:
   - CLI y orquestador reproducible para inspección post-hoc.
   - Reconstrucción de dataset y modelo consistente con `src/evaluate.py` a partir de `run_dir/config.yaml`.
   - Modos de selección de imágenes:
     - `ids`: selección explícita por identificador de imagen (`ID_image`).
     - `misclassified`: alineación posicional estricta entre `split.val_df()`/`test_df()` y `predictions.csv` (`pd.concat` por posición), filtrando errores y ordenando por confianza del error (`|y_prob - 0.5|` descendente) para los primeros `n_per_class` falsos positivos y falsos negativos.
     - `sample`: selección determinista de los primeros `n_per_class` casos por etiqueta (`label_norm`).
   - Procesamiento de imagen para display:
     - Independiente de la imagen que ve el modelo (`eval_transform`).
     - Para imágenes TIFF modo `"F"`, estiramiento de contraste por percentiles `[1, 99]` reescalado a uint8 `[0, 255]` y replicado a 3 canales.
     - Para imágenes convencionales, conversión estándar a RGB.
   - Nomenclatura determinista de archivos guardados: `<ID_image>_true<label>_score<prob>.png`.

3. **Documentación y enlaces actualizados**:
   - [`src/DOCS.md`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/DOCS.md): Árbol de archivos y secciones detalladas de uso para `interpretability.py` y `gradcam.py`.
   - [`CLAUDE.md`](file:///home/akira/snap/steam/federallearning/FedMammoBench/CLAUDE.md): Import check actualizado con `import src.gradcam`.

---

## 2. Resultados de la Verificación Sintética

Se ejecutó la suite de verificación sintética completa descrita en `docs/GRADCAM.md`:

| Prueba | Condición evaluada | Resultado |
|---|---|---|
| **1. Forma y rango del heatmap** | Tensor sintético `[1, 3, 224, 224]` con modelo `resnet50_imagenet_v2` + `StandardMLPHead` | `heatmap.shape == (224, 224)`, `min=0.0`, `max=0.99999` :white_check_mark: |
| **2. Backward real** | Propagación de gradiente sobre `model[0][7]` (layer4) con pesos congelados | Activaciones y gradientes calculados correctamente (`grad is not None`), sin mockeo :white_check_mark: |
| **3. Composición y robustez de overlay** | PIL RGB con heatmap normal y caso degenerado (gradientes cero / ReLU muerta) | Imagen RGB generada sin error, dimensiones exactas :white_check_mark: |
| **4. Alineación posicional `predictions.csv`** | Manifest sintético + `predictions.csv` con selección `misclassified`, `ids` y `sample` | Falsos positivos y falsos negativos seleccionados en sus posiciones exactas :white_check_mark: |
| **5. Manejo de imágenes TIFF modo `F` y RGB** | Apertura, percentile contrast stretching uint8 y generación de overlays | Overlays guardados correctamente en disco para ambos tipos :white_check_mark: |
| **6. Chequeo de importaciones** | `.venv/bin/python -c "import src.cli; import src.evaluate; import src.gradcam"` | Código de salida 0 (limpio) :white_check_mark: |

---

## 3. Comandos de Ejemplo para Ejecución en Workstation

Una vez disponibles las imágenes reales y checkpoints entrenados en la máquina de cómputo, se puede ejecutar directamente:

```bash
# Inspeccionar los 5 falsos positivos y falsos negativos más confiados del test set
.venv/bin/python -m src.gradcam \
    --config configs/exp05_fedmammobench_full_weighted.yaml \
    --checkpoint runs/exp05_fedmammobench_full_weighted/weights/best_epoch123.pt \
    --split test \
    --select misclassified \
    --n-per-class 5

# Inspeccionar imágenes específicas
.venv/bin/python -m src.gradcam \
    --config configs/exp05_fedmammobench_full_weighted.yaml \
    --checkpoint runs/exp05_fedmammobench_full_weighted/weights/best_epoch123.pt \
    --split test \
    --select ids \
    --image-ids CM000042,CM000107
```
