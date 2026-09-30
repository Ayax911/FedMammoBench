# Decisión: dejar de auditar el código, congelar lo centralizado y pasar a lo federado

**Fecha:** 2026-09-24 · **Estado:** propuesta para discusión · **Alcance:** FedMammoBench (y su paridad con el proyecto INC)

Todas las cifras de este documento salen de artefactos del repositorio y se pueden reproducir; la sección
[Fuentes y reproducibilidad](#fuentes-y-reproducibilidad) indica de dónde sale cada una.

Dos términos que se usan en todo el documento:

- **AUC pooled:** AUC sobre las 834 imágenes de test juntas. Es la cifra que se venía reportando (exp37 = 0,902).
- **AUC intra-base:** AUC calculado solo con pares maligno–benigno *de la misma base de datos*. Mide cuánto
  distingue el modelo la lesión, sin premiarlo por reconocer de qué hospital viene la imagen (§2.4).

---

## Resumen

1. **El código está verificado; seguir auditándolo ya no tiene dónde encontrar la causa del techo.** Tres
   auditorías, dos pruebas de paridad con INC y controles positivos y negativos coinciden: el pipeline no filtra
   datos entre splits ni tiene errores que inflen o hundan las métricas. Lo único encontrado son 5 imágenes
   duplicadas en los datos de origen (CMMD), con un efecto ≤ 0,005 de AUC (§1.2).
2. **Lo centralizado llegó a una meseta.** 34 corridas quedan entre 0,79 y 0,90 de AUC. Las tres mejores
   (exp37, exp31, exp28) son estadísticamente indistinguibles; dentro de cada base difieren en menos de 0,01.
3. **El techo no está en los splits ni en el ajuste de hiperparámetros.** train, val y test son
   indistinguibles, y 0,066 del 0,902 de exp37 viene de reconocer la base de datos. La capacidad real de
   distinguir malignidad es ~0,83 y depende de cómo está planteada la tarea (mamografía completa a 224×224),
   algo que ningún ajuste cambia.
4. **Lo pendiente —y el aporte de la tesis— está en lo federado.** Con la misma receta que exp37, el
   federado (exp46, FedProx) queda a **0,030** del centralizado dentro de cada base, no a 0,055 como sugiere el
   AUC pooled. Además aparece una pregunta de investigación concreta: kau-bcmd pierde 0,18 de AUC al federarse.
5. **Hay una condición bloqueante:** la comparación con "cada base entrenada sola" (exp24–27) usó otra
   receta, mucho más débil. Antes de afirmar que federarse conviene, hay que re-correr esas 4 corridas con la
   receta de exp37 (§5.1).

---

## 1. Por qué dejar de revisar el código

### 1.1 Lo que ya está verificado

| Qué se verificó | Resultado | Fuente |
|---|---|---|
| Fuga de información en el pipeline | Con etiquetas aleatorias, AUC de test = **0,5000** exacto; con señal controlada, **1,0000** | `AUDITORIA_SOBREAJUSTE_RESULTADOS.md` §4 |
| Pacientes disjuntos | 2.534 pacientes, ninguno compartido entre train, val y test | ídem §2 |
| Alineación imagen–etiqueta | 834/834 filas de test coinciden | ídem §2 |
| Augmentation solo en train, `eval()` + `no_grad`, BatchNorm congelada, recarga del mejor checkpoint | Verificado en código y en ejecución | ídem §2 |
| Integridad de los datos | 8.341 imágenes: 0 faltantes, 0 corruptas; DataLoader con 4 workers sin excepciones ni valores no finitos | `AUDITORIA_IMAGENES_RESULTADOS.md`, `AUDITORIA_IMAGENES_FASE2_RESULTADOS.md` |
| Capa federada | Envío y recuperación de parámetros idénticos bit a bit; los clientes reusan el mismo loop que el centralizado | `AUDITORIA_SOBREAJUSTE_RESULTADOS.md` §H7 |
| Paridad de carga FedMammoBench ↔ INC | Tensores idénticos (`torch.equal`, máx \|Δ\| = 0) | `RESNET_SCRATCH_RESNET18_RESULTADOS.md` §2 |
| Paridad de entrenamiento FedMammoBench ↔ INC | Mismo MNIST (64×64 TIFF), misma configuración: **99,56 %** vs **99,55 %** de accuracy en test | `configs/exp_mnist_smoketest.yaml` · INC `run_mnist_smoketest.sh` |
| Splits de la misma distribución | Un clasificador no logra distinguir train de val+test (AUC 0,507, p = 0,33) | `runs/centralizado/split_shift_audit/` (§2.3) |

La fila de MNIST es el control positivo que faltaba: ante una tarea fácil, ambos pipelines aprenden (99,5 %) y,
con la misma configuración, dan el mismo resultado. Los pipelines saben aprender; el techo está en lo que hay
que aprender.

### 1.2 Lo que queda abierto, y por qué no explica el techo

| Hallazgo | Tipo | Efecto medido |
|---|---|---|
| 5 imágenes de test duplicadas píxel a píxel en train, todas de CMMD. Hay una paciente registrada en las dos cohortes (D1-0202 = D2-0284: 4 imágenes) y una imagen con etiquetas contradictorias (CM000814 maligna = CM000475 benigna) | Datos, no código | Quitarlas mueve el AUC ≤ 0,005 (exp37: +0,0005 intra-base; exp46: +0,005) |
| `patience` desacoplada de `epochs` en el sweep (H2) | Configuración | Cómputo desperdiciado; el checkpoint guardado es el correcto |
| `>` vs `>=` en el umbral (H5) | Código, menor | Ninguna predicción tiene probabilidad exactamente 0,5: sin efecto |
| Validaciones de configuración que fallan en silencio (H6) | Deuda técnica | Solo afectan combinaciones mal configuradas |
| INC: la columna `Val_BCE-Loss` aplica CrossEntropy sobre probabilidades ya pasadas por softmax (no puede bajar de 0,313), y `training.py` no llama a `plot_loss_curve` | Reporte de INC | Afecta cómo se leen las curvas de INC, no ningún número de FedMammoBench |

El argumento central: **el mismo techo aparece en 34 configuraciones** con distintos backbones, cabezas,
pérdidas, optimizadores y manifests. Un error capaz de producir un techo común a todas tendría que estar en
un componente compartido (carga de datos, splits, loop, evaluación). Esos componentes son justamente los que
cubren los controles de §1.1, y ahí no aparece nada.

### 1.3 Cuándo reabrir la revisión

Solo ante un síntoma concreto:

- AUC exactamente 0,5000, NaN o pérdidas no finitas con datos reales.
- val y test que difieran más de ~0,05 en una corrida que no fue elegida por un sweep. Hoy coinciden:
  exp05 0,820 vs 0,818; exp57 0,819 vs 0,821.
- Un AUC pooled del federado que no cuadre con sus AUC por nodo.
- Un checkpoint cuyas métricas no se reproduzcan con `src.evaluate`.
- Cualquier cambio en `src/datasets/` o `src/train/`: repetir el smoke test de MNIST y
  `scripts/split_shift_audit.py` antes de confiar en resultados nuevos.

---

## 2. Por qué no seguir con lo centralizado

### 2.1 Meseta: 34 corridas, un solo techo

- 34 corridas con evaluación completa: AUC de test entre **0,792** (exp09) y **0,902** (exp37).
- Se movieron todas las perillas razonables: cuánto del backbone se entrena, ImageNet vs RadImageNet, varios
  tamaños y profundidades de cabeza, dropout de 0,3 a 0,5, weight decay, label smoothing, input dropout,
  ponderación de clases, un sweep bayesiano de ~300 trials y un ensemble de 4 modelos.
- Lo único que movió claramente el resultado fue entrenar el backbone de ImageNet completo (exp17 0,840 →
  exp22 0,887). Después, nada más movió la aguja: el ensemble de exp28–31 da **0,9015** y exp37 solo, **0,9016**.
- El modelo no está corto de capacidad: exp37 llega a AUC 0,996 en train (época 24) mientras val se queda en
  0,88. Memoriza train; lo que no logra es generalizar más allá de ~0,90.

### 2.2 Los mejores modelos están empatados

| Modelo | AUC pooled [IC 95 %] | AUC intra-base [IC 95 %] |
|---|---|---|
| exp37 (ganador del sweep) | 0,902 [0,873; 0,928] | 0,836 [0,787; 0,878] |
| exp31 | 0,889 [0,860; 0,916] | 0,832 [0,788; 0,871] |
| exp28 | 0,880 [0,850; 0,910] | 0,828 [0,784; 0,868] |

Diferencias pareadas, calculadas sobre las mismas imágenes con bootstrap por paciente:

| Comparación | Δ pooled [IC 95 %] | Δ intra-base [IC 95 %] |
|---|---|---|
| exp37 − exp28 | +0,022 [−0,002; +0,044] | +0,008 [−0,028; +0,040] |
| exp37 − exp31 | +0,013 [−0,007; +0,032] | +0,004 [−0,028; +0,034] |

Todos los intervalos incluyen el cero. **El sweep de ~300 trials compró una diferencia que no se distingue
del ruido**; dentro de cada base, menos de 0,01. Además, exp37 se eligió maximizando el AUC de val sobre el
mismo split de 254 pacientes, lo que vuelve optimista su cifra de val (H3 de la auditoría de sobreajuste). Otro
sweep optimizaría más ruido.

### 2.3 El problema tampoco son los splits

`scripts/split_shift_audit.py` comparó train, val y test de cinco maneras. Los p-valores se calcularon
permutando pacientes completos, y cada prueba tiene un control que demuestra que sí detecta diferencias cuando
existen:

| Prueba | Resultado entre splits | Control |
|---|---|---|
| Metadatos (base, clase, vista, lateralidad, densidad, BIRADS, anomalía, subtipo, edad) | Cramér's V ≤ 0,035; p ≥ 0,30 (edad: p = 0,07) | — |
| Estadísticas de píxeles por imagen | KS D ≤ 0,046 entre train y val+test | Entre bases: 0,35–0,75 |
| Un clasificador intenta adivinar el split desde la imagen | AUC 0,507 (p = 0,33); val vs test 0,461 | Adivinar la base: 0,999; la vista CC/MLO: 0,997 |
| ¿Lo aprendido en train sirve fuera? (sonda lineal de malignidad) | Pacientes nuevos de train 0,793 · val 0,825 · test 0,801 | El split oficial cae en el percentil 87 (val) y 45 (test) de 100 re-splits aleatorios |
| Distancia a la imagen más parecida de train | Mediana 0,884 · 0,885 · 0,886 | — |

El modelo rinde igual en val y test que en pacientes nuevos sacados del propio train. La brecha entre train
(0,996) y val/test (~0,90) es sobreajuste frente a una señal limitada, no una diferencia entre conjuntos.

### 2.4 Parte del 0,90 es reconocer la base de datos

- La prevalencia de malignos va de **4,6 %** (kau-bcmd) a **48,9 %** (cmmd), y la base se reconoce desde la
  imagen con AUC 0,999.
- Un puntaje que solo usa la base de origen, sin mirar la lesión, ya obtiene **AUC 0,726** en test.
- El 61 % de los pares maligno–benigno con que se calcula el AUC pooled comparan imágenes de bases distintas;
  en esos pares, reconocer la base basta para acertar.
- Contando solo pares de la misma base: exp37 pasa de 0,902 a **0,836**, exp31 de 0,889 a 0,832 y exp28 de
  0,880 a 0,828.

### 2.5 El techo depende de cómo está planteada la tarea

Las imágenes son mamografías completas guardadas a 224×224 px. El manifest trae columnas de ROI, máscara y
coordenadas de la lesión que el código no usa. Para superar ~0,83 intra-base habría que cambiar el
planteamiento: más resolución, recortes de la lesión, o detección y luego clasificación, como hace INC. Eso es
otra línea de trabajo: invalidaría todo el grid federado (exp41–56), que habría que volver a correr. Para una
tesis cuyo aporte es comparar esquemas de entrenamiento, el nivel absoluto actual alcanza, y la comparación es
válida sobre él.

### 2.6 Qué se conserva de lo centralizado

Lo centralizado no se abandona: se congela. **exp37 queda como referencia fija**, y el único trabajo
centralizado pendiente es correrla con 3 semillas para darle barras de error a la comparación final (§5.4).

---

## 3. Por qué lo federado es donde está el aporte

### 3.1 La receta de exp37 ya es la del federado

El grid federado (exp41–56) usa la receta de exp37: ImageNet v2, backbone entrenable completo,
`configurable_mlp` [512], AdamW (lr 6,45e-4), input dropout 0,45. "Llevar el mejor modelo al federado" ya está
hecho; no hay que re-correr nada.

Además, la estrategia ya está elegida:

- FedProx con μ = 0,1 (exp46) es la mejor, y FedAvg queda cerca.
- FedAdam y FedYogi quedan entre 0,58 y 0,73 de AUC agregado de validación, y colapsan con los parámetros por
  defecto de flwr.
- El barrido de μ entre 0,01 y 0,5 no encontró nada mejor que 0,1.

### 3.2 La brecha real con el centralizado es la mitad de lo que parecía

| AUC de test | Centralizado exp37 | Federado exp46 | Diferencia pareada [IC 95 %] |
|---|---|---|---|
| Pooled | 0,902 | 0,847 | −0,055 [−0,080; −0,032] |
| **Intra-base** | **0,836** | **0,806** | **−0,030 [−0,060; −0,002]** |
| cmmd (230 malignos en test) | 0,831 | 0,803 | −0,028 [−0,058; +0,002] |
| cdd-cesm (33) | 0,913 | 0,911 | −0,002 [−0,065; +0,046] |
| inbreast (8) | 0,855 | 0,816 | −0,039 [−0,172; +0,071] |
| kau-bcmd (6) | 0,941 | 0,761 | **−0,181 [−0,269; −0,106]** |

La brecha pooled se reduce a la mitad al medir dentro de cada base. El centralizado ve todas las bases a la vez
y puede aprovechar el atajo de §2.4; cada nodo federado solo ve la suya. Lo que queda:

- **cmmd:** diferencia pequeña, en el límite de la significancia. Por tamaño, cmmd concentra el 94 % de los
  pares intra-base, así que el −0,030 agregado es básicamente cmmd.
- **cdd-cesm:** empate.
- **inbreast:** demasiado pocos casos para concluir.
- **kau-bcmd:** la única pérdida grande.

### 3.3 La pregunta de investigación concreta: kau-bcmd

kau-bcmd tiene solo 4,3 % de malignos y pierde 0,18 de AUC al federarse. También pierde contra su propio
modelo local (0,909 solo vs 0,761 federado; −0,148 [−0,203; −0,098]), aunque ese modelo local usó una receta
claramente más débil (§5.1). Así que esa pérdida no puede atribuirse a la receta.

Es el caso clásico de un nodo cuya proporción de clases es muy distinta a la del resto. El paso natural es una
**personalización ligera por nodo**: ajustar localmente la cabeza o el umbral después de federar, o no agregar
las capas de BatchNorm.

Con solo 6 malignos en test, conviene confirmar la señal con más casos antes de afirmarla (§5.5).

---

## 4. Qué modelos tomar

| Rol | Modelo | Por qué |
|---|---|---|
| Referencia centralizada | **exp37**, congelado | Mejor punto medido y receta que ya usa el federado. exp28 y exp31 son equivalentes estadísticos (§2.2) |
| Referencia federada | **exp46**: FedProx, μ = 0,1, 30 rondas | Mejor del grid de estrategias y del barrido de μ |
| Referencia "cada base sola" | **Por re-correr** con la receta de exp37 | exp24–27 usaron otra receta (§5.1) |

---

## 5. Condiciones antes de sacar conclusiones

### 5.1 Re-correr "cada base sola" con la receta de exp37 (bloqueante)

exp24–27 usaron RadImageNet con el backbone congelado, `standard_mlp`, Adam (lr 1e-4), BCE y `norm_neg1_1`.
En lo centralizado, esa familia de receta rinde 0,794 (exp18), frente a 0,887–0,902 de la familia de exp37.
La ventaja actual del federado sobre "cada base sola" (+0,079 intra-base [+0,036; +0,120]) es del mismo orden
que esa desventaja de receta, así que **hoy no se puede atribuir a la federación**. La afirmación central de una
tesis federada —que a cada nodo le conviene federarse— necesita estas 4 corridas con la receta de exp37. Son
bases chicas, así que son baratas.

### 5.2 Limpiar los 5 duplicados

Hay que unificar a la paciente de CMMD registrada con dos IDs (P_000121_CM = P_001309_CM) y resolver el par
CM000814/CM000475, que tiene etiquetas contradictorias. El efecto medido es menor a 0,005, pero una es fuga de
datos y la otra una etiqueta errónea: deben quedar limpias antes de las cifras finales.

### 5.3 Estandarizar la evaluación

- Reportar siempre AUC pooled, intra-base y por nodo, con IC por paciente y diferencias pareadas
  (`scripts/compare_by_database.py`).
- Las métricas que dependen del umbral (sensibilidad, F1), solo después de calibrar el umbral por nodo en val
  (`scripts/calibrate_threshold.py --by-database`).

### 5.4 Tres semillas

Correr exp37, exp46 y las nuevas corridas de "cada base sola" con 3 semillas cada una. Ya hay configs de
semillas (exp34–36) que nunca se corrieron.

### 5.5 kau-bcmd e inbreast necesitan más casos

Tienen 6 y 8 malignos en test. Cualquier conclusión por nodo sobre ellas necesita más casos: juntar val y test,
o validación cruzada por paciente.

---

## 6. Objeciones previsibles

**"¿Y si hay un error que no vimos?"**
Los controles cubren justo los componentes que comparten todas las corridas:

- etiquetas aleatorias → 0,5000;
- señal controlada → 1,0000;
- MNIST → 99,5 % en ambos pipelines;
- carga idéntica bit a bit entre FedMammoBench e INC.

Un error capaz de fijar el mismo techo en 34 configuraciones tendría que estar ahí. Los criterios para reabrir
la revisión están en §1.3.

**"exp37 ganó el sweep; ¿por qué no otro sweep?"**
Porque su ventaja no se distingue de cero (§2.2), y porque haber ganado sobre el mismo split de val la vuelve
optimista.

**"¿No conviene subir primero el rendimiento centralizado?"**
Solo se puede cambiando el planteamiento de la tarea (§2.5), y eso obliga a re-correr todo lo federado. Si el
objetivo fuera un clasificador de uso clínico, ese sería el camino. Para comparar esquemas de entrenamiento,
el nivel actual alcanza.

**"Entonces el federado es peor."**
Dentro de cada base la diferencia es 0,030 [0,002; 0,060], y viene casi toda de cmmd, en el límite de la
significancia. La única pérdida grande es kau-bcmd, que es justo el caso que la investigación federada debe
explicar.

---

## 7. Límites de este análisis

- El AUC intra-base está dominado por cmmd (94 % de los pares). Las cifras individuales de kau-bcmd e inbreast
  son frágiles, y sus intervalos bootstrap, con tan pocos positivos, tienden a ser demasiado estrechos.
- Todo se midió sobre un único split: el oficial. La auditoría muestra que es típico, pero la comparación final
  ganaría con validación cruzada por paciente.
- La auditoría de splits usó features genéricas de ImageNet. Detectan diferencias tan sutiles como CC vs MLO
  (0,997), pero no garantizan detectar cualquier diferencia clínica.
- Federado y centralizado difieren en algo más que la agregación: los nodos no usan scheduler y calculan sus
  pesos de clase localmente. Es parte del diseño federado, no un factor a eliminar, pero conviene declararlo.

---

## Fuentes y reproducibilidad

| Cifras | De dónde salen |
|---|---|
| Auditorías de código, datos e imágenes (§1) | `docs/AUDITORIA_SOBREAJUSTE_RESULTADOS.md` (2026-09-20), `docs/AUDITORIA_IMAGENES_RESULTADOS.md`, `docs/AUDITORIA_IMAGENES_FASE2_RESULTADOS.md` (2026-09-22), `docs/RESNET_SCRATCH_RESNET18_RESULTADOS.md` |
| Tabla maestra centralizada y grid federado (§2.1, §3.1) | `.claude/context/experiments/CENTRALIZED.md`, `.claude/context/experiments/FEDERATED.md` |
| Paridad MNIST (§1.1) | `configs/exp_mnist_smoketest.yaml` → `runs/centralizado/exp_mnist_smoketest/`; INC `src/models/classification_images/run_mnist_smoketest.sh` → `results/inc_mnist_smoketest/` |
| Auditoría de splits, atajo de base y duplicados (§1.2, §2.3, §2.4) | `.venv/bin/python scripts/split_shift_audit.py` → `runs/centralizado/split_shift_audit/` (`report.json`, `audit.log`, figuras) |
| IC por paciente y diferencias pareadas (§2.2, §3.2, §3.3, §5.1) | `.venv/bin/python scripts/compare_by_database.py` (2.000 remuestreos por paciente, semilla 0), a partir de `runs/<exp>/test/predictions.csv` y `runs/federado/exp46_fedgrid_fedprox_r30/pooled_eval/test/predictions.csv` |
| Recetas comparadas (§3.1, §5.1) | `configs/exp37_hpsearch_v1_e7u7fprr.yaml`, `configs/federated/exp46_fedgrid_fedprox_r30/`, `configs/exp24–27_bydatabase_*.yaml` |
