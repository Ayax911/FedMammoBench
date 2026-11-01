# Guion de exposición virtual — GNN en mamografía (~17 min)

Imágenes en esta misma carpeta (`runs/centralizado/gnn_population_graph/exposicion/`), numeradas en orden.
Abre la primera con `eog 01_cnn.png`, comparte solo esa ventana y avanza con la flecha derecha.

Las definiciones van **dentro del diálogo**, en el momento en que aparece cada término, para que nadie se
pierda. Al final hay un glosario de respaldo por si preguntan.

---

## Inicio (sin imagen, solo cámara) · 1 min 15 s

"Buenos días, gracias por el espacio. Me llamo [nombre] y les voy a mostrar un experimento con redes neuronales de grafos, o GNN, aplicadas a mamografías reales.

El punto de partida es este: los modelos de inteligencia artificial para mamografía clasifican cada imagen sola. Pero un radiólogo nunca trabaja así. Cada mama se fotografía en dos vistas: la **CC**, o craneocaudal, que es la toma de arriba hacia abajo, y la **MLO**, o mediolateral oblicua, que es la toma en diagonal desde el costado. El diagnóstico se hace mirando las dos.

Entonces la pregunta fue: si le damos al modelo las relaciones entre imágenes, ¿clasifica mejor un cáncer?

Trabajé con 8.341 mamografías de 2.534 pacientes, de cuatro bases de datos públicas de distintos países. Las separé en tres grupos: **entrenamiento**, con el que el modelo aprende; **validación**, con el que ajusto decisiones; y **prueba**, que solo se usa al final para medir. Una misma paciente nunca aparece en dos grupos, para que el modelo no haga trampa reconociendo a alguien que ya vio.

Antes de los resultados, dos minutos de conceptos para que todos partamos del mismo lugar."

---

## `01_cnn.png` · 1 min 15 s

"Esto es una **red convolucional, o CNN**: el tipo de red neuronal estándar para analizar imágenes.

[señala la izquierda] Entra la mamografía. [segunda columna] La red aplica **filtros**: pequeñas ventanas, por ejemplo de 3 por 3 píxeles, que recorren toda la imagen buscando un patrón. En las primeras capas detectan cosas simples: bordes, el contorno de la mama, texturas del tejido. Estos son los mapas reales que produce la red con esta mamografía.

[tercera columna] Capa tras capa, esos patrones simples se combinan en estructuras más complejas, y la imagen se va reduciendo hasta mapas de 7 por 7.

[derecha] Al final todo se resume en un **vector de 2.048 números**, que se llama *embedding*: una especie de huella digital numérica de la imagen. Con eso se decide si es maligna.

La red que usé se llama **ResNet50**, una CNN de 50 capas muy usada. Aquí apliqué **transfer learning**, o aprendizaje por transferencia: en vez de entrenar la red desde cero, reutilicé una que ya había aprendido a ver con otro conjunto de datos. Usé los pesos de **ImageNet**, que es una base de más de un millón de fotos de objetos cotidianos. Aunque no son mamografías, esa red ya sabe detectar bordes, texturas y formas, y eso sirve para cualquier imagen. No la reentrené: la usé congelada, solo para sacar ese vector. Lo único que se entrena es la GNN.

Lo importante para lo que sigue: la CNN mira cada imagen sola. No sabe que existe otra vista de la misma mama."

---

## `02_gnn.png` · 2 min

"Ahora, una **red neuronal de grafos, o GNN**. No trabaja con píxeles sino con un **grafo**: un conjunto de puntos, llamados **nodos**, unidos por líneas, llamadas **aristas**. Piensen en una red social: las personas son nodos y las amistades son aristas.

[panel 1] En mi caso cada nodo es una mamografía, y lleva como información el vector que sacó la CNN. Algunos nodos tienen diagnóstico conocido, rojo maligno y azul benigno. El del centro es el que queremos clasificar.

[panel 2] La operación central se llama **paso de mensajes**: cada vecino le envía su vector al nodo, y el nodo combina su propia información con la de sus vecinos. [señala la fórmula de abajo] Esa es toda la idea: mi nuevo vector es una mezcla de mi vector y el de mis vecinos.

[panel 3] Esto se hace en capas. Con dos capas, la información viaja dos saltos: también llegan los vecinos de los vecinos.

Hay varias formas de hacer esa mezcla, y probé las tres más conocidas. Una analogía: un médico tiene que dar un diagnóstico y consulta a sus colegas.

- La primera es **GCN**, *Graph Convolutional Network*. El médico promedia su opinión con la de todos los colegas, y todos pesan lo mismo, incluido él. Es simple, pero confía ciegamente en los vecinos.
- La segunda es **GraphSAGE**. El médico tiene su propia opinión y escucha el resumen del grupo, pero los trata por separado. Como guarda su propia opinión aparte, los vecinos nunca la tapan del todo.
- La tercera es **GAT**, *Graph Attention Network*. El médico decide a qué colega prestarle más atención en cada caso. A eso se le llama mecanismo de atención.

Fíjense en la consecuencia: todas asumen que los vecinos se parecen al nodo. Si las conexiones unen cosas que no tienen nada que ver, la mezcla contamina. Guarden esa idea."

---

## `03_pipeline.png` · 1 min 15 s

"Así junté las dos piezas. [izquierda] Estas son dos vistas reales de la misma mama de una paciente: la CC arriba y la MLO abajo.

Cada una pasa por la ResNet50 congelada y sale su vector de 2.048 números. [segunda caja] Ese vector lo reduzco a 128 números con **PCA**, o análisis de componentes principales, una técnica estadística que comprime la información quedándose con lo más importante. Esos 128 números son el nodo.

[grafo] Luego conecto los nodos: la línea naranja une las dos vistas de la misma mama, y las grises unen imágenes que se parecen. [derecha] La GNN procesa el grafo y da la probabilidad de cáncer.

Un detalle de rigor: todas las imágenes están en el grafo, pero el modelo solo aprende con los diagnósticos de entrenamiento. Los diagnósticos de prueba nunca se usan. A esto se le llama **aprendizaje transductivo**: el modelo ve todas las imágenes, pero solo conoce la respuesta de algunas.

Todo este flujo está en un solo script dentro de nuestro repositorio FedMammoBench, `scripts/gnn_population_graph.py`, construido con la librería **PyTorch Geometric**, que es la estándar para GNN en Python."

---

## `04_homofilia_barras.png` · 2 min

"La decisión más importante del experimento fue qué significa que dos mamografías estén conectadas. Probé cinco tipos de grafo. Cada uno tiene un nombre corto, que es como aparece en el código y en las gráficas. [señala cada barra al nombrarla]

- **knn**, de *k-nearest neighbors* o k vecinos más cercanos: conecto cada imagen con las 10 imágenes cuyo vector más se le parece.
- **pop**, de *population graph* o grafo de población: lo mismo, pero buscando parecidas solo dentro del mismo hospital, para que no se mezclen equipos distintos.
- **patient**: conecto todas las imágenes de la misma paciente, las dos mamas.
- **breast**: conecto solo las dos vistas, CC y MLO, de la misma mama.
- **pop+breast**: una mezcla de pop y breast.

Para comparar los grafos medí algo llamado **homofilia**. La palabra viene de 'amor por lo igual'. Es el porcentaje de conexiones que unen dos imágenes con el mismo diagnóstico: benigna con benigna, o maligna con maligna. Si es alta, los vecinos 'dicen la verdad' sobre el nodo. Si es baja, lo confunden.

[línea roja] Si conectara al azar daría 0,55, no 0,5, porque hay más benignas que malignas y coincidir por casualidad es más fácil.

[barras grises] knn da 0,65: apenas mejor que el azar. La red ve que dos imágenes se parecen, pero se parecen por el equipo, la densidad o el hospital, no por el cáncer. patient da 0,71, porque mezcla la mama izquierda con la derecha, que pueden tener diagnósticos distintos.

[barra naranja] Y breast, misma mama: 0,99.

Antes de seguir: ¿cuál creen que funcionó mejor? Escríbanlo en el chat."

*(Espera unos 10 segundos.)*

---

## `05_resultados_auc.png` · 2 min

"Los resultados. La medida que uso es el **AUC**, el área bajo la curva ROC. La forma intuitiva de entenderla: si tomo al azar una mamografía maligna y una benigna, el AUC es la probabilidad de que el modelo le dé un puntaje más alto a la maligna. 1 es perfecto, y 0,5 es lo mismo que lanzar una moneda.

Uso el AUC por tres razones: no depende de dónde ponga el corte entre 'benigno' y 'maligno', no lo engaña que haya más casos benignos, y es la misma métrica de nuestro proyecto, así que puedo comparar.

Cada barra es el promedio de 5 entrenamientos con distinto punto de partida aleatorio, lo que se llama **semillas**. La rayita negra muestra cuánto varió entre ellos.

[barra de abajo] Con el grafo knn, la GNN saca 0,799. [siguiente] Como control comparé con un **MLP**, o perceptrón multicapa: una red neuronal simple que recibe el mismo vector pero mira cada imagen sola, sin grafo. Saca 0,833. O sea: un grafo mal diseñado empeora el resultado.

[barra naranja] Con el grafo breast, la GNN sube a 0,857: 2,4 puntos por encima del modelo sin grafo.

[barra gris oscura] Para ser honesto: si al MLP simplemente le promedio a mano las predicciones de las dos vistas de cada mama, llega a 0,855, prácticamente lo mismo. Casi toda la ganancia viene de combinar las vistas. Lo valioso es que la GNN lo aprendió sola, sin que nadie le programara esa regla.

[barra superior] Como referencia, **exp28**, nuestro mejor modelo anterior, una CNN entrenada completa de punta a punta, llega a 0,88. Aquí la CNN estaba congelada, así que la comparación justa es contra el MLP."

---

## `06_hallazgo_homofilia_vs_auc.png` · 2 min 30 s

*(Habla más lento: este es el mensaje principal.)*

"Esta gráfica resume todo el experimento. Les explico cómo leerla.

Cada punto es una combinación de un tipo de GNN con un tipo de grafo: 3 GNN por 5 grafos, 15 puntos. [leyenda superior] El **color** indica el grafo: knn, pop, patient, breast o la mezcla. [leyenda inferior] La **forma** indica la GNN: círculo GCN, cuadrado GraphSAGE, triángulo GAT.

[eje horizontal] En horizontal está la homofilia del grafo, qué tan honestas son sus conexiones. [eje vertical] En vertical, cuánto gana o pierde esa GNN frente al MLP sin grafo. [zona verde] Arriba de la línea, el grafo ayudó. [zona roja] Abajo, el grafo perjudicó.

[puntos de la izquierda] Todos los grafos con homofilia baja quedan en la zona roja, en las tres GNN: promedian con vecinos de otro diagnóstico y se contaminan. [puntos naranjas, a la derecha] Con breast, las tres GNN ganan.

[puntos verde azulado, abajo] Un caso curioso: patient tiene un poco más de homofilia que knn, y aun así es el peor para GCN y GAT. Conecta siempre la mama izquierda con la derecha, así que cuando una tiene cáncer y la otra no, la GNN mezcla justo la información equivocada.

Ahora miremos los tres puntos naranjas de la derecha: son los tres mejores resultados, las tres GNN con el grafo de misma mama. ¿Por qué quedaron en ese orden?

[círculo naranja, el más alto] **Primero, GCN, con 0,857.** Casi todas las imágenes tienen un solo vecino: su otra vista. GCN mezcla en partes iguales, así que hace mitad CC y mitad MLO. Es justo lo que haría un radiólogo: darles el mismo peso a las dos vistas. Por eso gana, y por eso da casi lo mismo que el promedio manual.

[cuadrado naranja] **Segundo, GraphSAGE, con 0,842.** Hace casi la misma mezcla, pero tiene que aprender por su cuenta cuánto pesa cada parte, y para eso usa el doble de parámetros. Más cosas que aprender con los mismos datos se traduce en un resultado un poco peor y más variable.

[triángulo naranja] **Tercero, GAT, con 0,840.** Su atención debería elegir a quién escuchar, pero con un solo vecino no hay mucho que elegir: terminó repartiendo casi 50/50, igual que GCN. Además, durante el entrenamiento apaga conexiones al azar para no memorizar; si apaga la única que tiene, pierde la otra vista. Eso le resta un poco.

La conclusión: cuando el grafo es bueno, la GNN más simple es la que mejor funciona. Y en general, cómo se diseñan las conexiones importa mucho más que el tipo de red."

*(Versión corta: "GCN mezcla mitad y mitad las dos vistas, que es lo ideal; GraphSAGE hace lo mismo pero con más cosas que aprender; GAT termina repartiendo igual y además a veces pierde la conexión al entrenar.")*

---

## `07_sensibilidad_especificidad.png` · 1 min 15 s

"El AUC es abstracto. En clínica importan dos números más concretos.

[barras rojas] La **sensibilidad**: de cada 100 cánceres reales, cuántos detecta el modelo. Es lo que más importa en tamizaje, porque un cáncer no detectado es el peor error.
[barras azules] La **especificidad**: de cada 100 casos benignos, cuántos descarta correctamente. Mide cuántas falsas alarmas evitamos.

Para calcularlas hay que decidir un **umbral**: a partir de qué probabilidad decimos 'maligno'. Lo elegí con el **índice de Youden**, que busca el mejor equilibrio entre sensibilidad y especificidad, usando solo el grupo de validación, nunca el de prueba.

El MLP sin grafo detecta 77 % de los cánceres con 73 % de especificidad. La GNN con breast mantiene el 77 % de detección y sube la especificidad a 77 %, o sea menos falsas alarmas. La GNN con knn es la peor: detecta solo 64 %, se le escapan 13 puntos más de cánceres.

Las rayitas de error son amplias, así que la mejora en especificidad es una tendencia, no algo concluyente."

---

## `08_subgrafo.png` · 1 min

"¿Por qué funciona la conexión breast? Este es un pedazo real del grafo. Lo dibujé con **t-SNE**, una técnica que acomoda los puntos en el plano de modo que las imágenes que la red ve parecidas quedan cerca.

[líneas grises] Las conexiones por parecido son cortas, por construcción, y unen rojos con azules sin ningún orden. [líneas naranjas] Las de misma mama cruzan todo el mapa: para la red, la CC y la MLO de una misma mama se ven muy distintas. Un método automático por parecido nunca las conectaría. Esa información solo entra si la ponemos nosotros, con conocimiento clínico, y por eso es nueva para el modelo."

---

## Cierre (vuelve a la cámara) · 1 min

"Para cerrar, tres ideas. Primero: en una GNN, el diseño del grafo pesa más que el tipo de red, y la homofilia permite revisarlo antes de entrenar. Segundo: las conexiones con conocimiento clínico, como misma mama, le ganan a la similitud automática. Tercero: que dos imágenes se parezcan para una red no significa que tengan el mismo diagnóstico.

Como próximos pasos: usar los vectores de nuestra CNN entrenada, que ya llega a 0,88, para ver si el grafo suma encima; evaluar con pacientes completamente nuevas; y entrenar una GNN **federada**, es decir, entre varios hospitales sin que los datos salgan de cada uno.

Muchas gracias, quedo atento a sus preguntas."

---

## Solo si preguntan

- **`09_extra_roc.png`** (si piden la curva ROC): "La curva ROC muestra, para todos los umbrales posibles, cuántos cánceres detecto (eje vertical) contra cuántas falsas alarmas doy (eje horizontal). Cuanto más arriba a la izquierda, mejor. La GNN con breast queda arriba justo entre 10 % y 30 % de falsas alarmas, la zona útil en clínica. El AUC es el área bajo esta curva."
- **`10_extra_todas_las_combinaciones.png`** (si preguntan por las tres GNN): "Aquí están los 15 resultados agrupados por tipo de GNN, con el significado de cada grafo en la leyenda. Las tres muestran el mismo patrón: solo breast, en naranja, supera la línea gris del MLP sin grafo. GraphSAGE es la que menos pierde con grafos malos."
- **"¿Hay fuga de información?"** "No. Las pacientes de prueba no están en entrenamiento, así que las conexiones de misma mama nunca cruzan entre grupos, y sus diagnósticos nunca entran al entrenamiento. La homofilia, medida solo con imágenes de entrenamiento, da los mismos valores."
- **"¿Por qué no usaste la CNN entrenada?"** "Ese modelo no estaba disponible en esta máquina; es el siguiente paso."
- **"¿Con qué pesos hiciste transfer learning?"** "Con los pesos oficiales de ImageNet de torchvision para ResNet50, la versión 2. También probé los de RadImageNet, que se preentrenaron con 1,35 millones de imágenes médicas de tomografía, resonancia y ecografía. En los dos casos la red quedó congelada: solo extrae el vector, y lo único que se entrena es la GNN."
- **"¿Por qué el preentrenamiento médico no ayudó?"** "Con RadImageNet congelado todo quedó unos 7 puntos abajo: 0,785 en el mejor caso, contra 0,857 con ImageNet. Una hipótesis es que RadImageNet no tiene mamografías, y que sus vectores no encajan con nuestro preprocesamiento sin reentrenar. Lo verificaría con fine-tuning."
- **"¿Cómo verificaste las hipótesis de cada GNN?"** "Reentrené cada modelo con la semilla 0 y miré por dentro. En GAT leí los coeficientes de atención de la primera capa: con knn, 0,072 para sí misma y 0,069 por vecino, que es justo el reparto uniforme; con breast, 0,49 contra 0,44; con patient, 0,30 contra 0,24. En GraphSAGE comparé el tamaño de la matriz de pesos propia y la de vecinos: el cociente da entre 0,92 y 1,00 en los tres grafos. Y en breast, el 90 % de los nodos tiene exactamente un vecino. Son hipótesis respaldadas por esos datos, no demostraciones."
- **"¿GraphSAGE es mejor que GAT?"** "GCN gana claramente; el segundo y el tercer puesto están empatados dentro del margen de error."
- **"¿Dónde está el código?"** "En el repositorio FedMammoBench: `scripts/gnn_population_graph.py` hace todo el experimento y `scripts/gnn_presentation_figures.py` genera las figuras. Los resultados quedan en `runs/centralizado/gnn_population_graph/`."
- **"¿Cuánto tarda?"** "Todo el barrido, 160 entrenamientos, toma unos 3 minutos en una GPU de laptop."

---

## Glosario rápido (respaldo)

| Término | En una frase |
|---|---|
| CC / MLO | Las dos vistas de una mamografía: de arriba hacia abajo / en diagonal desde el costado |
| CNN | Red neuronal para imágenes que usa filtros que recorren la imagen |
| ResNet50 | CNN de 50 capas, preentrenada; aquí se usó congelada |
| Transfer learning | Reutilizar una red ya entrenada en otros datos en vez de entrenarla desde cero |
| ImageNet | Base de más de 1 millón de fotos cotidianas; sus pesos dieron los resultados principales |
| RadImageNet | Base de 1,35 millones de imágenes médicas (TC, RM, ecografía); se probó como alternativa |
| Congelada | La red no se modifica durante el entrenamiento; solo extrae vectores |
| Embedding / vector | Los números con que la CNN resume una imagen (2.048) |
| PCA | Técnica que comprime el vector a 128 números conservando lo importante |
| Grafo / nodo / arista | Red de puntos (imágenes) unidos por líneas (relaciones) |
| GNN | Red neuronal que trabaja sobre grafos mezclando cada nodo con sus vecinos |
| Paso de mensajes | Cada vecino envía su vector y el nodo los combina con el suyo |
| GCN | GNN que promedia con los vecinos, todos con el mismo peso |
| GraphSAGE | GNN que trata por separado la información propia y la de los vecinos |
| GAT | GNN que aprende a qué vecinos prestar más atención |
| MLP | Red neuronal simple sin grafo; nuestro control |
| knn | Grafo: cada imagen unida a sus 10 más parecidas |
| pop | Grafo: parecidas, solo dentro del mismo hospital |
| patient | Grafo: todas las imágenes de la misma paciente |
| breast | Grafo: solo CC ↔ MLO de la misma mama |
| pop+breast | Grafo: mezcla de pop y breast |
| Homofilia | % de conexiones que unen imágenes con el mismo diagnóstico (azar = 0,55) |
| Transductivo | El modelo ve todas las imágenes pero solo conoce el diagnóstico de las de entrenamiento |
| AUC | Probabilidad de que el modelo puntúe más alto a una maligna que a una benigna (0,5 azar, 1 perfecto) |
| Curva ROC | Detección vs. falsas alarmas para todos los umbrales; el AUC es su área |
| Sensibilidad | % de cánceres que el modelo detecta |
| Especificidad | % de benignos que el modelo descarta bien |
| Umbral / Youden | Probabilidad a partir de la cual se dice "maligno"; Youden elige el mejor equilibrio |
| Semilla | Punto de partida aleatorio; repetir con 5 semillas muestra cuánto varía el resultado |
| t-SNE | Técnica para dibujar en 2D vectores de muchas dimensiones, acercando los parecidos |
| exp28 | Nuestro mejor modelo previo: CNN entrenada completa, AUC 0,880 |
| Federado | Entrenar entre varios hospitales sin que los datos salgan de cada uno |

---

## Archivos del proyecto (para ti; no se leen en voz alta)

La GNN no está dentro de `src/`: es un script independiente que reutiliza el manifest y los splits del proyecto. Rutas relativas a `FedMammoBench/`.

| Paso del pipeline | Archivo | Dónde, dentro del archivo |
|---|---|---|
| Lista de imágenes, diagnósticos y splits | `manifests/fedmammobench_norm_0_1.csv` | columnas `preprocessed_image_path`, `classification`, `split`, `patient_id`, `laterality` |
| Imágenes (224×224, TIFF) | `/home/akira/snap/steam/preproccesed_julian/` | carpeta `norm_0_1/` |
| ResNet50 con pesos ImageNet / RadImageNet | `scripts/gnn_population_graph.py` | `build_backbone()`, línea 79 |
| Pesos RadImageNet | `../inc-project-models-classification-detection-main/ResNet50.pt` | se renombran capas con `RADIMAGENET_REMAP` |
| Extracción del vector de 2.048 | `scripts/gnn_population_graph.py` | `extract()`, línea 94 |
| PCA a 128 | `scripts/gnn_population_graph.py` | dentro de `main()`, línea 401 |
| Grafos knn y pop | `scripts/gnn_population_graph.py` | `knn_edges()`, línea 110 |
| Grafos patient y breast | `scripts/gnn_population_graph.py` | `patient_edges()`, línea 122; código de mama en la línea 381 |
| Armado de cada uno de los 5 grafos | `scripts/gnn_population_graph.py` | `build_graph()`, línea 134 |
| Homofilia | `scripts/gnn_population_graph.py` | `homophily()`, línea 148 |
| Modelos GCN / GraphSAGE / GAT / MLP | `scripts/gnn_population_graph.py` | clase `Net`, línea 155 (`GCNConv`, `SAGEConv`, `GATConv` de PyTorch Geometric) |
| Entrenamiento (solo etiquetas de train) | `scripts/gnn_population_graph.py` | `train_one()`, línea 181 |
| Evaluación (AUC, Youden, sensibilidad, especificidad) | `scripts/gnn_population_graph.py` | `evaluate()`, línea 216 |
| Figuras de esta exposición | `scripts/gnn_presentation_figures.py` | una función `fig_*` por figura |
| Resultados guardados | `runs/centralizado/gnn_population_graph/` | `results_summary.csv`, `results_per_seed.csv`, `report.json`, `emb_imagenet.npy`, `emb_radimagenet.npy` |
| Imágenes y guion de la exposición | `runs/centralizado/gnn_population_graph/exposicion/` | `01_cnn.png` … `10_extra_…png`, este archivo |

Para reproducir todo: `.venv/bin/python -m scripts.gnn_population_graph` (unos 3 minutos) y luego `.venv/bin/python -m scripts.gnn_presentation_figures`.
