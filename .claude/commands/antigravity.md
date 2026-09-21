# /antigravity — Encarga trabajo a Antigravity y audita lo que entrega

Formaliza el ciclo de dos pasadas: tú preparas el encargo (`tarea`), Antigravity lo ejecuta fuera de
esta sesión y escribe su informe en el repo, y tú lo auditas (`revisar`) antes de que nadie lo dé por
bueno. Nace de la auditoría de sobreajuste de 2026-09-20
(`docs/AUDITORIA_SOBREAJUSTE.md` → `docs/AUDITORIA_SOBREAJUSTE_RESULTADOS.md` →
`docs/FIXES_AUDITORIA_SOBREAJUSTE.md`): ese ciclo se hizo a mano una vez; esta skill es la versión
repetible.

**Por qué existe una revisión separada, no basta con que Antigravity entregue un informe**: en el
primer ciclo, un informe por lo demás sólido traía dos números mal (uno de ellos recomendaba la
dirección *opuesta* a la correcta para arreglar un bug, contradiciendo un comentario del código que
documentaba esa decisión a propósito). Nada de eso se habría detectado sin re-derivar los números y
releer el código citado.

## Uso

```
/antigravity tarea "<tema>" [--archivo docs/NOMBRE.md]
/antigravity revisar <ruta/al/informe.md> [--tarea docs/NOMBRE.md] [--aplicar]
```

**Ejemplos:**
- `/antigravity tarea "por qué exp37 no bate a exp28 con significancia"` → explora el código/artefactos
  primero, luego escribe `docs/EXP37_VS_EXP28.md`
- `/antigravity revisar docs/AUDITORIA_SOBREAJUSTE_RESULTADOS.md --tarea docs/AUDITORIA_SOBREAJUSTE.md`
  → audita ese informe contra el brief que lo originó
- `/antigravity revisar docs/FIXES_RESULTADOS.md --aplicar` → audita, y si el veredicto es limpio,
  aplica directamente las actualizaciones de documentación que correspondan (`CLAUDE.md`,
  `.claude/context/*/*.md`) en vez de solo proponerlas

## El ciclo completo (referencia)

1. Se identifica un tema que vale la pena investigar con más cómputo/tiempo del que tiene sentido
   gastar en esta sesión — típicamente algo que necesita ejecutar código real (datos, GPU) que esta
   sesión no tiene, o barrer mucho código/artefactos de una vez.
2. `/antigravity tarea "<tema>"` — explora el código primero, nunca redacta el brief a ciegas, y
   escribe `docs/<TEMA>.md` autocontenido.
3. Se le pasa ese archivo (o su ruta) a Antigravity.
4. Antigravity trabaja y escribe su propio informe en el repo (convención:
   `docs/<TEMA>_RESULTADOS.md`), a veces junto con cambios de código si el brief se los pidió.
5. Se avisa ("revisa esto") o se corre `/antigravity revisar <informe>`.
6. Se audita el informe (Fase 2) y se produce: ediciones directas a la documentación si el informe
   aguanta la verificación, o un `docs/FIXES_<TEMA>.md` con instrucciones corregidas si hay trabajo
   de seguimiento para otra vuelta de Antigravity.
7. El ciclo se repite hasta cerrar el tema; el `docs/FIXES_*.md` de una vuelta es el `tarea` de la
   siguiente.

---

## Fase 1 — Preparar la tarea (`tarea`)

### Antes de escribir una palabra del brief

No se redacta desde la memoria de la conversación. Se verifica en vivo, con las herramientas
disponibles en esta sesión:

- **Qué existe realmente en esta máquina** — mounts de datos, pesos, GPU/CUDA. El brief debe decir
  explícitamente qué puede y qué no puede ejecutar Antigravity si va a correr en el mismo entorno
  restringido que esta sesión (sin datos reales, ver `docs/AUDITORIA_SOBREAJUSTE.md` §1 como
  plantilla), o si va a correr en la workstation con datos — en ese caso decirlo también, porque
  cambia qué hipótesis son verificables de verdad.
- **Qué ya se sabe** — lanza 2-3 Explore (o audita directamente) sobre el código/artefactos
  relevantes antes de escribir hipótesis. El brief no debe pedirle a Antigravity que re-descubra algo
  que ya está resuelto en `CLAUDE.md`, `.claude/context/`, o un `docs/*_RESULTADOS.md` anterior — cítalo
  con archivo y línea y que quede fuera de alcance explícitamente.
- **Qué es una decisión deliberada de este repo, no un bug** — repásate la tabla de "Desviaciones
  deliberadas" de `.claude/context/code/ARCHITECTURE.md` y los comentarios largos que documentan el
  *porqué* (convención de la casa, ver "Conventions" en `CLAUDE.md`). Una hipótesis que en realidad
  choca con una decisión ya documentada debe decirlo en el brief, no dejar que Antigravity la
  redescubra como si fuera nueva.

### Estructura del brief (`docs/<TEMA>.md`)

Sigue el formato que ya funcionó en `docs/AUDITORIA_SOBREAJUSTE.md` y `docs/FIXES_AUDITORIA_SOBREAJUSTE.md`:

1. **Resumen ejecutivo** — qué se sabe ya y qué queda abierto, en cuatro líneas. Antigravity debe
   poder decidir si vale la pena seguir leyendo solo con esto.
2. **Restricción de entorno** — verificada con comandos reales, no supuesta.
3. **Lo que YA está descartado / cerrado** — con `archivo:línea` o número medido, para que no se
   repita el trabajo.
4. **El trabajo pedido**, como lista priorizada (hipótesis, tareas o fixes, según el tema) con, por
   cada ítem: cómo verificarlo, y el criterio de aceptación — qué resultado cuenta como "confirmado",
   cuál como "descartado". Si es un fix de código, el snippet exacto y el archivo, no una descripción
   vaga: cuanto más determinista, menos espacio para que Antigravity improvise una dirección
   equivocada (ver el caso de H5 en `docs/FIXES_AUDITORIA_SOBREAJUSTE.md` §del encabezado).
5. **Qué NO hacer** — alcance explícitamente fuera de esta vuelta, y por qué (para que no se disperse
   ni reabra algo ya cerrado en una vuelta anterior).
6. **Entregable esperado** — nombre de archivo exacto para su informe, y qué debe contener.
7. **Verificación** — cómo se sabe que el trabajo quedó bien hecho, sin necesitar los datos que esta
   sesión no tiene si es posible (harness sintético, tests con datos de juguete, etc.).

El brief se escribe con `Write`, no con `Edit` sobre uno viejo del mismo tema — cada vuelta del ciclo
es un archivo nuevo (`docs/<TEMA>.md`, luego `docs/FIXES_<TEMA>.md` si hace falta una segunda vuelta),
para que quede rastro de qué se pidió en cada iteración.

---

## Fase 2 — Revisar el informe (`revisar`)

### Regla central: ningún número se cita sin re-derivarlo

No se lee el informe y se asiente. Por cada cifra que el informe presenta como evidencia fuerte
(la que probablemente se va a citar en otro documento o va a cambiar una decisión), se recalcula
desde la fuente — el mismo artefacto (`.json`/`.csv` en `runs/`), el mismo grep sobre el código, el
mismo manifest — **sin mirar el número que dio Antigravity primero**. Si coincide, se cita con
confianza. Si no, se investiga por qué antes de asumir que uno de los dos está mal.

No hace falta re-derivar todo — prioriza lo que es: (a) sorprendente, (b) lo que va a terminar en
`CLAUDE.md`/`.claude/context/` como hecho establecido, o (c) la base de una recomendación de cambiar
código.

### Qué buscar específicamente

- **Citas de `archivo:línea` que no dicen lo que el informe afirma.** Ábrelas. Es el error más barato
  de cometer (leer una línea vecina, o una entrada de un dict-registro y confundirla con una
  instanciación) y el más caro de dejar pasar si otro documento la hereda sin verificar.
- **La dirección de un fix propuesto, contra el propio código.** Antes de aceptar "cambiar A a B",
  busca si A tiene un comentario que explique por qué es A y no B. Si lo tiene, el fix probablemente
  va al revés — corrige la dirección, no solo el destino. Grep amplio de dónde más aparece ese mismo
  patrón en el código (puede haber un tercer sitio que ninguno de los dos vio).
  Con `extra="forbid"` en cada modelo Pydantic del repo (`src/config.py`), la única forma en que una
  recomendación rompe algo en silencio es tocando un campo que sí existe pero con el valor incorrecto
  — no hay "campo ignorado" que la esconda.
- **Conteos y agregaciones no verificados en su propio código.** Si el informe agrupa por
  `(patient_id, laterality)` o similar y da un número, reprodúcelo con pandas sobre el mismo manifest.
  Un join mal hecho (ej. `predictions.csv` sin `patient_id`, unido por orden de fila contra el
  manifest) produce un resultado plausible y falso — es exactamente el tipo de error que un fact-check
  superficial no atrapa.
- **Alcance excedido o insuficiente.** ¿Tocó algo que el brief marcó como "no hacer"? ¿Dejó pendiente
  algo que el brief marcó como entregable obligatorio?
- **Afirmaciones sin verificación posible en las condiciones dadas.** Si el brief decía "sin datos
  reales" y el informe reporta un número que solo se puede obtener con datos reales, algo no cuadra —
  o hay acceso que el brief no anticipó, o el número es inventado/extrapolado sin decirlo.

### Clasificación de cada hallazgo del informe (no mezclar categorías)

- **Confirmado, verificado independientemente** — se re-derivó y coincide.
- **Confirmado, no re-verificado** — se le cree por coherencia interna pero no se comprobó a mano;
  decirlo así en vez de darle el mismo peso que lo anterior.
- **Refutado** — la re-derivación da otro resultado. Documentar ambos números y cuál es el correcto.
- **Bug de citación** — el hallazgo de fondo puede ser válido pero la evidencia (`archivo:línea`) que
  lo respalda no dice lo que afirma. Corregir la cita, no descartar el hallazgo solo por eso.
- Y, sobre cada hallazgo real: **bug** vs **decisión de diseño discutible** vs **problema
  metodológico** — la misma separación de tres vías que ya usa `docs/AUDITORIA_SOBREAJUSTE.md`. No
  llamar "bug" a una decisión documentada solo porque produce un número feo.

### Entregable de la revisión

No es solo una respuesta en el chat. Según lo que encuentre la revisión:

- **Si el informe aguanta la verificación** (como pasó con H2, H4, H5 de la auditoría de
  sobreajuste): actualizar directamente los documentos que dependen de esa conclusión —
  `.claude/context/experiments/*.md` si son resultados, `CLAUDE.md` si es una regla operativa, el
  `DOCS.md` correspondiente si es un contrato de método. Con `--aplicar`, hacerlo sin preguntar; sin
  el flag, proponer el diff y esperar confirmación si el cambio es a un documento que otros ya
  citan.
- **Si hay trabajo de seguimiento** (como pasó con las direcciones de fix de H5/H6.b): escribir
  `docs/FIXES_<TEMA>.md` con instrucciones corregidas y tan deterministas como el brief original —
  este es el `tarea` de la siguiente vuelta del ciclo.
- **Si se encontraron refutaciones**: decirlas explícitamente al usuario en la respuesta, con el
  número/cita que las contradice. Corregir un informe de Antigravity no es un fallo del ciclo, es el
  ciclo funcionando.

---

## Automatizar más el ciclo, si hace falta

Hoy el paso 5 (avisar que hay informe nuevo) es manual — el usuario dice "revisa esto". Si el volumen
de idas y vueltas lo justifica, dos formas de automatizarlo sin sobre-construir:

- **Polling ligero**: una tarea programada (`ScheduleWakeup`/`Monitor`) que compruebe cada cierto
  tiempo si apareció un archivo nuevo que matchee `docs/*_RESULTADOS.md` más reciente que el último
  `tarea` correspondiente, y dispare `/antigravity revisar` sola. Vale la pena solo si los ciclos se
  vuelven frecuentes — no montarlo para un uso ocasional.
- **Hook de filesystem** (`settings.json`, ver skill `update-config`): si Antigravity siempre escribe
  al mismo patrón de ruta, un hook que dispare sobre la creación/modificación de `docs/*_RESULTADOS.md`
  es más barato que un polling activo.

No implementar ninguna de las dos hasta que el ciclo manual (este comando) se haya usado un par de
veces más y quede claro cuál cadencia real tiene.
