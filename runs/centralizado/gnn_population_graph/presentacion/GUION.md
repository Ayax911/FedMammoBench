# Guion — "Grafos de población para mamografía" (exposición virtual, ~15 min)

El texto completo para decir en cada diapositiva está en sus **notas del orador** (modo Present).
Esta hoja es el mapa de la charla para tener a la vista.

## Flujo

| # | Diapositiva | Tiempo | Idea que debe quedar |
|---|---|---|---|
| 1 | Portada | 0:30 | La pregunta: ¿el contexto de otras imágenes ayuda a clasificar? |
| 2 | Agenda | 0:30 | Cinco partes; las preguntas van al chat y se responden al final |
| 3 | El problema | 1:00 | Una CNN mira cada imagen sola; un radiólogo combina CC y MLO |
| 4 | Datos | 0:45 | 8.341 imágenes, 4 bases, splits separados por paciente |
| 5 | ¿Cómo funciona una CNN? | 1:00 | Filtros → patrones → **un vector por imagen** |
| 6 | ¿Cómo funciona una GNN? | 1:15 | Paso de mensajes: **promediar con los vecinos** (si los vecinos mienten, contamina) |
| 7 | Pipeline | 1:00 | CNN describe, GNN conecta; transductivo, sin fuga de etiquetas |
| 8 | Cinco grafos + homofilia | 1:30 | Similitud ≈ azar (0,65); misma mama = 0,99. **Pregunta al chat** |
| 9 | Protocolo | 0:45 | El control clave es "MLP + promedio por mama" |
| 10 | Resultados (escalera) | 1:30 | Grafo malo baja el AUC (0,799); grafo bueno lo sube (0,857) |
| 11 | El hallazgo | 1:15 | **El grafo importa más que la arquitectura** (hablar lento) |
| 12 | ROC | 0:45 | Ganancia en la zona clínica de 10–30 % de falsos positivos |
| 13 | Sensibilidad / especificidad | 1:00 | 77 % / 77 %; el grafo malo pierde 13 puntos de detección |
| 14 | Subgrafo | 0:45 | CC y MLO no se parecen, pero comparten diagnóstico |
| 15 | Lectura honesta | 1:00 | La GNN iguala al promedio manual; no hace magia |
| 16 | Conclusiones | 1:00 | Tres ideas + próximos pasos (GNN federada) |
| 17 | ¿Preguntas? | — | Respuestas preparadas en las notas |
| 18 | Anexo | — | Solo si preguntan por GCN vs. SAGE vs. GAT |

Si vas corto de tiempo, salta la 12 (ROC): la 13 cuenta lo mismo en lenguaje clínico.

## Checklist virtual

- Ensaya una vez con cronómetro; apunta a 14 minutos para dejar margen.
- Comparte **solo la ventana** del navegador en modo Present, no todo el escritorio.
- Las notas, en otra pantalla o en el celular.
- Silencia notificaciones (correo, Slack, WhatsApp Web).
- Cámara encendida, a la altura de los ojos; luz de frente.
- Señala con palabras, no con el mouse: "la barra naranja", "los puntos de la izquierda".
- Haz pausas de 2–3 segundos tras las diapositivas 8 y 11: en virtual el silencio se siente más largo, pero el público lo necesita.
- Lee en voz alta cada pregunta del chat antes de responderla.
- Plan B: exporta la presentación a PDF (Share › Export) por si falla la conexión.

## Las tres frases para recordar

1. "La GNN promedia con sus vecinos: si las aristas mienten, el promedio contamina."
2. "La homofilia se mide antes de entrenar; es un chequeo barato."
3. "Una arista con conocimiento clínico vale más que mil aristas por similitud."
