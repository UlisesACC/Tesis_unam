# Tesis UNAM

Compilar `main.tex` desde la raíz con `latexmk -pdf main.tex` y Biber para la bibliografía.

## Organización

- `configuracion/`: paquetes y portada.
- `preliminares/`: portada, resúmenes y abreviaturas.
- `capitulos/01_introduccion/` a `capitulos/06_conclusiones/`: cada `index.tex` incorpora secciones mediante `\input`.
- `capitulos/02_marco_teorico/fundamentos/`: series de tiempo y modelos.
- `capitulos/03_estado_del_arte/articulos/`: reseñas de artículos.
- `capitulos/04_metodologia/`: procedimiento y archivos por experimento.
- `capitulos/05_resultados/`: evidencia, análisis y evaluación pendiente.
- `figuras/05_resultados/`: copias de las figuras usadas por el documento.
- `apendices/`: una carpeta e índice por apéndice.
- `bibliografia/`: referencias.
- `codigos de docker/`: programas, datos y salidas originales en sus rutas originales.

Todas las rutas de los inputs se resuelven desde la raíz. Para añadir una sección, crear su archivo y registrarlo en el índice del capítulo.

## OLS actual

El modelo recibe 24 generaciones pasadas y predice la siguiente, con 24 ejemplos por ajuste móvil. Las fechas no son predictores. Su evaluación sobre datos reales está pendiente; la figura OLS con calendario se conserva como histórica.
