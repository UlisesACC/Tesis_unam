# Proyecto de tesis en LaTeX

El archivo principal es `main.tex`. Los datos de portada se editan en `configuracion/datos.tex` y el contenido se distribuye entre `preliminares`, `capitulos` y `apendices`.

## Compilación local

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

Si no se dispone de `latexmk`:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Overleaf

Sube el archivo ZIP como proyecto nuevo. Overleaf detectará `main.tex` automáticamente. Conserva los nombres de carpetas y evita espacios en los nombres de archivos.

## Organización

- `configuracion`: paquetes, estilos y datos de portada.
- `preliminares`: portada, resumen, abstract y abreviaturas.
- `capitulos`: contenido principal de la tesis.
- `bibliografia`: referencias en formato BibLaTeX.
- `figuras`: imágenes usadas en el documento.
- `tablas`: tablas externas, si se requieren.
- `codigo`: programas incorporados mediante `listings`.
- `apendices`: material complementario.
