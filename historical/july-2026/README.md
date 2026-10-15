# Redundancia Visual y Susceptibilidad al *Leakage* en Fitopatología

Estudio de doble dominio (laboratorio vs. campo) sobre la **fuga de datos** (*data leakage*) por
redundancia visual en la clasificación de enfermedades foliares de papa (*Solanum tuberosum*), con dos
aportes metodológicos: un protocolo de partición **VC-GKF** basado en embeddings **DINOv2** y el
**Índice de Sensibilidad al Leakage (LSI)**.

**Autores:** Andrés Abraham Guerrero Jaramillo, Hugo Froilán Vega Huerta
**Afiliación:** Universidad Nacional Mayor de San Marcos (UNMSM), Facultad de Ingeniería de Sistemas e
Informática — Curso de Inteligencia Artificial, 2026.

---

## Resumen

El aprendizaje profundo reporta exactitudes cercanas al 100 % en fitopatología, pero esas cifras suelen
obtenerse con particiones aleatorias que permiten la fuga de datos por imágenes casi duplicadas. Este
trabajo:

1. **Cuantifica la redundancia visual** con clustering aglomerativo sobre embeddings auto-supervisados
   DINOv2 (sin la circularidad de usar el *backbone* del clasificador), en dos dominios de papa con las
   mismas tres clases: PlantVillage (laboratorio) y un conjunto de campo (Shabrina + Sankalana).
2. Propone el **Índice de Sensibilidad al Leakage (LSI)**, que mide cuánta exactitud es atribuible a la
   redundancia.
3. Halla que la redundancia es **masiva en laboratorio** (1 % de imágenes singleton; un clúster agrupa el
   44 % del dataset) y modesta en campo, pero que su impacto en la exactitud es **pequeño** para tareas
   fáciles, y que la **susceptibilidad al leakage depende de la arquitectura**: ResNet50 y ViT-B/16 se
   inflan ~8× más que MobileNetV3 y EfficientNetV2-S.

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `articulo_revisado.tex` | Fuente LaTeX del artículo (IEEEtran). |
| `referencias.bib` | Bibliografía (36 referencias, todas citadas). |
| `papaFinal.pdf` | Artículo compilado. |
| `experimentos.ipynb` | Notebook de reproducibilidad (datos, DINOv2, clustering, LSI, grilla de entrenamiento). |
| `resultados.zip` | Resultados experimentales (CSVs por dominio, modelo y semilla). |

## Cómo compilar el artículo

Requiere una distribución LaTeX con `IEEEtran` y `babel-spanish` (o usa [Overleaf](https://www.overleaf.com)).

```bash
pdflatex articulo_revisado.tex
bibtex   paper2_articulo
pdflatex articulo_revisado.tex
pdflatex articulo_revisado.tex
```

En Overleaf: sube `articulo_revisado.tex` y `referencias.bib`, y compila. No requiere imágenes externas
(la única figura se genera con TikZ).

## Cómo reproducir los experimentos

El notebook está pensado para **Kaggle con GPU** (o Colab GPU):

1. Sube `experimentos.ipynb` a Kaggle. Activa **GPU (T4)** e **Internet**.
2. Adjunta como *Input* los datasets: `arjuntejaswi/plant-village`,
   `warcoder/potato-leaf-disease-dataset` y `nirmalsankalana/potato-leaf-healthy-and-late-blight`.
3. Ajusta `RUN_DOMAINS` (p. ej. `["potato_lab"]` o `["potato"]`) y ejecuta con **Save & Run All (Commit)**.
4. Los resultados quedan en `RESULTS/` (los CSVs incluidos en `resultados.zip` son la salida de estas
   corridas).

## Datasets

- **PlantVillage** (papa, laboratorio) — Hughes & Salathé, 2015.
- **Potato Leaf Disease in Uncontrolled Environment** (campo) — Shabrina et al., 2024.
- **Potato Leaf Healthy and Late Blight** (sanas de campo, adicional) — Sankalana (Kaggle).

Todos de acceso público en Kaggle.

## Cita

Si usas este trabajo, cita el artículo (`papaFinal.pdf`) de Guerrero Jaramillo y Vega Huerta (UNMSM, 2026).

## Licencia

Proyecto académico del curso de Inteligencia Artificial (UNMSM). Uso educativo.
