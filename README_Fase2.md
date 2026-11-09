# EduPredict — Fase 2: carga y validación de datos

Este paquete contiene una implementación de la Fase 2 para el proyecto EduPredict.

## Contenido

- `app.py`: aplicación Streamlit con las páginas Dashboard, Carga de datos,
  Análisis académico, Predicciones y Reportes.
- `modules/data_loader.py`: lectura de CSV/Excel, normalización de encabezados,
  validación, análisis de nulos/duplicados y limpieza opcional.
- `requirements.txt`: dependencias del proyecto.

## Instalación en Windows / PowerShell

Desde la carpeta principal `EduPredict`:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Si todavía no existe la carpeta `modules`:

```powershell
New-Item -ItemType Directory -Force modules
```

Copia la carpeta `modules` y el archivo `app.py` de este paquete a la raíz
de tu proyecto. Antes de reemplazar tu `app.py` actual, guarda una copia.

## Columnas obligatorias

| Columna | Tipo esperado | Regla |
|---|---|---|
| `Estudiante` | Texto | Identificador o nombre del estudiante |
| `Materia` | Texto | Asignatura |
| `Horas_Estudio` | Número | Mayor o igual que 0 |
| `Asistencia` | Número | Entre 0 y 100 |
| `Tareas` | Número | Mayor o igual que 0 |
| `Nota_Final` | Número | Entre 0 y 5 |

Se reconocen algunos encabezados equivalentes, por ejemplo `Asignatura` para
`Materia`, `Horas de estudio` para `Horas_Estudio` y `Calificacion final`
para `Nota_Final`.

La aplicación lee la primera hoja de los archivos Excel.

## Qué hace la aplicación

1. Permite seleccionar CSV, XLSX o XLS.
2. Revisa que estén las seis columnas obligatorias.
3. Convierte las columnas numéricas y reporta textos que no se puedan convertir.
4. Valida los rangos de asistencia y nota.
5. Muestra nulos por columna, filas con nulos y duplicados.
6. Presenta una vista previa de los primeros 20 registros.
7. Permite elegir si se eliminan filas con nulos en columnas obligatorias
   y/o filas completamente duplicadas.
8. Solo después de pulsar **Usar estos datos en el dashboard** reemplaza
   la fuente simulada por el archivo cargado.

## Criterios académicos utilizados

- Aprobado: `Nota_Final >= 3.0`.
- Reprobado: `Nota_Final < 3.0`.
- Riesgo alto: nota menor que 3.0 o asistencia menor que 70%.
- Riesgo medio: nota menor que 3.5 o asistencia menor que 80%, siempre que
  no cumpla la condición de riesgo alto.
- Riesgo bajo: los demás casos.

La etiqueta de riesgo es una regla inicial para análisis exploratorio.
Más adelante se puede comparar y reemplazar por un modelo entrenado.

## Importante

La detección de registros repetidos por combinación `Estudiante + Materia`
se muestra como advertencia, pero no se eliminan automáticamente: podrían
ser registros legítimos de distintos periodos o intentos. La limpieza de
duplicados disponible elimina únicamente filas idénticas en todas sus columnas.

Los datos simulados siguen disponibles hasta que se cargue y aplique un archivo.
