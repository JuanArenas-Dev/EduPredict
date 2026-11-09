from pathlib import Path
import re
import unicodedata

import pandas as pd


REQUIRED_COLUMNS = [
    "Estudiante",
    "Materia",
    "Horas_Estudio",
    "Asistencia",
    "Tareas",
    "Nota_Final",
]

NUMERIC_COLUMNS = [
    "Horas_Estudio",
    "Asistencia",
    "Tareas",
    "Nota_Final",
]


ALIASES = {
    "estudiante": "Estudiante",
    "estudiante_id": "Estudiante",
    "id_estudiante": "Estudiante",
    "nombre_estudiante": "Estudiante",
    "student": "Estudiante",
    "student_id": "Estudiante",

    "materia": "Materia",
    "asignatura": "Materia",
    "subject": "Materia",

    "horas_estudio": "Horas_Estudio",
    "horas_de_estudio": "Horas_Estudio",
    "horas_estudiadas": "Horas_Estudio",
    "study_hours": "Horas_Estudio",
    "hours_studied": "Horas_Estudio",

    "asistencia": "Asistencia",
    "porcentaje_asistencia": "Asistencia",
    "asistencia_porcentaje": "Asistencia",
    "attendance": "Asistencia",
    "attendance_pct": "Asistencia",

    "tareas": "Tareas",
    "tareas_entregadas": "Tareas",
    "tareas_completadas": "Tareas",
    "assignments_completed": "Tareas",

    "nota_final": "Nota_Final",
    "calificacion_final": "Nota_Final",
    "calificacion": "Nota_Final",
    "nota": "Nota_Final",
    "final_grade": "Nota_Final",
    "grade": "Nota_Final",
}


def _normalizar_encabezado(nombre):
    """Normaliza un encabezado para reconocer mayúsculas, tildes y espacios."""
    texto = unicodedata.normalize("NFKD", str(nombre))
    texto = texto.encode("ascii", "ignore").decode("ascii")
    texto = texto.strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", texto).strip("_")


def normalizar_columnas(df):
    """Limpia encabezados y asigna nombres estándar cuando existe un alias."""
    df = df.copy()
    nuevas = []
    usadas = {}

    for columna in df.columns:
        clave = _normalizar_encabezado(columna)
        nombre = ALIASES.get(clave, str(columna).strip())

        if nombre in usadas:
            raise ValueError(
                f"Las columnas '{usadas[nombre]}' y '{columna}' "
                f"terminan representando el mismo campo: '{nombre}'. "
                "Deja solo una de ellas o cambia su encabezado."
            )

        usadas[nombre] = columna
        nuevas.append(nombre)

    df.columns = nuevas
    return df


def cargar_archivo(archivo):
    """
    Lee CSV, XLSX o XLS desde el objeto UploadedFile de Streamlit.
    Para CSV intenta UTF-8 y luego Latin-1.
    """
    extension = Path(archivo.name).suffix.lower()

    if extension == ".csv":
        errores = []
        for codificacion in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                archivo.seek(0)
                return pd.read_csv(
                    archivo,
                    sep=None,
                    engine="python",
                    encoding=codificacion,
                )
            except UnicodeDecodeError as exc:
                errores.append(str(exc))
            except pd.errors.ParserError:
                # Puede ocurrir si la detección automática del separador falla.
                archivo.seek(0)
                try:
                    return pd.read_csv(archivo, encoding=codificacion)
                except Exception as exc:
                    errores.append(str(exc))

        raise ValueError(
            "No fue posible interpretar el CSV. Revisa que el archivo "
            "esté separado por comas o punto y coma y tenga encabezados."
        )

    if extension == ".xlsx":
        archivo.seek(0)
        return pd.read_excel(archivo, engine="openpyxl")

    if extension == ".xls":
        archivo.seek(0)
        return pd.read_excel(archivo, engine="xlrd")

    raise ValueError("Formato no compatible. Usa un archivo CSV, XLSX o XLS.")


def validar_y_preparar(df):
    """
    Normaliza nombres, valida columnas y tipos, y agrega Estado/Riesgo.
    Devuelve: (DataFrame preparado, lista de errores, lista de advertencias).
    """
    errores = []
    advertencias = []

    if df is None or df.empty:
        return pd.DataFrame(), ["El archivo no contiene registros."], advertencias

    try:
        df = normalizar_columnas(df)
    except ValueError as exc:
        return pd.DataFrame(), [str(exc)], advertencias

    faltantes = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if faltantes:
        errores.append(
            "Faltan columnas obligatorias: " + ", ".join(faltantes) + "."
        )
        return df, errores, advertencias

    # Convertir campos numéricos. Los textos no convertibles pasan a NaN
    # y se reportan como advertencia para que el usuario decida cómo tratarlos.
    for columna in NUMERIC_COLUMNS:
        original = df[columna].copy()
        convertido = pd.to_numeric(original, errors="coerce")
        no_convertibles = original.notna() & convertido.isna()

        cantidad = int(no_convertibles.sum())
        if cantidad:
            advertencias.append(
                f"En '{columna}' hay {cantidad} valor(es) que no son numéricos. "
                "Se marcaron como nulos."
            )

        df[columna] = convertido

    # Validación de rangos definidos 
    rangos = {
        "Horas_Estudio": (0, None),
        "Asistencia": (0, 100),
        "Tareas": (0, None),
        "Nota_Final": (0, 5),
    }

    for columna, (minimo, maximo) in rangos.items():
        valores = df[columna].dropna()
        fuera = pd.Series(False, index=valores.index)

        if minimo is not None:
            fuera = fuera | (valores < minimo)
        if maximo is not None:
            fuera = fuera | (valores > maximo)

        cantidad = int(fuera.sum())
        if cantidad:
            if maximo is None:
                detalle = f"debe ser mayor o igual que {minimo}"
            else:
                detalle = f"debe estar entre {minimo} y {maximo}"

            errores.append(
                f"'{columna}' contiene {cantidad} valor(es) fuera del rango: "
                f"{detalle}."
            )

    # Limpiar espacios sin convertir los valores ausentes en el texto "nan".
    for columna in ("Estudiante", "Materia"):
        df[columna] = df[columna].astype("string").str.strip()
        df[columna] = df[columna].replace("", pd.NA)

    # Estado se calcula a partir de Nota_Final para mantener un criterio uniforme.
    estado_venia_en_archivo = "Estado" in df.columns
    df["Estado"] = df["Nota_Final"].apply(
        lambda nota: (
            "Aprobado" if pd.notna(nota) and nota >= 3.0
            else "Reprobado" if pd.notna(nota)
            else pd.NA
        )
    )

    # Riesgo es una etiqueta orientativa para el análisis, no un diagnóstico.
    # Alto: nota < 3.0 o asistencia < 70.
    # Medio: nota < 3.5 o asistencia < 80.
    # Bajo: cumple los dos umbrales superiores.
    def calcular_riesgo(fila):
        nota = fila["Nota_Final"]
        asistencia = fila["Asistencia"]

        if pd.isna(nota) or pd.isna(asistencia):
            return pd.NA
        if nota < 3.0 or asistencia < 70:
            return "Alto"
        if nota < 3.5 or asistencia < 80:
            return "Medio"
        return "Bajo"

    riesgo_calculado = df.apply(calcular_riesgo, axis=1)

    if "Riesgo" not in df.columns:
        df["Riesgo"] = riesgo_calculado
        advertencias.append(
            "La columna 'Riesgo' no venía en el archivo y se generó con "
            "umbrales orientativos de nota y asistencia."
        )
    else:
        df["Riesgo"] = df["Riesgo"].astype("string").str.strip()
        df["Riesgo"] = df["Riesgo"].replace("", pd.NA)
        df["Riesgo"] = df["Riesgo"].fillna(riesgo_calculado)

    if estado_venia_en_archivo:
        advertencias.append(
            "La columna 'Estado' se recalculó usando Nota_Final >= 3.0 "
            "como criterio de aprobación."
        )
    else:
        advertencias.append(
            "La columna 'Estado' se generó usando Nota_Final >= 3.0 "
            "como criterio de aprobación."
        )

    return df, errores, advertencias


def analizar_calidad(df):
    """Genera métricas de nulos y duplicados para mostrar en la interfaz."""
    if df is None or df.empty:
        return {
            "por_columna": pd.DataFrame(
                columns=["Columna", "Valores nulos", "Porcentaje nulos"]
            ),
            "filas_con_nulos": 0,
            "duplicados_exactos": 0,
            "duplicados_estudiante_materia": 0,
        }

    nulos = df.isna().sum()
    porcentaje = (nulos / len(df) * 100).round(2)

    por_columna = pd.DataFrame({
        "Columna": nulos.index,
        "Valores nulos": nulos.values,
        "Porcentaje nulos": porcentaje.values,
    })

    duplicados_clave = 0
    if {"Estudiante", "Materia"}.issubset(df.columns):
        duplicados_clave = int(
            df.duplicated(subset=["Estudiante", "Materia"]).sum()
        )

    return {
        "por_columna": por_columna,
        "filas_con_nulos": int(df.isna().any(axis=1).sum()),
        "duplicados_exactos": int(df.duplicated().sum()),
        "duplicados_estudiante_materia": duplicados_clave,
    }


def limpiar_datos(df, eliminar_nulos=False, eliminar_duplicados=False):
    """
    Aplica únicamente las limpiezas que el usuario selecciona en la interfaz.
    Los nulos se evalúan sobre las columnas obligatorias.
    """
    resultado = df.copy()

    if eliminar_nulos:
        resultado = resultado.dropna(subset=REQUIRED_COLUMNS)

    if eliminar_duplicados:
        resultado = resultado.drop_duplicates()

    return resultado.reset_index(drop=True)
