import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_loader import (
    REQUIRED_COLUMNS,
    analizar_calidad,
    cargar_archivo,
    limpiar_datos,
    validar_y_preparar,
)

from modules.model_training import (
    train_models,
    predict_student,
    get_feature_importance
)

from io import BytesIO

st.set_page_config(
    page_title="EduPredict | Minería de datos",
    page_icon="🎓",
    layout="wide",
)



EDUPREDICT_CSS = """
<style>
:root { color-scheme: dark; }
.stApp {background: radial-gradient(ellipse at 85% 0%, #142644 0%, #0b1220 42%, #090f1b 100%); color:#e5edf8;}
[data-testid="stHeader"] {background:rgba(11,18,32,.88);}
[data-testid="stSidebar"] {background:#0e192b; border-right:1px solid #223550;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#a9bad3;}
[data-testid="stSidebar"] div[role="radiogroup"] label {border-radius:10px; padding:9px 12px; margin:3px 0; transition:background .15s;}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background:#1b304d;}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {background:#183a59; border:1px solid #27618b;}
.block-container {padding-top:2rem; max-width:1450px;}
h1,h2,h3 {letter-spacing:-.035em; color:#f4f8ff !important;}
h1 {font-weight:750 !important;} h3 {font-weight:650 !important;}
[data-testid="stCaptionContainer"] {color:#95a9c4;}
[data-testid="stMetric"] {background:linear-gradient(155deg,#17263b,#111c2e);border:1px solid #263b57;border-radius:15px;padding:18px 19px;min-height:118px;box-shadow:0 8px 22px #050a1330;}
[data-testid="stMetricLabel"] {color:#9cb0ca;}
[data-testid="stMetricValue"] {color:#f3f8ff;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#263b57;}
[data-testid="stDataFrame"], [data-testid="stFileUploader"], [data-testid="stForm"] {border-radius:13px;overflow:hidden;}
[data-testid="stForm"] {background:#121e30; border:1px solid #263b57; padding:18px;}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {background:linear-gradient(100deg,#1477bd,#2752b9);color:#fff;border:1px solid #358fe3;border-radius:10px;font-weight:650;}
.stButton > button, .stDownloadButton > button {border-radius:10px;min-height:41px;}
.stButton > button:hover, .stDownloadButton > button:hover {border-color:#38bdf8;}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {border-radius:10px;}
[data-testid="stAlert"] {border-radius:12px;}
hr {border-color:#253852;}
::selection {background:#245a87;color:#fff;}
@media(max-width:700px){.block-container{padding:1.1rem .8rem;}[data-testid="stMetric"]{min-height:100px;padding:12px;}}
</style>
"""

PLOTLY_COLORS = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#fb7185", "#60a5fa"]


def aplicar_estilo():
    st.markdown(EDUPREDICT_CSS, unsafe_allow_html=True)


def grafico_moderno(figura, **kwargs):
    """Presenta figuras existentes sin modificar sus datos ni cálculos."""
    figura.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d6ea", family="Arial"),
        title=dict(font=dict(size=17, color="#f4f8ff"), x=0.02),
        margin=dict(l=24, r=22, t=64, b=34),
        colorway=PLOTLY_COLORS,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    figura.update_xaxes(gridcolor="#253750", zerolinecolor="#253750")
    figura.update_yaxes(gridcolor="#253750", zerolinecolor="#253750")
    st.plotly_chart(figura, **kwargs)


def generar_datos_simulados(cantidad=300):
    """Datos de demostración."""
    rng = np.random.default_rng(42)

    materias = [
        "Matemáticas",
        "Programación",
        "Bases de Datos",
        "Inglés",
        "Física",
    ]

    estudiantes = [f"Estudiante {i:03d}" for i in range(1, cantidad + 1)]
    materia = rng.choice(materias, size=cantidad)
    horas_estudio = np.clip(rng.normal(8, 3, cantidad), 0, 20).round(1)
    asistencia = np.clip(rng.normal(82, 12, cantidad), 35, 100).round(1)
    tareas = rng.integers(1, 11, cantidad)

    nota = (
        1.2
        + horas_estudio * 0.16
        + asistencia * 0.012
        + tareas * 0.075
        + rng.normal(0, 0.45, cantidad)
    )
    nota = np.clip(nota, 0, 5).round(1)

    df = pd.DataFrame({
        "Estudiante": estudiantes,
        "Materia": materia,
        "Horas_Estudio": horas_estudio,
        "Asistencia": asistencia,
        "Tareas": tareas,
        "Nota_Final": nota,
    })

    df["Estado"] = np.where(df["Nota_Final"] >= 3.0, "Aprobado", "Reprobado")
    df["Riesgo"] = np.select(
        [
            (df["Nota_Final"] < 3.0) | (df["Asistencia"] < 70),
            (df["Nota_Final"] < 3.5) | (df["Asistencia"] < 80),
        ],
        ["Alto", "Medio"],
        default="Bajo",
    )

    return df


def obtener_datos_activos():
    """Prioriza el archivo cargado; si no existe, utiliza los datos simulados."""
    datos = st.session_state.get("edupredict_datos_cargados")

    if isinstance(datos, pd.DataFrame) and not datos.empty:
        return datos.copy(), "Archivo cargado"

    return generar_datos_simulados(), "Datos simulados"


def mostrar_encabezado_pagina(titulo, descripcion=None):
    st.title(titulo)
    if descripcion:
        st.caption(descripcion)


def pagina_carga():
    mostrar_encabezado_pagina(
        "Carga y validación de datos",
        "Importa un archivo académico, revisa su estructura y decide qué limpieza aplicar.",
    )

    st.info(
        "Columnas obligatorias: Estudiante, Materia, Horas_Estudio, "
        "Asistencia, Tareas y Nota_Final. La nota debe estar en escala de 0.0 a 5.0."
    )

    archivo = st.file_uploader(
        "Selecciona un archivo CSV o Excel",
        type=["csv", "xlsx", "xls"],
        help="Para Excel se lee la primera hoja del libro.",
    )

    datos_activos = st.session_state.get("edupredict_datos_cargados")
    if isinstance(datos_activos, pd.DataFrame) and not datos_activos.empty:
        st.success(
            f"Actualmente el dashboard está utilizando un archivo cargado "
            f"con {len(datos_activos):,} registros."
        )
        if st.button("Volver a los datos simulados", key="restaurar_simulados"):
            st.session_state.pop("edupredict_datos_cargados", None)
            st.rerun()

    if archivo is None:
        st.markdown(
            """
            **Todavía no has seleccionado un archivo.**

            Puedes continuar usando el dashboard con los datos simulados.
            Cuando cargues un archivo, EduPredict mostrará su vista previa
            y un informe básico de calidad antes de incorporarlo.
            """
        )
        return

    try:
        datos_originales = cargar_archivo(archivo)
        datos, errores, advertencias = validar_y_preparar(datos_originales)
    except Exception as exc:
        st.error(f"No fue posible leer el archivo: {exc}")
        return

    st.subheader("Estructura del archivo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", f"{len(datos_originales):,}")
    c2.metric("Columnas", f"{len(datos_originales.columns):,}")
    c3.metric("Archivo", archivo.name)

    st.write("Encabezados detectados:")
    st.code(", ".join(str(c) for c in datos_originales.columns))

    if errores:
        st.error("El archivo requiere correcciones antes de cargarlo:")
        for error in errores:
            st.write(f"- {error}")
        st.write(
            "Revisa los encabezados y los rangos indicados. "
            "No se han sustituido los datos activos del dashboard."
        )
        return

    for advertencia in advertencias:
        st.warning(advertencia)

    st.subheader("Calidad de los datos")
    calidad = analizar_calidad(datos)

    m1, m2, m3 = st.columns(3)
    m1.metric("Filas con al menos un nulo", calidad["filas_con_nulos"])
    m2.metric("Filas duplicadas exactas", calidad["duplicados_exactos"])
    m3.metric(
        "Repeticiones Estudiante + Materia",
        calidad["duplicados_estudiante_materia"],
    )

    st.caption(
        "Las repeticiones de Estudiante + Materia se reportan por separado: "
        "no se eliminan automáticamente porque podrían representar intentos "
        "o registros legítimos. La opción de duplicados elimina filas idénticas."
    )

    st.dataframe(
        calidad["por_columna"],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Vista previa")
    st.dataframe(datos.head(20), use_container_width=True, hide_index=True)

    st.subheader("Preparación de los datos")
    st.write(
        "Selecciona cómo tratar los registros incompletos y las filas "
        "exactamente duplicadas. Por defecto, EduPredict conserva los datos "
        "originales y no elimina filas automáticamente."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        eliminar_nulos = st.checkbox(
            "Eliminar filas con nulos en columnas obligatorias",
            value=False,
            help=(
                "Elimina filas que tengan algún valor ausente en las seis "
                "columnas obligatorias."
            ),
        )
    with col_b:
        eliminar_duplicados = st.checkbox(
            "Eliminar filas completamente duplicadas",
            value=False,
            help="Solo elimina filas idénticas en todas sus columnas.",
        )

    datos_limpios = limpiar_datos(
        datos,
        eliminar_nulos=eliminar_nulos,
        eliminar_duplicados=eliminar_duplicados,
    )

    st.caption(
        f"Registros que se incorporarían al dashboard: "
        f"{len(datos_limpios):,} de {len(datos):,}."
    )

    if datos_limpios.empty:
        st.error("Con las opciones seleccionadas no quedan registros para cargar.")
    else:
        if st.button(
            "Usar estos datos en el dashboard",
            type="primary",
            key="aplicar_datos_cargados",
        ):
            st.session_state["edupredict_datos_cargados"] = datos_limpios
            st.success(
                "Los datos quedaron seleccionados como fuente del dashboard. "
                "Abre la página Dashboard para consultarlos."
            )
            st.rerun()


def pagina_dashboard(df, fuente):
    mostrar_encabezado_pagina(
        "Dashboard académico",
        f"Fuente de información: {fuente}.",
    )

    if df.empty:
        st.warning("No hay registros para mostrar.")
        return

    materias = sorted(df["Materia"].dropna().astype(str).unique().tolist())
    opciones = ["Todas"] + materias
    materia_seleccionada = st.selectbox(
        "Filtrar por materia",
        opciones,
        key="filtro_materia_dashboard",
    )

    filtrado = df.copy()
    if materia_seleccionada != "Todas":
        filtrado = filtrado[
            filtrado["Materia"].astype("string") == materia_seleccionada
        ]

    notas_validas = filtrado["Nota_Final"].dropna()
    promedio = notas_validas.mean() if not notas_validas.empty else np.nan
    porcentaje_aprobacion = (
        (notas_validas >= 3.0).mean() * 100
        if not notas_validas.empty
        else np.nan
    )
    porcentaje_reprobacion = (
        (notas_validas < 3.0).mean() * 100
        if not notas_validas.empty
        else np.nan
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registros", f"{len(filtrado):,}")
    k2.metric("Estudiantes únicos", f"{filtrado['Estudiante'].nunique():,}")
    k3.metric(
        "Promedio de nota",
        f"{promedio:.2f}/5.0" if pd.notna(promedio) else "Sin datos",
    )
    k4.metric(
        "Aprobación",
        f"{porcentaje_aprobacion:.1f}%"
        if pd.notna(porcentaje_aprobacion)
        else "Sin datos",
    )

    if pd.notna(porcentaje_reprobacion):
        st.caption(f"Porcentaje de reprobación: {porcentaje_reprobacion:.1f}%")

    izquierda, derecha = st.columns(2)

    with izquierda:
        st.subheader("Estado académico")
        conteo_estado = (
            filtrado["Estado"]
            .dropna()
            .value_counts()
            .rename_axis("Estado")
            .reset_index(name="Cantidad")
        )
        if not conteo_estado.empty:
            figura = px.pie(
                conteo_estado,
                names="Estado",
                values="Cantidad",
                hole=0.4,
                title="Distribución de aprobados y reprobados",
            )
            grafico_moderno(figura, use_container_width=True)
        else:
            st.info("No hay notas válidas para calcular el estado académico.")

    with derecha:
        st.subheader("Distribución de notas")
        if not notas_validas.empty:
            figura = px.histogram(
                filtrado.dropna(subset=["Nota_Final"]),
                x="Nota_Final",
                nbins=15,
                title="Frecuencia de notas finales",
                labels={"Nota_Final": "Nota final"},
            )
            figura.update_xaxes(range=[0, 5])
            grafico_moderno(figura, use_container_width=True)
        else:
            st.info("No hay notas válidas para construir el histograma.")

    st.subheader("Asistencia frente a nota final")
    datos_dispersion = filtrado.dropna(subset=["Asistencia", "Nota_Final"])
    if not datos_dispersion.empty:
        figura = px.scatter(
            datos_dispersion,
            x="Asistencia",
            y="Nota_Final",
            color="Estado",
            hover_data=["Estudiante", "Materia"],
            title="Relación entre asistencia y rendimiento",
            labels={
                "Asistencia": "Asistencia (%)",
                "Nota_Final": "Nota final",
            },
        )
        figura.update_yaxes(range=[0, 5])
        grafico_moderno(figura, use_container_width=True)
    else:
        st.info("No hay suficientes datos válidos de asistencia y nota.")

    st.subheader("Registros académicos")
    st.dataframe(filtrado, use_container_width=True, hide_index=True)


def pagina_analisis(df, fuente):
    mostrar_encabezado_pagina(
        "Análisis académico",
        f"Resumen descriptivo de la fuente: {fuente}.",
    )

    if df.empty:
        st.warning("No hay registros para analizar.")
        return

    resumen = (
        df.groupby("Materia", dropna=False)
        .agg(
            Registros=("Estudiante", "size"),
            Estudiantes=("Estudiante", "nunique"),
            Promedio=("Nota_Final", "mean"),
            Asistencia_promedio=("Asistencia", "mean"),
            Horas_estudio_promedio=("Horas_Estudio", "mean"),
        )
        .reset_index()
    )

    notas_por_materia = df.dropna(subset=["Nota_Final"]).copy()
    if not notas_por_materia.empty:
        notas_por_materia["Aprobado"] = notas_por_materia["Nota_Final"] >= 3.0
        aprobacion = (
            notas_por_materia.groupby("Materia", dropna=False)["Aprobado"]
            .mean()
            * 100
        )
        resumen["Aprobacion_pct"] = resumen["Materia"].map(aprobacion)
    else:
        resumen["Aprobacion_pct"] = np.nan

    st.dataframe(
        resumen.round(2),
        use_container_width=True,
        hide_index=True,
    )

    datos_grafico = resumen.dropna(subset=["Promedio"])
    if not datos_grafico.empty:
        figura = px.bar(
            datos_grafico,
            x="Materia",
            y="Promedio",
            color="Materia",
            title="Promedio de nota por materia",
            labels={"Promedio": "Nota promedio"},
        )
        figura.update_yaxes(range=[0, 5])
        grafico_moderno(figura, use_container_width=True)


def pagina_predicciones(df, fuente):
    mostrar_encabezado_pagina(
        "Entrenamiento y predicciones",
        f"Modelos de clasificación del rendimiento académico. Fuente: {fuente}.",
    )

    st.write(
        "EduPredict entrenará dos modelos para clasificar el resultado académico "
        "como Aprobado o Reprobado. La nota final se utiliza únicamente para "
        "construir la etiqueta durante el entrenamiento; no se utiliza como "
        "variable de entrada al realizar una predicción."
    )

    if df.empty:
        st.warning("No hay datos disponibles para entrenar los modelos.")
        return

    st.subheader("Preparación y entrenamiento")

    st.caption(
        "Variables de entrada: Materia, Horas de estudio, Asistencia y Tareas. "
        "La clasificación se construye con Nota_Final (nota >= 3.0: Aprobado)."
    )

    if st.button("Entrenar modelos", type="primary", key="entrenar_modelos"):
        try:
            with st.spinner("Entrenando y evaluando los modelos..."):
                resultados = train_models(df)
            st.session_state["edupredict_modelos"] = resultados
            st.session_state["edupredict_modelos_fuente"] = fuente
            st.session_state["edupredict_modelos_filas"] = len(df)
            st.success("Los modelos se entrenaron y evaluaron correctamente.")
        except Exception as exc:
            st.session_state.pop("edupredict_modelos", None)
            st.error(f"No fue posible entrenar los modelos: {exc}")

    resultados = st.session_state.get("edupredict_modelos")

    # Evita presentar modelos entrenados con una fuente anterior.
    if resultados is not None and (
        st.session_state.get("edupredict_modelos_fuente") != fuente
        or st.session_state.get("edupredict_modelos_filas") != len(df)
    ):
        st.warning(
            "La fuente o la cantidad de registros cambió. Vuelve a entrenar "
            "los modelos para evaluar los datos actuales."
        )
        resultados = None

    if not resultados:
        st.info(
            "Cuando tengas los datos listos, selecciona «Entrenar modelos» "
            "para comenzar."
        )
        return

    st.subheader("Comparación de modelos")

    filas_metricas = []
    for nombre, resultado in resultados.items():
        fila = {"Modelo": nombre}
        fila.update(resultado["metrics"])
        filas_metricas.append(fila)

    tabla_metricas = pd.DataFrame(filas_metricas)
    st.dataframe(
        tabla_metricas.style.format({
            "Exactitud": "{:.2%}",
            "Precisión (Reprobado)": "{:.2%}",
            "Sensibilidad (Reprobado)": "{:.2%}",
            "F1 (Reprobado)": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "La sensibilidad (Recall) de Reprobado indica qué proporción de los "
        "estudiantes que realmente reprobaron fue identificada por el modelo."
    )

    nombres_modelos = list(resultados.keys())
    modelo_seleccionado = st.selectbox(
        "Selecciona un modelo para consultar su evaluación",
        nombres_modelos,
        key="modelo_evaluacion",
    )

    resultado = resultados[modelo_seleccionado]

    st.subheader("Matriz de confusión")

    matriz = resultado["confusion_matrix"]
    figura_matriz = px.imshow(
        matriz,
        text_auto=True,
        x=["Predicho: Aprobado", "Predicho: Reprobado"],
        y=["Real: Aprobado", "Real: Reprobado"],
        color_continuous_scale="Blues",
        aspect="auto",
        title=f"Matriz de confusión — {modelo_seleccionado}",
    )
    grafico_moderno(figura_matriz, use_container_width=True)
    
    st.subheader("Importancia de las variables")

    modelo_arbol = resultados["Árbol de Decisión"]["model"]

    try:
        importancia = get_feature_importance(modelo_arbol)

        st.write(
            "Este gráfico muestra cuánto contribuye cada variable "
            "a las decisiones del Árbol de Decisión."
        )

        figura_importancia = px.bar(
            importancia,
            x="Importancia",
            y="Variable",
            orientation="h",
            title="Importancia de las variables predictoras",
            labels={
                "Importancia": "Importancia relativa",
                "Variable": "Variable"
            }
        )

        figura_importancia.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )

        grafico_moderno(
            figura_importancia,
            use_container_width=True
        )

        st.dataframe(
            importancia.round(4),
            use_container_width=True,
            hide_index=True
        )

    except Exception as exc:
        st.warning(
            f"No fue posible calcular la importancia: {exc}"
        )


    st.subheader("Predicción de un estudiante")

    st.write(
        "Ingresa los datos de un estudiante. No necesitas proporcionar su "
        "nota final, porque precisamente queremos estimar su clasificación."
    )

    materias = sorted(
        df["Materia"].dropna().astype(str).unique().tolist()
    )

    if not materias:
        st.warning("No hay materias válidas para realizar predicciones.")
        return

    with st.form("formulario_prediccion_estudiante"):
        materia = st.selectbox("Materia", materias)
        col1, col2, col3 = st.columns(3)

        with col1:
            horas = st.number_input(
                "Horas de estudio",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5,
            )

        with col2:
            asistencia = st.number_input(
                "Asistencia (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0,
            )

        with col3:
            tareas = st.number_input(
                "Tareas entregadas",
                min_value=0,
                value=5,
                step=1,
            )

        ejecutar_prediccion = st.form_submit_button(
            "Predecir rendimiento",
            type="primary",
        )

    if ejecutar_prediccion:
        try:
            modelo = resultados[modelo_seleccionado]["model"]
            prediccion, probabilidades = predict_student(
                modelo,
                materia,
                horas,
                asistencia,
                tareas,
            )

            if prediccion == "Aprobado":
                st.success(f"Resultado estimado: {prediccion}")
            else:
                st.warning(f"Resultado estimado: {prediccion}")

            probabilidad = probabilidades.get(prediccion, 0.0)
            st.metric(
                "Probabilidad estimada de la clase predicha",
                f"{probabilidad:.1%}",
            )

            st.caption(
                "Esta estimación no garantiza el resultado académico. "
                "Debe interpretarse como apoyo orientativo y depende de "
                "la calidad y representatividad de los datos de entrenamiento."
            )

        except Exception as exc:
            st.error(f"No fue posible generar la predicción: {exc}")

    st.subheader("Modelos guardados")
    st.write(
        "Los pipelines entrenados se guardan en la carpeta `models`: "
        "`arbol_decision.joblib` y `regresion_logistica.joblib`."
    )



def pagina_reportes(df, fuente):
    mostrar_encabezado_pagina(
        "Reportes académicos",
        f"Informes y recomendaciones basados en: {fuente}.",
    )

    if df.empty:
        st.warning("No hay registros disponibles para generar reportes.")
        return

    datos = df.copy()

    # Asegurar que las columnas numéricas sean tratadas como números.
    for columna in [
        "Nota_Final",
        "Asistencia",
        "Horas_Estudio",
        "Tareas"
    ]:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        )

    notas = datos["Nota_Final"].dropna()

    total_registros = len(datos)
    total_estudiantes = datos["Estudiante"].nunique()
    promedio = notas.mean() if not notas.empty else 0

    aprobados = (notas >= 3.0).sum()
    reprobados = (notas < 3.0).sum()

    porcentaje_aprobacion = (
        aprobados / len(notas) * 100 if len(notas) else 0
    )

    porcentaje_reprobacion = (
        reprobados / len(notas) * 100 if len(notas) else 0
    )

    st.subheader("Indicadores generales")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Registros académicos", f"{total_registros:,}")
    c2.metric("Estudiantes únicos", f"{total_estudiantes:,}")
    c3.metric("Promedio académico", f"{promedio:.2f}/5.0")
    c4.metric("Aprobación", f"{porcentaje_aprobacion:.1f}%")

    c5, c6 = st.columns(2)
    c5.metric("Estudiantes aprobados", f"{aprobados:,}")
    c6.metric("Estudiantes reprobados", f"{reprobados:,}")

    st.subheader("Resumen por materia")

    resumen = (
        datos.groupby("Materia", dropna=False)
        .agg(
            Registros=("Estudiante", "size"),
            Estudiantes=("Estudiante", "nunique"),
            Promedio=("Nota_Final", "mean"),
            Asistencia_promedio=("Asistencia", "mean"),
            Horas_estudio_promedio=("Horas_Estudio", "mean"),
        )
        .reset_index()
    )

    aprobacion_materia = (
        datos.dropna(subset=["Nota_Final"])
        .assign(Aprobado=lambda x: x["Nota_Final"] >= 3.0)
        .groupby("Materia")["Aprobado"]
        .mean()
        .mul(100)
    )

    resumen["Aprobacion_pct"] = resumen["Materia"].map(
        aprobacion_materia
    )

    st.dataframe(
        resumen.round(2),
        use_container_width=True,
        hide_index=True
    )

    if not resumen.empty:
        figura = px.bar(
            resumen,
            x="Materia",
            y="Promedio",
            color="Materia",
            title="Promedio académico por materia",
            labels={"Promedio": "Promedio de notas"}
        )

        figura.update_yaxes(range=[0, 5])
        grafico_moderno(figura, use_container_width=True)

    st.subheader("Recomendaciones académicas")

    if porcentaje_reprobacion >= 30:
        st.warning(
            "Se observa un porcentaje de reprobación igual o superior "
            "al 30 %. Se recomienda implementar acompañamiento académico "
            "y seguimiento a los estudiantes con dificultades."
        )
    elif porcentaje_reprobacion > 0:
        st.info(
            "Aunque la mayoría de los registros corresponde a estudiantes "
            "aprobados, es conveniente hacer seguimiento a quienes presentan "
            "notas inferiores a 3.0."
        )
    else:
        st.success(
            "No se identifican registros reprobados en los datos analizados. "
            "Se recomienda mantener las estrategias de acompañamiento."
        )

    asistencia_promedio = datos["Asistencia"].mean()

    if pd.notna(asistencia_promedio):
        if asistencia_promedio < 70:
            st.warning(
                "La asistencia promedio es inferior al 70 %. "
                "Se recomienda revisar las causas de inasistencia "
                "y establecer mecanismos de seguimiento."
            )
        elif asistencia_promedio < 80:
            st.info(
                "La asistencia promedio se encuentra por debajo del 80 %. "
                "Se recomienda fortalecer las estrategias de permanencia."
            )
        else:
            st.success(
                "La asistencia promedio es igual o superior al 80 %. "
                "Conviene mantener las acciones que favorecen la participación."
            )

    materias_criticas = resumen[
        (resumen["Promedio"] < 3.0)
        & resumen["Promedio"].notna()
    ]

    if not materias_criticas.empty:
        lista_materias = ", ".join(
            materias_criticas["Materia"].astype(str).tolist()
        )

        st.warning(
            f"Las materias con promedio inferior a 3.0 son: "
            f"{lista_materias}. Se recomienda revisar sus contenidos "
            f"y reforzar el acompañamiento."
        )

    st.caption(
        "Las recomendaciones son orientativas y se generan mediante "
        "reglas descriptivas. No representan diagnósticos individuales."
    )

    st.subheader("Registros que requieren seguimiento")

    seguimiento = datos[
        (datos["Nota_Final"] < 3.0)
        | (datos["Asistencia"] < 70)
    ].copy()

    if seguimiento.empty:
        st.success("No hay registros que cumplan estos criterios.")
    else:
        columnas_seguimiento = [
            "Estudiante",
            "Materia",
            "Nota_Final",
            "Asistencia",
            "Horas_Estudio",
            "Tareas"
        ]

        st.dataframe(
            seguimiento[columnas_seguimiento],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Criterios: nota inferior a 3.0 o asistencia inferior al 70 %."
        )

    st.subheader("Descargar reportes")

    # Exportación CSV.
    csv_datos = datos.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "Descargar datos académicos (CSV)",
        data=csv_datos,
        file_name="edupredict_datos_academicos.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Construir un libro Excel con varias hojas.
    archivo_excel = BytesIO()

    indicadores = pd.DataFrame({
        "Indicador": [
            "Fuente de información",
            "Registros académicos",
            "Estudiantes únicos",
            "Promedio académico",
            "Aprobados",
            "Reprobados",
            "Porcentaje de aprobación",
            "Porcentaje de reprobación"
        ],
        "Valor": [
            fuente,
            total_registros,
            total_estudiantes,
            round(promedio, 2),
            int(aprobados),
            int(reprobados),
            round(porcentaje_aprobacion, 2),
            round(porcentaje_reprobacion, 2)
        ]
    })

    with pd.ExcelWriter(
        archivo_excel,
        engine="openpyxl"
    ) as writer:

        datos.to_excel(
            writer,
            sheet_name="Datos academicos",
            index=False
        )

        resumen.round(2).to_excel(
            writer,
            sheet_name="Resumen por materia",
            index=False
        )

        indicadores.to_excel(
            writer,
            sheet_name="Indicadores",
            index=False
        )

        seguimiento.to_excel(
            writer,
            sheet_name="Seguimiento",
            index=False
        )

    st.download_button(
        "Descargar reporte completo (Excel)",
        data=archivo_excel.getvalue(),
        file_name="edupredict_reporte_academico.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )



def main():
    aplicar_estilo()
    px.defaults.color_discrete_sequence = PLOTLY_COLORS
    st.sidebar.title("EduPredict")
    st.sidebar.caption("Inteligencia Academica  ·  Analitica")
    st.sidebar.markdown("---")

    pagina = st.sidebar.radio(
        "Navegación",
        [
            "Dashboard",
            "Carga de datos",
            "Análisis académico",
            "Predicciones",
            "Reportes",
        ],
    )

    df, fuente = obtener_datos_activos()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Fuente prederteminada: {fuente}")

    if pagina == "Dashboard":
        pagina_dashboard(df, fuente)
    elif pagina == "Carga de datos":
        pagina_carga()
    elif pagina == "Análisis académico":
        pagina_analisis(df, fuente)
    elif pagina == "Predicciones":
        pagina_predicciones(df, fuente)
    elif pagina == "Reportes":
        pagina_reportes(df, fuente)


if __name__ == "__main__":
    main()
