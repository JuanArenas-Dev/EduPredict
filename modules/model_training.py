
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Columnas utilizadas para realizar las predicciones.
FEATURES = [
    "Materia",
    "Horas_Estudio",
    "Asistencia",
    "Tareas"
]

TARGET = "Estado"

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def prepare_data(df):
    """
    Prepara los datos y construye la variable objetivo
    a partir de Nota_Final.
    """

    required = FEATURES + ["Nota_Final"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias: {', '.join(missing)}"
        )

    data = df[required].copy()

    # Convertir variables numéricas.
    numeric_cols = [
        "Horas_Estudio",
        "Asistencia",
        "Tareas",
        "Nota_Final"
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Eliminar registros sin nota final, ya que no tienen etiqueta.
    data = data.dropna(subset=["Nota_Final"])

    # Construir la variable que aprenderá a predecir el modelo.
    data[TARGET] = data["Nota_Final"].apply(
        lambda nota: "Aprobado" if nota >= 3.0 else "Reprobado"
    )

    # Las demás variables se imputarán dentro del pipeline.
    data = data.dropna(subset=["Materia"])

    return data


def build_pipeline(model):
    """
    Construye el preprocesamiento y el modelo
    en un único pipeline.
    """

    numeric_features = [
        "Horas_Estudio",
        "Asistencia",
        "Tareas"
    ]

    categorical_features = ["Materia"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    return pipeline


def train_models(df):
    """
    Entrena Árbol de Decisión y Regresión Logística.
    Devuelve los modelos, métricas y matrices de confusión.
    """

    data = prepare_data(df)

    if len(data) < 8:
        raise ValueError(
            "Se necesitan al menos 8 registros válidos para "
            "intentar entrenar y evaluar los modelos."
        )

    class_counts = data[TARGET].value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "Los datos deben contener estudiantes aprobados "
            "y reprobados."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Cada clase debe tener al menos 2 registros. "
            "Agrega más estudiantes aprobados o reprobados."
        )

    X = data[FEATURES]
    y = data[TARGET]

    # Separar datos: 75 % entrenamiento y 25 % prueba.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    models = {
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ),

        "Regresión Logística": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    }

    results = {}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, estimator in models.items():

        pipeline = build_pipeline(estimator)

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        metrics = {
            "Exactitud": accuracy_score(y_test, predictions),
            "Precisión (Reprobado)": precision_score(
                y_test,
                predictions,
                pos_label="Reprobado",
                zero_division=0
            ),
            "Sensibilidad (Reprobado)": recall_score(
                y_test,
                predictions,
                pos_label="Reprobado",
                zero_division=0
            ),
            "F1 (Reprobado)": f1_score(
                y_test,
                predictions,
                pos_label="Reprobado",
                zero_division=0
            )
        }

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=["Aprobado", "Reprobado"]
        )

        # Guardar el pipeline completo.
        safe_name = (
            "arbol_decision"
            if name == "Árbol de Decisión"
            else "regresion_logistica"
        )

        joblib.dump(
            pipeline,
            MODEL_DIR / f"{safe_name}.joblib"
        )

        results[name] = {
            "model": pipeline,
            "metrics": metrics,
            "confusion_matrix": matrix,
            "y_test": y_test,
            "predictions": predictions,
            "report": classification_report(
                y_test,
                predictions,
                zero_division=0,
                output_dict=True
            )
        }

    return results


def predict_student(model, materia, horas, asistencia, tareas):
    """
    Predice el resultado académico de un estudiante nuevo.
    """

    student = pd.DataFrame([{
        "Materia": materia,
        "Horas_Estudio": horas,
        "Asistencia": asistencia,
        "Tareas": tareas
    }])

    prediction = model.predict(student)[0]

    probabilities = model.predict_proba(student)[0]
    classes = model.named_steps["classifier"].classes_

    confidence = dict(zip(classes, probabilities))

    return prediction, confidence


def get_feature_importance(model):
    """
    Obtiene la importancia de las variables de un Árbol de Decisión.
    """

    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    if not hasattr(classifier, "feature_importances_"):
        raise ValueError(
            "Este modelo no dispone de importancia de variables."
        )

    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "Variable": feature_names,
        "Importancia": classifier.feature_importances_
    })

    # Simplificar los nombres para mostrarlos en el dashboard.
    importance["Variable"] = (
        importance["Variable"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    importance = importance.sort_values(
        "Importancia",
        ascending=False
    ).reset_index(drop=True)

    return importance

