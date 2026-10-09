from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).with_name("data.csv")

MEAN_NUMERIC = ["experience_years"]
MEDIAN_NUMERIC = ["age", "monthly_income"]
CATEGORICAL = ["city", "education", "employment_type"]
TARGET = "target"


def build_preprocessor() -> ColumnTransformer:
    mean_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    median_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric_mean", mean_numeric_pipeline, MEAN_NUMERIC),
            ("numeric_median", median_numeric_pipeline, MEDIAN_NUMERIC),
            ("categorical", categorical_pipeline, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor()

    # ВАЖНО: fit только на train. Test не участвует в вычислении
    # среднего, медианы, масштаба и категорий для OneHotEncoder.
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    print("Исходная размерность train:", X_train.shape)
    print("Исходная размерность test: ", X_test.shape)
    print("После предобработки train:", X_train_ready.shape)
    print("После предобработки test: ", X_test_ready.shape)
    print("Количество итоговых признаков:", len(feature_names))
    print("Итоговые признаки:")
    for name in feature_names:
        print(" -", name)

    assert X_train_ready.shape[1] == X_test_ready.shape[1]
    assert X_train_ready.shape[1] == len(feature_names)
    print("\nПроверка пройдена: train и test имеют одинаковое число столбцов.")
    print("Утечки нет: пайплайн обучался только на train.")

    # y_train/y_test здесь не преобразуются: они понадобятся модели.
    print(f"Объектов: train={len(y_train)}, test={len(y_test)}")


if __name__ == "__main__":
    main()
