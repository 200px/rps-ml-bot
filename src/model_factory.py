import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Optional
from .config import RPSConfig
from sklearn.dummy import DummyClassifier


class ModelFactory:
    """Создание моделей"""

    def __init__(self, config: Optional[RPSConfig] = None):
        self.config = config or RPSConfig()

    def create_stacking(self, X: pd.DataFrame, y: pd.Series) -> StackingClassifier:
        """Создание метамодели"""
        cfg = self.config

        # Кросс-валидация для обучения метамодели
        cv_strategy = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True)

        # Методы предобработки данных
        onehot = OneHotEncoder(drop=None, handle_unknown="ignore")
        onehot_drop = OneHotEncoder(drop="first", handle_unknown="ignore")

        # Количество k соседей для knn
        knn_n_neighbors = min(cfg.knn_max_neighbors, len(y) // 2) if len(y) > 1 else 1

        # Pipilines базовых моделей
        pipe_lr = Pipeline(
            [("preproc", onehot_drop), ("clf", LogisticRegression(max_iter=1000))]
        )
        pipe_knn = Pipeline(
            [
                ("preproc", onehot),
                ("clf", KNeighborsClassifier(n_neighbors=knn_n_neighbors)),
            ]
        )
        pipe_svc = Pipeline(
            [("preproc", onehot), ("clf", SVC(probability=True, kernel="rbf"))]
        )
        pipe_rf = Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=cfg.rf_n_estimators,
                        max_depth=cfg.rf_max_depth,
                        n_jobs=-1,
                    ),
                )
            ]
        )
        pipe_dummy = Pipeline([("clf", DummyClassifier())])
        pipe_catboost = Pipeline(
            [
                (
                    "clf",
                    CatBoostClassifier(
                        iterations=cfg.catboost_iterations,
                        learning_rate=cfg.catboost_learning_rate,
                        depth=cfg.catboost_depth,
                        verbose=False,
                        allow_writing_files=False,
                        cat_features=list(range(X.shape[1])),
                    ),
                )
            ]
        )

        estimators = [
            ("lr", pipe_lr),
            ("rf", pipe_rf),
            ("knn", pipe_knn),
            ("svc", pipe_svc),
            ("dummy", pipe_dummy),
            ("cat", pipe_catboost),
        ]

        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=cv_strategy,
            passthrough=False,
        )
        return stacking_clf
