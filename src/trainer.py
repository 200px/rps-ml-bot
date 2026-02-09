import pandas as pd
import joblib
from typing import Optional
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from .config import RPSConfig
from .features import FeatureEngineer
from .model_factory import ModelFactory


class RPSTrainer:
    """Основной класс для обучения моделей"""

    def __init__(
        self, df: pd.DataFrame, config: Optional[RPSConfig] = None, model_name: str = ""
    ):
        self.config = config or RPSConfig()
        self.df = df
        self.model_name = model_name
        self.models = {}

        # Устанавливаем максимальный раунд для обучения по нему модели
        round_grouped = df.groupby("round")["enemy_move"].count()
        self.max_round = round_grouped[
            round_grouped > self.config.min_games_per_round
        ].idxmin()

    @classmethod
    def create_noname_predictor(cls):
        """Создание RPSTrainer для всех противников"""
        config = RPSConfig(min_games_per_round=50)
        df = pd.read_csv(config.first_games_path)
        return RPSTrainer(df, config=config, model_name="basic")

    @classmethod
    def create_current_enemy_predictor(cls, enemy_name):
        """Создание RPSTrainer для конкретного противника"""
        config = RPSConfig(min_games_per_round=25)
        df = pd.read_csv(config.games_path)
        df = df[df["enemy_name"] == enemy_name]
        return RPSTrainer(df=df, config=config, model_name=enemy_name)

    def _train_models(self, cv_test=False) -> None:
        """Основной метод обучения моделей для каждого раунда"""
        feature_engineer = FeatureEngineer()

        # Для первого раунда просто предсказываем самую часто выбираемую фигуры используя DummyClassifier
        X, y = feature_engineer.prepare_round_data(self.df, 1)
        last_X = X.tail(self.config.first_rounds_tail)
        last_y = y.tail(self.config.first_rounds_tail)

        model = DummyClassifier()
        model.fit(last_X, last_y)
        self.models[1] = model

        model_factory = ModelFactory()

        # Создание и обучении отдельной мета модели для каждого раунда
        for round_num in range(2, self.max_round + 1):
            X, y = feature_engineer.prepare_round_data(self.df, round_num)

            model = model_factory.create_stacking(X, y)

            if cv_test:
                cv_scores = cross_val_score(
                    model, X, y, cv=self.config.cv_splits, scoring="accuracy", n_jobs=-1
                )
                mean_acc = cv_scores.mean()
                std_acc = cv_scores.std()
                print(
                    f"Round {round_num:<2} | CV Acc: {mean_acc:.2%} (+/- {std_acc:.2%})"
                )

            # Финальное обучение и сохранение модели
            model.fit(X, y)
            self.models[round_num] = model

    def _save_models(self):
        data = {
            "models": self.models,
            "max_round": self.max_round,
        }
        model_path = self.config.models_dir / f"{self.model_name}.joblib"
        joblib.dump(data, model_path)

    def train(self):
        self._train_models()
        self._save_models()
