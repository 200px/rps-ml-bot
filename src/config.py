from dataclasses import dataclass
from pathlib import Path


@dataclass
class RPSConfig:
    """Хранилище настроек"""

    # Сколько последних первых раундов предсказывают первый ход
    first_rounds_tail: int = 100

    # Минимальное количество игр на раунд, для обучения модели для раунда
    min_games_per_round: int = 25

    # Гиперпараметры моделей
    rf_n_estimators: int = 500
    rf_max_depth: int = 6
    catboost_iterations: int = 100
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 4
    knn_max_neighbors: int = 20
    cv_splits: int = 5

    # Пути к файлам и папкам
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    games_file: str = "games.csv"
    first_games_file: str = "first_games_only.csv"

    @property
    def games_path(self) -> Path:
        return self.data_dir / self.games_file

    @property
    def first_games_path(self) -> Path:
        return self.data_dir / self.first_games_file

    def model_path(self, enemy_name: str = "") -> Path:
        filename = f"{enemy_name}.joblib" if enemy_name else "basic.joblib"
        return self.models_dir / filename
