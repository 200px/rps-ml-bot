import numpy as np
import joblib
from .config import RPSConfig
from .features import FeatureEngineer


class RPSPredictor:
    """
    Класс для предсказаний на основе обученных моделей
    """

    def __init__(self, enemy_name="basic"):
        self.config = RPSConfig()
        self._load_models(self.config.models_dir / f"{enemy_name}.joblib")
        self.feature_engineer = FeatureEngineer()

    def _load_models(self, models_path):
        """Загрузка сохраненных моделей"""
        data = joblib.load(models_path)
        self.models = data["models"]
        self.max_round = data["max_round"]

    def _calc_best_move(self, probs):
        """Рассчет лучшего хода на основании вероятностей выбора врагом каждой фигуры"""
        r_score = probs[2] - probs[1]
        p_score = probs[0] - probs[2]
        s_score = probs[1] - probs[0]
        scores = {"r": r_score, "p": p_score, "s": s_score}
        return max(scores, key=scores.get)

    def predict(self, history: str) -> str:
        """Основной метод предсказания на основании текущей истории игры"""
        # Определяем, какой раунд нужно предсказать:
        next_round = len(history) // 2 + 1

        # Если для этого раунда нет модели возвращаем рандомный ход
        if next_round > self.max_round:
            return np.random.choice(["r", "p", "s"])

        X_pred = self.feature_engineer.history_to_features(history)

        # Получаем модель
        model = self.models[next_round]

        # Получаем предсказние модели и высчитываем лучший ход
        enemy_move_probs = model.predict_proba(X_pred)[0]
        best_move = self._calc_best_move(enemy_move_probs)
        return best_move
