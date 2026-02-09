import pandas as pd
import numpy as np


class FeatureEngineer:
    """Предобработчик данных"""

    def __init__(self):
        self.mapping = {"r": 0, "p": 1, "s": 2}

    def prepare_round_data(
        self, df: pd.DataFrame, round_num: int
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Создание подвыборки по раунду и кодирование через маппинг."""
        data = df[df["round"] <= round_num].copy()
        pivot = data.pivot(
            index="id", columns="round", values=["my_move", "enemy_move"]
        )
        pivot = pivot.dropna()
        pivot.columns = [f"{col}_{r}" for col, r in pivot.columns]
        pivot = pivot.map(self.mapping.get)

        target = f"enemy_move_{round_num}"
        drop_cols = [f"my_move_{round_num}", target]
        return pivot.drop(columns=drop_cols), pivot[target]

    def history_to_features(self, history: str) -> pd.DataFrame:
        """Кодирование истории текущей игры для подачи в модель как X фичи"""
        current_round = len(history) // 2
        my_moves = list(history[::2])
        enemy_moves = list(history[1::2])

        cols = [f"my_move_{i + 1}" for i in range(current_round)] + [
            f"enemy_move_{i + 1}" for i in range(current_round)
        ]
        df = pd.DataFrame([my_moves + enemy_moves], columns=cols)
        return df.map(self.mapping.get)
