import pandas as pd
from src.trainer import RPSTrainer
from src.predictor import RPSPredictor
from src.config import RPSConfig


def train_basic_model():
    print("Начинаем обучение базовой модели...")
    trainer = RPSTrainer.create_noname_predictor()
    trainer.train()
    print("Обучение законченно")


def train_current_enemy_model(enemy_name):
    print(f"Начинаем обучение модели против врага {enemy_name}...")
    trainer = RPSTrainer.create_current_enemy_predictor(enemy_name)
    trainer.train()
    print("Обучение законченно")


def predict_move(history: str):
    # Загружаем "базовую" модель или модель врага
    try:
        predictor = RPSPredictor(enemy_name="basic")
        move = predictor.predict(history)
        print(f"Для истории: {history} -> Лучший ход: {move}")
    except FileNotFoundError:
        print("Error: Модель не найдена, сначала обучите модель")


if __name__ == "__main__":
    # Пример использования
    # 1. Обучение
    train_basic_model()

    # 2. Предсказание
    # История: Я сыграл камень (r), он бумагу (p) -> 'rp'
    fake_history = "sprpss"
    predict_move(fake_history)
