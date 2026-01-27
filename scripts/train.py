"""
Обучение и оценка наивного байесовского классификатора спама.
Запуск: python scripts/train.py
"""

import pandas as pd
from src.model import NaiveBayesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def main():
    print("Загрузка данных...")
    df_train = pd.read_csv("data/processed/train.csv")
    df_test = pd.read_csv("data/processed/test.csv")

    df_train = df_train.dropna(
        subset=['cleaned', 'label']).reset_index(drop=True)
    df_test = df_test.dropna(
        subset=['cleaned', 'label']).reset_index(drop=True)

    X_train = df_train['cleaned'].tolist()
    y_train = df_train['label'].tolist()
    X_test = df_test['cleaned'].tolist()
    y_test = df_test['label'].tolist()

    print("Обучение модели...")
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)

    print("Оценка на тестовой выборке...")
    y_pred = [nb.predict(text) for text in X_test]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='spam')
    rec = recall_score(y_test, y_pred, pos_label='spam')

    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ")
    print("="*50)
    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision (spam): {prec:.4f}")
    print(f"Recall (spam):    {rec:.4f}")
    print("\nПодробно:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
