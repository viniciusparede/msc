import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Tuple

from sklearn.preprocessing import LabelEncoder

from datasets import Dataset
from datasets import DatasetDict

REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_PATH, "data")

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")


def get_data(data_path: str):
    df = pd.read_csv(data_path, sep=",", decimal=".")
    return df


def get_train_data():
    train_path = os.path.join(DATA_PATH, "train.csv")
    return get_data(train_path)


def get_test_data():
    test_path = os.path.join(DATA_PATH, "test.csv")
    return get_data(test_path)


def get_validation_data():
    validation_path = os.path.join(DATA_PATH, "validation.csv")
    return get_data(validation_path)


def emotions_to_category(row):
    emotions = ["happy", "sad", "anger", "fear", "disgust", "surprise"]

    # Criar uma lista de tuplas (emocao, média)
    emotions_average = [(emotion, row[emotion]) for emotion in emotions]

    # Verifica se todos as médias são zeros
    if all(average == 0 for _, average in emotions_average):
        return "neutral"

    # Conta quantas tuplas têm a média diferente de zero
    count_non_zero = sum(average != 0 for _, average in emotions_average)

    # Verifica se apenas uma tupla têm média diferente de zero
    if count_non_zero == 1:
        for emotion, average in emotions_average:
            if average != 0:
                return emotion

    # MultiLabel: Escolhe a emoção com a maior média se houver mais de uma média diferente de zero
    if count_non_zero > 1:
        sorted_emotions = sorted(emotions_average, key=lambda x: x[1], reverse=True)
        # Verifica se há um empate na maior média
        if sorted_emotions[0][1] == sorted_emotions[1][1]:
            return "undefined"  # Ou pode retornar None
        return sorted_emotions[0][0]


def transform_data(df: pd.DataFrame) -> pd.DataFrame:

    df.drop(
        columns=["video", "start_time", "end_time", "ASR", "sentiment"], inplace=True
    )

    df["emotion_category"] = df.apply(emotions_to_category, axis=1)

    cleaned_df = df.loc[df["emotion_category"] != "undefined"].copy()

    cleaned_df["text_clean"] = df["text"].apply(clean_text)

    cleaned_df.drop(
        columns=[
            "text",
            "happy",
            "sad",
            "anger",
            "fear",
            "disgust",
            "surprise",
        ],
        inplace=True,
    )

    cleaned_df.rename(
        columns={"text_clean": "text", "emotion_category": "label"}, inplace=True
    )

    df.reset_index(drop=True, inplace=True)

    return cleaned_df[
        [
            "label",
            "text",
        ]
    ].copy()


def clean_text(text):
    # Remover caracteres especiais e dígitos
    text = re.sub(r"[^a-zA-Z\s]", "", text, re.I | re.A)
    text = re.sub(r"\d+", "", text)

    # Converter para letras minúsculas
    text = text.lower()

    # Remover stop words
    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]

    # Lematização
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(lemmatized_words)


def get_cmu_mosei_dataset(
    format_data: bool = False,
):

    train = get_train_data()
    test = get_test_data()
    validation = get_validation_data()

    if format_data:

        train = transform_data(train.copy())
        test = transform_data(test.copy())
        validation = transform_data(validation.copy())

    le = LabelEncoder()
    train["label"] = le.fit_transform(train["label"])
    validation["label"] = le.transform(validation["label"])
    test["label"] = le.transform(test["label"])

    train_ds = Dataset.from_pandas(train).remove_columns("__index_level_0__")
    test_ds = Dataset.from_pandas(test).remove_columns("__index_level_0__")
    validation_ds = Dataset.from_pandas(validation).remove_columns("__index_level_0__")

    return DatasetDict(
        {
            "train": train_ds,
            "test": test_ds,
            "validation": validation_ds,
        }
    )


if __name__ == "__main__":
    ds = get_cmu_mosei_dataset(format_data=True)

    print(ds)
