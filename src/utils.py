import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Tuple


def get_data(data_path: str):
    df = pd.read_csv(data_path, sep=",", decimal=".")
    return df


def get_train_data():
    train_path = "/home/vinicius/repositories/ppgia/msc/data/train.csv"
    return get_data(train_path)


def get_test_data():
    test_path = "/home/vinicius/repositories/ppgia/msc/data/test.csv"
    return get_data(test_path)


def get_validation_data():
    validation_path = "/home/vinicius/repositories/ppgia/msc/data/validation.csv"
    return get_data(validation_path)


def transform_data(df: pd.DataFrame) -> pd.DataFrame:

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

        return None

    df.drop(columns=["video", "start_time", "end_time", "ASR"], inplace=True)

    df["emotion_category"] = df.apply(emotions_to_category, axis=1)

    cleaned_df = df.loc[df["emotion_category"] != "undefined"].copy()

    cleaned_df["text_clean"] = df["text"].apply(clean_text)

    return cleaned_df[
        [
            "sentiment",
            "happy",
            "sad",
            "anger",
            "fear",
            "disgust",
            "surprise",
            "emotion_category",
            "text_clean",
        ]
    ].copy()


# Certifique-se de baixar os recursos necessários do NLTK (executar apenas uma vez)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")


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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = get_train_data()
    test = get_test_data()
    validation = get_validation_data()

    if format_data:

        train = transform_data(train.copy())
        test = transform_data(test.copy())
        validation = transform_data(validation.copy())

    return train, test, validation


if __name__ == "__main__":
    train, test, validation = get_cmu_mosei_dataset(format_data=True)

    print(validation)
