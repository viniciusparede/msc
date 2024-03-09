import os
import re
import string

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

from openai import OpenAI


from data_utils import read_cmu_mosei, create_dataset, clean_dataset

from typing import Tuple


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repo_dir = os.getcwd()
    data_dir = os.path.join(repo_dir, "data")

    words, labels = read_cmu_mosei(data_dir=data_dir)

    df = create_dataset(words, labels)

    train, valid, test = clean_dataset(dataset=df)

    return train, valid, test


def gpt_classification(sentence: str, prompt: list[dict]) -> str:

    test_key = "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"
    official_key = "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"

    client = OpenAI(
        api_key=official_key,
    )

    emotions_category = [
        "Happiness",
        "Anger",
        "Neutral",
        "Fear",
        "Sadness",
        "Disgust",
        "Surprise",
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=prompt + [{"role": "user", "content": f"{sentence}"}],
        temperature=0,
    )

    emotion = response.choices[0].message.content

    if not emotion in emotions_category:
        return "Other"

    print(emotion)

    return emotion


if __name__ == "__main__":
    # Carregar dados
    train, valid, test = load_data()

    valid.to_csv("valid_gpt.csv", index=False, sep=";")
    test.to_csv("test_gpt.csv", index=False, sep=";")

    system_prompt = {
        "role": "system",
        "content": "I will give you text sentences. You are required to classify each sentence into one of the 7 categories based on the emotion it expresses. If the emotion of the sentence does not match any of the categories listed, you should classify it as OTHERS.\n\nCategories:\n1. Happiness: A state of well-being and contentment; joy.\n2. Sadness: A condition or feeling of sorrow or unhappiness.\n3. Anger: A strong feeling of displeasure or hostility.\n4. Fear: An intensely unpleasant emotion in response to perceiving or recognizing a danger or threat.\n5. Disgust: A strong feeling of dislike or disapproval.\n6. Surprise: A feeling caused by something unexpected happening.\n7. Neutral: Neither particularly good nor bad; not showing or involving any emotion.\nOutput Format: Category name",
    }

    emotions_category = train["Emotion_Category"].unique()

    few_shots = []

    for emotion_category in emotions_category:
        # Filtra o dataframe para a categoria de emoção atual e seleciona 5 amostras aleatórias
        random_sample = train[train["Emotion_Category"] == emotion_category].sample(n=2)

        for _, row in random_sample.iterrows():
            # Adiciona o par de 'user' e 'assistant' para cada amostra selecionada
            few_shots.append({"role": "user", "content": row["Sentence"]})
            few_shots.append({"role": "assistant", "content": row["Emotion_Category"]})

    prompt = [system_prompt] + few_shots

    valid["Emotion_GPT"] = valid["Sentence"].apply(gpt_classification, args=(prompt,))
    test["Emotion_GPT"] = test["Sentence"].apply(gpt_classification, args=(prompt,))

    valid.to_csv("valid_gpt.csv", index=False, sep=";")
    test.to_csv("test_gpt.csv", index=False, sep=";")

    # Identificando os índices das linhas onde Emotion_GPT é igual a 'Other'
    indices_to_drop_valid = valid[valid["Emotion_GPT"] == "Other"].index
    indices_to_drop_test = test[test["Emotion_GPT"] == "Other"].index

    # Removendo essas linhas dos DataFrames
    valid.drop(indices_to_drop_valid, inplace=True)
    test.drop(indices_to_drop_test, inplace=True)

    # Validação
    print(
        classification_report(
            y_true=valid["Emotion_Category"].values, y_pred=valid["Emotion_GPT"].values
        )
    )
    print(
        confusion_matrix(
            y_true=valid["Emotion_Category"].values, y_pred=valid["Emotion_GPT"].values
        )
    )

    print()
    print()

    # Teste
    print(
        classification_report(
            y_true=test["Emotion_Category"].values, y_pred=test["Emotion_GPT"].values
        )
    )
    print(
        confusion_matrix(
            y_true=test["Emotion_Category"].values, y_pred=test["Emotion_GPT"].values
        )
    )
