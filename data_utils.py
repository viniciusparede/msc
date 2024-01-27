# Objetivo aqui é gerar o dataset


import pandas as pd
import numpy as np


from mmsdk import mmdatasdk


CMU_MOSEI_LABELS = [
    "Sentiment",
    "Happiness",
    "Sadness",
    "Anger",
    "Fear",
    "Disgust",
    "Surprise",
]


def open_sentence(words, labels, key: str) -> pd.DataFrame:
    data = []
    current_sentence = []
    current_label = None

    word = words[key]
    label = labels[key]

    word_features = word["features"][:]
    word_features = np.array(
        [word_feature[0].decode("utf-8") for word_feature in word_features]
    )

    word_intervals = word["intervals"][:]

    label_features = label["features"][:]
    label_intervals = label["intervals"][:]

    print(word)

    for token_interval, token in zip(word_intervals, word_features):
        token_start, token_end = token_interval

        if token == "sp":
            continue

        for label_interval, label in zip(label_intervals, label_features):
            label_start, label_end = label_interval

            if label_start <= token_start < label_end:
                # Comparar se as labels são diferentes
                if current_label is not None and not np.array_equal(
                    current_label, label
                ):
                    # Adicionando a frase e o rótulo anterior ao DataFrame
                    data.append([" ".join(current_sentence)] + list(current_label))
                    current_sentence = []
                current_label = label
                break

        current_sentence.append(token)

    # Adicionando a última frase e rótulo, se existirem
    if current_sentence and current_label is not None:
        data.append([" ".join(current_sentence)] + list(current_label))

    # Criar DataFrame
    df = pd.DataFrame(data, columns=["Sentence"] + CMU_MOSEI_LABELS)

    # Exibir o DataFrame
    display(df)


def create_dataset() -> pd.DataFrame:
    pass


def read_cmu_mosei(data_dir: str):
    dataset = mmdatasdk.mmdataset(data_dir)
    words = dataset.computational_sequences["words"]
    labels = dataset.computational_sequences["All Labels"]

    return words, labels


def main():
    cmumosei_dir = "/home/vinicius/Documentos/Repositories/test-cmumosei/data"
    words, labels = read_cmu_mosei(data_dir=cmumosei_dir)

    print(open_sentence(words, labels, key="kwkhYpCHWPw"))

    # df = open()


if __name__ == "__main__":
    main()
