import os
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from gensim.models import KeyedVectors
from catboost import CatBoostClassifier


import pandas as pd
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from itertools import cycle

from data_utils import read_cmu_mosei, create_dataset, clean_dataset
from typing import Tuple

# Download NLTK
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

# Word2Vec pré-treinado -> Google News
word_vectors = KeyedVectors.load_word2vec_format(
    "/home/vinicius/Documentos/Repositories/msc/GoogleNews-vectors-negative300.bin",
    binary=True,
)


def get_word2vec_embedding(sentence):
    tokens = word_tokenize(sentence.lower())

    vectors = [word_vectors[word] for word in tokens if word in word_vectors]

    if not vectors:
        # vetor de 300 dimensões
        return np.zeros(300)

    sentence_embedding = np.mean(vectors, axis=0)
    return sentence_embedding


def text_preprocessing_pipeline(text, remove_stopwords=True, lemmatize=True):
    # Conversão para minúsculas
    text = text.lower()

    # Remoção de URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remoção de HTML
    text = re.sub(r"<.*?>", "", text)

    # Remoção de números
    text = re.sub(r"\d+", "", text)

    # Remoção de pontuação
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenização
    tokens = word_tokenize(text)

    # Remoção de stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

    # Lematização
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Reconstrução do texto
    preprocessed_text = " ".join(tokens)

    return preprocessed_text


# Função para carregar dados
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repo_dir = os.getcwd()
    data_dir = os.path.join(repo_dir, "data")

    words, labels = read_cmu_mosei(data_dir=data_dir)
    df = create_dataset(words, labels)
    train, valid, test = clean_dataset(dataset=df)

    # Encontrar o número mínimo de exemplos em todas as classes
    # min_class_count = train['Emotion_Category'].value_counts().min()

    # Realizar undersampling para cada classe
    # train_balanced = train.groupby('Emotion_Category').apply(lambda x: x.sample(min_class_count)).reset_index(drop=True)

    # Embaralhar os dados
    # train_balanced = train_balanced.sample(frac=1).reset_index(drop=True)

    return train, valid, test


def plot_multiclass_roc(clf, X_test, y_test, n_classes):
    # Obter as pontuações de decisão ou probabilidades das previsões
    y_score = clf.decision_function(X_test)

    # Computar ROC curve e ROC area para cada classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Definir cores para cada classe
    colors = cycle(["blue", "red", "green", "yellow", "purple", "orange", "brown"])

    # Identificar as classes únicas
    classes = np.unique(np.argmax(y_test, axis=1))

    # Plotar todas as ROC curves
    plt.figure()
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})"
            "".format(classes[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multi-class")
    plt.legend(loc="lower right")
    # Salvar a figura em vez de exibi-la
    plt.savefig("teste.png")


# Função principal
def main():
    # Carregar dados
    train, valid, test = load_data()

    # Converter sentenças para BERT embeddings
    X_train = np.vstack(train["Sentence"].apply(lambda x: get_word2vec_embedding(x)))
    X_valid = np.vstack(valid["Sentence"].apply(lambda x: get_word2vec_embedding(x)))
    X_test = np.vstack(test["Sentence"].apply(lambda x: get_word2vec_embedding(x)))

    y_train = train["Emotion_Category"].values
    y_valid = valid["Emotion_Category"].values
    y_test = test["Emotion_Category"].values

    # Binarize os rótulos
    y = np.concatenate([y_train, y_valid, y_test])
    classes = np.unique(y)
    n_classes = len(classes)

    y_train_bin = label_binarize(y_train, classes=classes)
    y_valid_bin = label_binarize(y_valid, classes=classes)
    y_test_bin = label_binarize(y_test, classes=classes)

    # Treinar e avaliar o modelo SVM
    svc = SVC()
    svc.fit(X_train, y_train)

    # Avaliação com o conjunto de validação
    predictions_valid = svc.predict(X_valid)
    print("Avaliação no Conjunto de Validação:")
    print(classification_report(y_valid, predictions_valid))
    print(confusion_matrix(y_valid, predictions_valid))

    # Chamar a função para o modelo SVC
    plot_multiclass_roc(svc, X_valid, y_valid_bin, n_classes)

    # Avaliação com o conjunto de test
    predictions_test = svc.predict(X_test)
    print("Avaliação no Conjunto de Validação:")
    print(classification_report(y_test, predictions_test))
    print(confusion_matrix(y_test, predictions_test))

    # Chamar a função para o modelo SVC
    plot_multiclass_roc(svc, X_test, y_test_bin, n_classes)

    """# Inicializar e treinar o modelo CatBoost
    catboost_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        depth=10,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=200,
    )

    catboost_model.fit(
        X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True
    )

    # Avaliação com o conjunto de validação
    predictions_valid = catboost_model.predict(X_valid)
    print("Avaliação no Conjunto de Validação:")
    print(classification_report(y_valid, predictions_valid))
    print(confusion_matrix(y_valid, predictions_valid.flatten()))

    # Avaliação com o conjunto de teste
    predictions_test = catboost_model.predict(X_test)
    print("Avaliação no Conjunto de Teste:")
    print(classification_report(y_test, predictions_test))
    print(confusion_matrix(y_test, predictions_test.flatten()))
    """


if __name__ == "__main__":
    main()
