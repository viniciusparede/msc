import os
import time

import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


from data_utils import read_cmu_mosei, create_dataset, clean_dataset

from typing import Tuple


API_KEY_TEST = "kO6QZbZYULZHdmBzZPFt06GXsSNjrV6o"
API_KEY_PROD = "8VilW6b6cP6vcnZ5konSHWrf2VufmI65"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repo_dir = os.getcwd()
    data_dir = os.path.join(repo_dir, "data")

    words, labels = read_cmu_mosei(data_dir=data_dir)

    df = create_dataset(words, labels)

    train, valid, test = clean_dataset(dataset=df)

    return train, valid, test


def run_mistral(user_message, model="mistral-medium"):
    client = MistralClient(api_key=API_KEY_PROD)
    messages = [ChatMessage(role="user", content=user_message)]
    chat_response = client.chat(model=model, messages=messages, temperature=0.01)
    return chat_response.choices[0].message.content


def user_message(inquiry):
    user_message = f"""
        You are an expert in classifying emotions. You are required to classify each sentence 
        <<<>>> into one of the following predefined categories: 
        
        Happiness
        Sadness
        Anger
        Fear
        Disgust
        Surprise
        Neutral
        
        If the text doesn't fit into any of the above categories, classify it as:
        Other
        
        You will only respond with the predefined category. Do not include the word "Category". Do not provide explanations or notes. 
        
        ####
        Here are some examples:
        
        Inquiry: I am overjoyed to announce I've been promoted at work!
        Category: Happiness
        Inquiry: I feel devastated after hearing about the loss of my childhood friend.
        Category: Sadness
        Inquiry: It infuriates me how unfair the system can be towards those who need help the most.
        Category: Anger
        Inquiry: I'm terrified of speaking in public, and I have a presentation next week.
        Category: Fear
        ###
    
        <<<
        Inquiry: {inquiry}
        >>>
        """
    return user_message


def mistral_emotion_classification(sentence: str) -> str:
    max_attempts = 10
    attempt_count = 0

    while attempt_count < max_attempts:
        try:
            emotion = run_mistral(user_message(inquiry=sentence))
            break  # Sai do loop após uma classificação bem-sucedida
        except:
            attempt_count += 1  # Incrementa o contador de tentativas após uma falha
            print(
                f"Falha na tentativa {attempt_count} para classificar a sentença: '{sentence}'"
            )
            time.sleep(1)  # Pausa antes de tentar novamente

            if attempt_count == max_attempts:
                print(
                    f"Maximo de tentativas atingidas para a sentença: '{sentence}'. Classificação não realizada."
                )
                return "API_ERROR"

    return emotion


if __name__ == "__main__":
    # Carregar dados
    train, valid, test = load_data()

    valid.to_csv("valid_gpt.csv", index=False, sep=";")
    test.to_csv("test_gpt.csv", index=False, sep=";")

    valid_classification_mistral = list()
    request_count = 0
    for _, data_valid in valid.iterrows():
        sentence = data_valid["Sentence"]
        emotion = mistral_emotion_classification(sentence)
        valid_classification_mistral.append(emotion)

        print(sentence, emotion)

        request_count += 1

        # Verificar se já foram feitas 2 requisições
        if request_count % 2 == 0:
            time.sleep(1)  # Pausa de 1 segundo após cada 2 requisições

    valid["Emotion_Mistral"] = valid_classification_mistral

    valid.to_csv("valid_mistral.csv", index=False, sep=";")

    test_classification_mistral = list()
    for _, data_test in test.iterrows():
        sentence = data_valid["Sentence"]
        emotion = mistral_emotion_classification(sentence)
        test_classification_mistral.append(emotion)

        print(sentence, emotion)

        request_count += 1

        # Verificar se já foram feitas 2 requisições
        if request_count % 2 == 0:
            time.sleep(1)  # Pausa de 1 segundo após cada 2 requisições

    test["Emotion_Mistral"] = test_classification_mistral

    test.to_csv("test_mistral.csv", index=False, sep=";")
