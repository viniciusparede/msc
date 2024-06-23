import pandas as pd
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from utils import get_cmu_mosei_dataset
from models.hugging_face import HuggingFaceClassifier


import time
import os


def get_data():
    train, test, validation = get_cmu_mosei_dataset(format_data=True)
    return train, test, validation


def get_random_sentence(df, emotion):
    return df[df["emotion_category"] == emotion].sample(5)["text_clean"].values[0]


def get_shots():
    emotions = [
        {"category": "happy", "label": "happy"},
        {"category": "sad", "label": "sadness"},
        {"category": "anger", "label": "anger"},
        {"category": "fear", "label": "fear"},
        {"category": "disgust", "label": "disgust"},
        {"category": "surprise", "label": "surprise"},
        {"category": "neutral", "label": "neutral"},
    ]

    return [
        {
            "sentence": get_random_sentence(train_reduced, emotion["category"]),
            "emotion": emotion["label"],
        }
        for emotion in emotions
    ]


def get_few_shot_prompt():

    examples = get_shots()

    example_template = """
    Text: {sentence}
    Emotion: {emotion}
    """

    example_prompt = PromptTemplate(
        input_variables=["sentence", "emotion"], template=example_template
    )

    prefix = """Below are examples of text paired with their corresponding emotions. 
    Your task is to classify the emotion of a new text based on these examples.
    Choose from the emotions: happy, sadness, anger, fear, disgust, surprise, neutral.
    """

    suffix = """
    Text: {sentence}
    Emotion:
    """

    prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["sentence"],
        example_separator="\n\n",
    )

    return prompt_template


def get_zero_shot_prompt():

    template = """Your task is to classify the primary emotion expressed in the following text snippet. 
    Choose from one of the following emotions: happy, sadness, anger, fear, disgust, surprise, neutral.
    Text: "{sentence}"
    Predicted Emotion: [Insert the predicted emotion here]
    
    Context: These snippets are transcriptions of spoken dialogue from videos. While the full context of the video may not be available, focus on the emotion conveyed by the words in the text.
    
    Carefully analyze the text and select the emotion that best matches the sentiment expressed. Provide the emotion that most accurately reflects the content and tone of the text.
    """

    prompt_template = PromptTemplate(input_variables=["query"], template=template)

    return prompt_template


def reduce_samples(df: pd.DataFrame, n_samples: int):
    # Agrupar os dados por categoria de emoção
    grouped = df.groupby("emotion_category")

    selected_samples = []

    # Para cada categoria de emoção, selecionar n_samples aleatoriamente
    for group_name, group_data in grouped:
        # Verificar se há mais amostras do que n_samples na categoria
        if len(group_data) >= n_samples:
            # Amostrar n_samples de cada grupo de forma estratificada
            sampled_group = group_data.sample(n=n_samples, random_state=42)
        else:
            # Se houver menos amostras do que n_samples, usar todas as amostras disponíveis
            sampled_group = group_data

        selected_samples.append(sampled_group)

    # Concatenar os resultados
    selected_df = pd.concat(selected_samples)

    return selected_df


if __name__ == "__main__":

    train, test, validation = get_data()

    # Reduzir o conjunto de test para 50 amostras de cada classe
    test_reduced = reduce_samples(test, n_samples=50)

    # Reduzir o conjunto de validation para 50 amostras de cada classe
    validation_reduced = reduce_samples(validation, n_samples=50)

    # Reduzir o conjunto de train para 50 amostras de cada classe
    train_reduced = reduce_samples(train, n_samples=50)

    HUGGING_FACE_API_KEY = "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"

    answers = []

    for model_name in [
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.3",
        # "tiiuae/falcon-7b-instruct",
        # "j-hartmann/emotion-english-distilroberta-base",
    ]:
        model = HuggingFaceClassifier(model_name, token=HUGGING_FACE_API_KEY)

        for prompt in ["zero_shot", "few_shot"]:

            zero_shot_prompt = get_zero_shot_prompt()
            few_shot_prompt = get_few_shot_prompt()

            for inquiry in validation_reduced["text_clean"].values:

                try:
                    zero_shot_prompt = zero_shot_prompt.format(sentence=inquiry)
                    few_shot_prompt = few_shot_prompt.format(sentence=inquiry)

                    if model_name == "facebook/bart-large-mnli":
                        candidate_labels = [
                            "happy",
                            "sadness",
                            "anger",
                            "fear",
                            "disgust",
                            "surprise",
                            "neutral",
                        ]
                        log = {
                            "model": model_name,
                            "sentence": inquiry,
                            "type_prompt": prompt,
                            "response": model.text_classification(
                                inquiry, candidate_labels
                            ),
                        }

                    else:
                        log = {
                            "model": model_name,
                            "sentence": inquiry,
                            "type_prompt": prompt,
                            "response": model.text_classification(inquiry),
                        }
                    answers.append(log)
                    print(log)
                    time.sleep(0.1)
                except:
                    continue

        csv_model_name = model_name.replace("/", "_")

        data_output = "/home/vinicius/repositories/ppgia/msc/data/output/zero_shot"
        archive_output = os.path.join(data_output, f"answers_{csv_model_name}.csv")

        pd.DataFrame(answers).to_csv(archive_output, sep=";", decimal=",")
