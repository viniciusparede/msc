from huggingface_hub import InferenceClient
from typing import Optional


class HuggingFaceClassifier:
    ENDPOINT = "https://api-inference.huggingface.co/models"

    INFERENCE_API_MODELS = {
        "meta-llama/Meta-Llama-3-70B-Instruct": "text_generation",
        "meta-llama/Meta-Llama-3-8B-Instruct": "text_generation",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "text_generation",
        "mistralai/Mistral-7B-Instruct-v0.3": "text_generation",
        # "openai-community/gpt2-large": "text_generation",
        "j-hartmann/emotion-english-distilroberta-base": "text_classification",
        "facebook/bart-large-mnli": "zero_shot_classification",
    }

    def __init__(
        self,
        model_name: str,
        token: str,
    ):
        self.model_name = model_name
        self.token = token

        if not model_name in self.INFERENCE_API_MODELS.keys():
            raise ValueError(
                f"Model {model_name} not available. Available models: {self.INFERENCE_API_MODELS}"
            )

        client = InferenceClient(self.model_name, token=self.token)
        self.client = client

    def text_classification(
        self, text: str, candidate_labels: Optional[list[str]] = None
    ) -> str:

        type_model = self.INFERENCE_API_MODELS[self.model_name]
        match type_model:
            case "text_generation":
                response = self.client.text_generation(
                    prompt=text,
                    temperature=0.02,
                    max_new_tokens=5,
                )

            case "text_classification":
                response = self.client.text_classification(text)

            case "zero_shot_classification":
                if candidate_labels is None:
                    raise ValueError(
                        "For zero_shot_classification, candidate_labels must be provided."
                    )
                response = self.client.zero_shot_classification(
                    text, candidate_labels, multi_label=False
                )

            case _:
                raise ValueError(f"Model {self.model_name} not available")

        if type_model in ["text_classification", "zero_shot_classification"]:
            predicted_class = response[0].label

        else:
            predicted_class = response

        match predicted_class:
            case "joy":
                return "happy"

        return predicted_class


# Exemplo de uso
if __name__ == "__main__":

    model = HuggingFaceClassifier(
        model_name="facebook/bart-large-mnli",
        token="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",
    )

    inquiry = "I hate this product!"

    emotions = ["happy", "sad", "anger", "fear", "disgust", "surprise", "neutral"]
    predicted_class = model.text_classification(inquiry, candidate_labels=emotions)
    print(f"Sentence: {inquiry} \nPredicted class: {predicted_class}")
