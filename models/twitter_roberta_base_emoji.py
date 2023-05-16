from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax

class TwitterRobertaBaseEmoji():
    def __init__(self):
        self.model_id = "cardiffnlp/twitter-roberta-base-emoji"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)

        self.labels=[
            "heart",
            "heart_eyes",
            "rolling_on_the_floor_laughing",
            "two_hearts",
            "fire",
            "blush",
            "sunglasses",
            "sparkles",
            "blue_heart",
            "kissing_heart",
            "camera",
            "us",
            "sunny",
            "purple_heart",
            "wink",
            "100",
            "grin",
            "christmas_tree",
            "camera_with_flash",
            "stuck_out_tongue_winking_eye",
        ]

    def query(self, prompt):
        encoded_input = self.tokenizer(prompt, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        return self.labels[ranking[-1]]

if __name__ == '__main__':
    emojiSelector = TwitterRobertaBaseEmoji()
    print(emojiSelector.query("Looking forward to Christmas"))
