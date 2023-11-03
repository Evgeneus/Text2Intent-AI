# -*- coding: utf-8 -*-
import json
import logging
import os
import string
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from transformers import AutoTokenizer

# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)


class IntentClassifier:
    def __init__(self):
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def load_tokenizer(self, model_path: str) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_path, config=model_path)
        return tokenizer

    def run_onnx_session(self, model_path: str) -> ort.InferenceSession:
        onnx_session = ort.InferenceSession(os.path.join(model_path, "model.onnx"))
        return onnx_session

    def load(self, model_path: str, top_k: int = 3) -> None:
        # Set top-k values
        self.top_k = top_k
        # Load id2label mapping
        config_path = os.path.join(os.path.join(model_path, "config.json"))
        with open(config_path, "r") as fp:
            config = json.load(fp)
        self.id2label = {int(key): value for key, value in config["id2label"].items()}
        # Load tokenizer
        self.tokenizer = self.load_tokenizer(model_path)
        # Load ONNX model and initialise session
        self.onnx_session = self.run_onnx_session(model_path)
        self.output_names = [output.name for output in self.onnx_session.get_outputs()]
        # Set property that the model is ready for inference
        self._is_ready = True

    def _transform_text(self, text: str) -> str:
        # Make text lower cased
        text = text.lower()
        # Remove punctuations
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        This method predicts user's intent from raw text.
        :param text: input text for classification
        :return: list of top-k predictions where each element is dict that contains
                the following entries:
                1) key is "label" (str), value is label representation (str)
                2) key is "confidence" (str), values is prob of the values (float)
        Example:
            input:
                text: "find all airfare from chicago to la for today"
            output:
                [
                    {'label': 'airfare', 'confidence': 0.99469936},
                    {'label': 'flight+airfare', 'confidence': 0.00120617},
                    {'label': 'ground_fare' , 'confidence': 0.00065020967},
                ]

        Note this implementation is CPU based, while we can easily extend it
        to GPU execution by installing onnx runtime-gpu and setting the CUDAExecutionProvider flag
        """
        # Transform text to follow the training data representation
        text = self._transform_text(text)

        # Generate inputs
        inputs = dict(self.tokenizer([text], return_tensors="np"))
        # Run single inference
        logits = self.onnx_session.run(self.output_names, inputs)[0][0]
        sofmax_dist = softmax(logits)

        # Get top-k class_id and probabilities, fast but idx not sorted
        top_k_class_ids = np.argpartition(sofmax_dist, -self.top_k)[-self.top_k :]
        top_k_probs = sofmax_dist[top_k_class_ids]
        # Sort the top_k_probs in descending order
        _sorted_prob_indices = np.argsort(top_k_probs)[::-1]
        top_k_class_ids = top_k_class_ids[_sorted_prob_indices]
        top_k_probs = top_k_probs[_sorted_prob_indices]

        # Form the output
        output = []
        for class_id, prob in zip(top_k_class_ids, top_k_probs):
            output.append(
                {
                    "label": self.id2label[class_id],
                    "confidence": float(prob),
                }
            )

        return output


if __name__ == "__main__":
    model_path = "models/all-mpnet-base-v2-tuned"
    top_k = 3
    classifier = IntentClassifier()
    classifier.load(model_path, top_k)
    output = classifier.predict("Show me the all flights from chicago today.")
    logging.info(output)
