# -*- coding: utf-8 -*-

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any, Dict, Tuple

from flask import Flask, request

from intent_classifier import IntentClassifier

# HTTP response status codes
HTTP_OK = 200
HTTP_NOT_READY = 423
HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503


app = Flask(__name__)
model = IntentClassifier()


# Decorator to check if the model is ready
def model_ready_required(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if model.is_ready:
            return func(*args, **kwargs)
        else:
            response_error = {"label": "MODEL_NOT_READY", "message": "Model is loading, try later."}
            return response_error, HTTP_SERVICE_UNAVAILABLE

    return decorated_function


@app.errorhandler(HTTP_INTERNAL_ERROR)
def internal_error(error: Exception) -> Tuple[Dict[str, str], int]:
    error_message = getattr(error, "description", "An internal server error occurred.")
    error_response = {
        "label": "INTERNAL_ERROR",
        "message": error_message,
    }
    return error_response, HTTP_INTERNAL_ERROR


class ErrorHandler:
    @staticmethod
    def body_missing_error() -> Tuple[Dict[str, str], int]:
        error_response = {"label": "BODY_MISSING", "message": "Request doesn't have a body."}
        return error_response, HTTP_BAD_REQUEST

    @staticmethod
    def text_missing_error() -> Tuple[Dict[str, str], int]:
        error_response = {"label": "TEXT_MISSING", "message": '"text" missing from the request body.'}
        return error_response, HTTP_BAD_REQUEST

    @staticmethod
    def invalid_type_error() -> Tuple[Dict[str, str], int]:
        error_response = {"label": "INVALID_TYPE", "message": '"text" is not a string.'}
        return error_response, HTTP_BAD_REQUEST

    @staticmethod
    def text_empty_error() -> Tuple[Dict[str, str], int]:
        error_response = {"label": "TEXT_EMPTY", "message": '"text" is empty.'}
        return error_response, HTTP_BAD_REQUEST


@app.route("/intent", methods=["POST"])
@model_ready_required
def intent() -> Tuple[Dict[str, Any], int]:
    # Run sanity checks on request body
    content_type = request.headers.get("Content-Type")
    if content_type != "application/json":
        return ErrorHandler.body_missing_error()

    request_data = request.get_json()
    if "text" not in request_data:
        return ErrorHandler.text_missing_error()

    text = request_data["text"]
    if not isinstance(text, str):
        return ErrorHandler.invalid_type_error()

    if not text:
        return ErrorHandler.text_empty_error()

    # Perform intent classification
    predictions = model.predict(text)
    response = {"intents": predictions}

    return response, HTTP_OK


@app.route("/ready", methods=["GET"])
def ready():
    if model.is_ready:
        return "OK", HTTP_OK
    else:
        return "Not ready", HTTP_NOT_READY


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", type=str, required=True, help="Path to model directory.")
    arg_parser.add_argument("--top_k", type=int, default=3, help="Top-k predictions to return.")
    arg_parser.add_argument("--port", type=int, default=os.getenv("PORT", 8080), help="Server port number.")
    args = arg_parser.parse_args()
    # Load the model asynchronously
    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(lambda: model.load(args.model))
    # Run flask app
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
