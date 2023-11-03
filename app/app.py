# -*- coding: utf-8 -*-

import os
import argparse
from typing import Dict, Any, Tuple
from flask import Flask, request


# HTTP response status codes
HTTP_OK = 200
HTTP_NOT_READY = 423
HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_ERROR = 500


app = Flask(__name__)


@app.errorhandler(500)
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
        error_response = {
            "label": "BODY_MISSING",
            "message": "Request doesn't have a body."
        }
        return error_response, HTTP_BAD_REQUEST

    @staticmethod
    def text_missing_error() -> Tuple[Dict[str, str], int]:
        error_response = {
            "label": "TEXT_MISSING",
            "message": "\"text\" missing from the request body."
        }
        return error_response, HTTP_BAD_REQUEST

    @staticmethod
    def invalid_type_error() -> Tuple[Dict[str, str], int]:
        error_response = {
            "label": "INVALID_TYPE",
            "message": "\"text\" is not a string."
        }
        return error_response, HTTP_BAD_REQUEST

    @staticmethod
    def text_empty_error() -> Tuple[Dict[str, str], int]:
        error_response = {
            "label": "TEXT_EMPTY",
            "message": "\"text\" is empty."
        }
        return error_response, HTTP_BAD_REQUEST


@app.route("/intent",  methods=["POST"])
def intent() -> Tuple[Dict[str, Any], int]:
    # Run sanity checks on request body
    content_type = request.headers.get('Content-Type')
    if content_type != 'application/json':
        return ErrorHandler.body_missing_error()

    request_data = request.get_json()
    if 'text' not in request_data:
        return ErrorHandler.text_missing_error()

    text = request_data['text']
    if not isinstance(text, str):
        return ErrorHandler.invalid_type_error()

    if not text:
        return ErrorHandler.text_empty_error()

    # Perform intent classification
    # TODO: run model inference, form response
    response = {
         "intents": [{
           "label": "flight",
           "confidence": 0.73
         }, {
           "label": "aircraft",
           "confidence": 0.12
         }, {
           "label": "capacity",
           "confidence": 0.03
         }],
    }
    return response, HTTP_OK


@app.route("/ready", methods=["GET"])
def ready():
    # TODO: Implement
    # if model.is_ready():
    #     return "OK", HTTP_OK
    # else:
    #     return "Not ready", HTTP_NOT_READY

    return "OK", HTTP_OK


def main():
    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('--model', type=str, required=True, help='Path to model directory or file.')
    arg_parser.add_argument('--port', type=int, default=os.getenv('PORT', 8080), help='Server port number.')
    args = arg_parser.parse_args()
    print("Port: ", args.port)
    app.run(host='0.0.0.0', port=args.port)
    # model.load(args.model)


if __name__ == '__main__':
    main()
