import json
import os
import sys

from flask import Flask, request

from classifier import Classifier

app = Flask(__name__)


@app.route("/classify", methods=["POST"])
def classify():
    req = request.get_json(force=True)
    query = req["query"]  # str
    classes = req["classes"]  # list of dicts
    options = req["options"]  # dict
    classifier = Classifier(
        open_ai_api_key=os.environ.get("OPEN_AI_API_KEY", default=""),
        temperature=get_temperature(),
        top_p=get_top_p(),
    )
    result = classifier.classify(query, classes, options)
    print(f"result: {result}")
    return json.dumps(result)


def get_top_p():
    if len(sys.argv) >= 3:
        return float(sys.argv[2]) or 1.0
    return 1.0


def get_temperature():
    if len(sys.argv) >= 2:
        return float(sys.argv[1]) or 0.0
    return 0.0


if __name__ == "__main__":
    print(
        f"Running server with Temperature: {get_temperature()} and Top P: {get_top_p()}",
    )
    app.run(port=8000, debug=True)
