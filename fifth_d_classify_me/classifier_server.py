import os
import sys

from flask import Flask, request

from fifth_d_classify_me.classifier import Classifier

app = Flask(__name__)


@app.route("/classify", methods=["POST"])
def classify():
    """Classify a query using OpenAI's GPT-3.5 API."""
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
    return result.model_dump_json()


def get_top_p():
    """Get the top_p value from the command line arguments."""
    if len(sys.argv) >= 3:
        return float(sys.argv[2]) or 1.0
    return 1.0


def get_temperature():
    """Get the temperature value from the command line arguments."""
    if len(sys.argv) >= 2:
        return float(sys.argv[1]) or 0.0
    return 0.0


if __name__ == "__main__":
    print(
        f"Running server with Temperature: {get_temperature()} and Top P: {
            get_top_p()}",
    )
    app.run(port=8000, debug=True)
