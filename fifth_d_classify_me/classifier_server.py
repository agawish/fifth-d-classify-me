import os
import sys

from flask import Flask, request
from pydantic import ValidationError

from fifth_d_classify_me.classifier import Classifier, ClassifyRequest

app = Flask(__name__)


@app.route("/classify", methods=["POST"])
def classify():
    """Classify a query using OpenAI's GPT-3.5 API."""
    try:
        req = ClassifyRequest(**request.get_json(force=True))

        classifier = Classifier(
            open_ai_api_key=os.environ.get("OPEN_AI_API_KEY", default=""),
            model=get_model(),
            temperature=get_temperature(),
            top_p=get_top_p(),
        )
        result = classifier.classify(req)
        return result.model_dump_json()
    except ValidationError as e:
        print("Validation Error:", e)
        return {"error": str(e)}, 400


def get_top_p():
    """Get the top_p value from the command line arguments."""
    if len(sys.argv) >= 4:
        return float(sys.argv[3]) or 1.0
    return 1.0


def get_temperature():
    """Get the temperature value from the command line arguments."""
    if len(sys.argv) >= 3:
        return float(sys.argv[2]) or 0.0
    return 0.0


def get_model():
    """Get the model value from the command line arguments."""
    if len(sys.argv) >= 2:
        return sys.argv[1]
    return "gpt-3.5-turbo"


if __name__ == "__main__":
    print(
        f"Running server with Model: {get_model()} Temperature: {get_temperature()} and Top P: {
            get_top_p()}",
    )
    app.run(port=8000, debug=True)
