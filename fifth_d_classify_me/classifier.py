import json

import openai
from pydantic import BaseModel, ValidationError

from .classifier_prompts import ClassifierPrompts


class ClassifyRequest(BaseModel):
    """Request schema for the classify endpoint."""

    query: str
    classes: list[dict]
    options: dict


class ClassifierResponse(BaseModel):
    """Response from the classifier"""
    result: list[str]
    reasoning: str


class Classifier:
    """Classifier class"""
    client: openai.OpenAI

    def __init__(
        self,
        open_ai_api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

        # TODO: consider using langchain instead of openai to be able to use other models
        self.client = openai.OpenAI(api_key=open_ai_api_key)
        self.prompts = ClassifierPrompts()

    def is_result_data_valid_classes(
        self, classes: list[dict], result: list[str]
    ) -> bool:
        """Validate the result data against the classes"""
        if len(result) == 0 or len(classes) == 0:
            return True

        classes_ids = [item["class_id"] for item in classes]

        for a_class in result:
            if a_class not in classes_ids:
                return False
        return True

    def classify(
        self, request: ClassifyRequest
    ) -> ClassifierResponse:
        """Classify the query using the classes"""
        # PERF: caching: check a hash of input to return existing result with expiry however caching can be a problem if we cache without evaluating the query
        classes = request.classes
        options = request.options
        query = request.query
        system_prompt = (
            self.prompts.multilabel_prompt(classes)
            if options.get("multilabel")
            else self.prompts.singlelabel_prompt(classes)
        )
        # TODO: check response and use secure prompt libs, will incur a cost
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            top_p=self.top_p,
            n=1,
            temperature=self.temperature,
            max_tokens=250,
        )

        # NOTE: potentially run the result against another AI query in the chain to validate and enhance the response is correct and not just a partial match to the classes which could act as a tip, however it will incur a cost
        try:
            # Validate classes response first
            response = ClassifierResponse(
                # type: ignore
                **json.loads(response.choices[0].message.content))
            if self.is_result_data_valid_classes(classes, response.result):
                return response
            else:
                print(f"Error: Invalid classes response: {response.result}")
                return ClassifierResponse(result=[], reasoning="")
        except ValidationError as e:
            print(f"Error: {e}")
            return ClassifierResponse(result=[], reasoning="")
