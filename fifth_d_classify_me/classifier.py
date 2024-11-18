import json

import openai


class Classifier:
    client: openai.OpenAI

    def __init__(
        self,
        open_ai_api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        top_p: float = 0.3,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

        # TODO: consider using langchain instead of openai to be able to use other models
        self.client = openai.OpenAI(api_key=open_ai_api_key)

    def classify(self, query: str, classes: list[dict], options: dict) -> dict:
        # TODO: caching: check a hash of input to return existing result with expiry
        prompt_singlelabel = f"""
            You are a string classifier that can classify text based on a list of classes following the format:
            class_id: Identifier
            class_name: Name of the class
            class_description: Description and more information about the class

            For example, for the classes in a json format:
                "classes": [{{
                        "class_id": "Y1",
                        "class_name": "Yes",
                        "class_description": "User responded with an affirmative"
                }},
                {{
                        "class_id": "N1",
                        "class_name": "No",
                        "class_description": "User responded with a negative"
                }}
                  ]
            When the query is: "I consent to the processing of my personal data"
            Your answer should be: {{"result": ["Y1"], "reasoning": "because the query is agreeing to the processing of personal data."}}
            Remember you HAVE TO return a single class identifier in the result field, and the result field can only have one item, no empty list allowed.
            Here are the classes:
            {classes}
        """
        prompt_multilabel = f"""
            Given a set of classes, your role is to act as text classifier that can classify text based on a the provided list of classes. Each class follows the format:
            "class_id": Identifier
            "class_name": Name of the class
            "class_description": Description and more information about the class

            For example, for the classes:
                "classes": [
                {{
                    "class_id": "C1",
                    "class_name": "Chocolate Ice Cream",
                    "class_description": "With real chocolate"
                }},
                {{
                    "class_id": "V1",
                    "class_name": "Vanilla Ice Cream",
                    "class_description": "Made with real Madagascan vanilla"
                }}
                ]
            When the query is: "Can I get a scoop of vanilla and a scoop of chocolate?"
            Think about the classes provided step by step and consider the full details from the "class_name", and the hint given by the "class_description" field and then output the answer like the following: {{"result": ["V1","C1"], "reasoning": "Because the query is asking for a scoop of vanilla ice cream and a scoop of chocolate ice cream."}} and nothing else.
            Consider ALL the classes before answering the enquiry.
            IT IS VERY IMPORTANT TO ONLY RETURN THE VALUE OF "class_id" IN THE RESPONSE "result" field.
            IT IS EXTREMELY IMPORTANT TO RETURN THE RESULT IN A VALID JSON FORMAT WITH "result" (which can be an empty list) and "reasoning"(which can be an empty string) fields and NOTHING ELSE.
            Before returning the response, consider the reasoning and validate if it is correct, if it is not return an empty list and an empty string as the reasoning.
            Here are the classes in a json format:
            {classes}
            """
        system_prompt = (
            prompt_multilabel if options.get("multilabel") else prompt_singlelabel
        )
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

        # TODO: potentially run the result against an AI query to validate the response is correct and not just a partial match to the classes which could
        # act as a tip

        # TODO: check response and valiate json
        # TODO: check response and use secure prompt libs
        return json.loads(response.choices[0].message.content)  # type: ignore
