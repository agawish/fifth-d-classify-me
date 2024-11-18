import json
import unittest
from unittest import mock

from fifth_d_classify_me.classifier import Classifier


class ClassifierTestSuite(unittest.TestCase):
    @mock.patch("fifth_d_classify_me.classifier.openai")
    def test_classify_with_singlelabel(self, mock_openai):
        mock_openai.OpenAI().chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(
            {
                "result": ["Y1"],
                "reasoning": "Because the query is agreeing with something that has been said.",
            },
        )
        ai_classifier = Classifier(
            open_ai_api_key="fake_key",
            temperature=0.3,
            top_p=0.3,
        )
        result = ai_classifier.classify(
            query="I'm craving for some chocolate ice cream today.",
            classes=[
                {
                    "class_id": "Y1",
                    "class_name": "Yes",
                    "class_description": "User responded with an affirmative",
                },
                {
                    "class_id": "N1",
                    "class_name": "No",
                    "class_description": "User responded with a negative",
                },
            ],
            options={
                "multilabel": False,
            },
        )
        assert result.result == ["Y1"]
        mock_openai.OpenAI().chat.completions.create.assert_called_once()
        mock_openai.OpenAI().chat.completions.create.assert_called_with(
            model="gpt-3.5-turbo",
            temperature=0.3,
            top_p=0.3,
            messages=mock.ANY,
            max_tokens=250,
            n=1,
        )

    @mock.patch("fifth_d_classify_me.classifier.openai")
    def test_classify_with_multilabel(self, mock_openai):
        mock_openai.OpenAI().chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(
            {
                "result": ["V1", "C1"],
                "reasoning": "Because the query is asking for two scoops of vanilla and a scoop of chocolate.",
            }
        )
        ai_classifier = Classifier(
            open_ai_api_key="fake_key",
            temperature=0.3,
            top_p=0.3,
        )
        result = ai_classifier.classify(
            query="I'm craving for some chocolate ice cream today.",
            classes=[
                {
                    "class_id": "C1",
                    "class_name": "Chocolate Ice Cream",
                    "class_description": "With real chocolate",
                },
                {
                    "class_id": "V1",
                    "class_name": "Vanilla Ice Cream",
                    "class_description": "Made with real Madagascan vanilla",
                },
            ],
            options={
                "multilabel": True,
            },
        )
        assert result.result == ["V1", "C1"]
        mock_openai.OpenAI().chat.completions.create.assert_called_once()
        mock_openai.OpenAI().chat.completions.create.assert_called_with(
            model="gpt-3.5-turbo",
            temperature=0.3,
            top_p=0.3,
            messages=mock.ANY,
            max_tokens=250,
            n=1,
        )

    @mock.patch("fifth_d_classify_me.classifier.openai")
    def test_classify_with_ai_hellucinating_a_class(self, mock_openai):
        mock_openai.OpenAI().chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(
            {
                "result": ["YN1"],
                "reasoning": "Because the query is agreeing with something that has been said.",
            },
        )
        ai_classifier = Classifier(
            open_ai_api_key="fake_key",
            temperature=0.3,
            top_p=0.3,
        )
        result = ai_classifier.classify(
            query="I'm craving for some chocolate ice cream today.",
            classes=[
                {
                    "class_id": "Y1",
                    "class_name": "Yes",
                    "class_description": "User responded with an affirmative",
                },
                {
                    "class_id": "N1",
                    "class_name": "No",
                    "class_description": "User responded with a negative",
                },
            ],
            options={
                "multilabel": False,
            },
        )
        assert result.result == []
        assert result.reasoning == ""
        mock_openai.OpenAI().chat.completions.create.assert_called_once()
        mock_openai.OpenAI().chat.completions.create.assert_called_with(
            model="gpt-3.5-turbo",
            temperature=0.3,
            top_p=0.3,
            messages=mock.ANY,
            max_tokens=250,
            n=1,
        )

    @mock.patch("fifth_d_classify_me.classifier.openai")
    def test_classify_with_ai_hellucinating_response(self, mock_openai):
        mock_openai.OpenAI().chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(
            {
                "result": "The answer is Yes because the query is agreeing to something he said",
            },
        )
        ai_classifier = Classifier(
            open_ai_api_key="fake_key",
            temperature=0.3,
            top_p=0.3,
        )
        result = ai_classifier.classify(
            query="I'm craving for some chocolate ice cream today.",
            classes=[
                {
                    "class_id": "Y1",
                    "class_name": "Yes",
                    "class_description": "User responded with an affirmative",
                },
                {
                    "class_id": "N1",
                    "class_name": "No",
                    "class_description": "User responded with a negative",
                },
            ],
            options={
                "multilabel": False,
            },
        )
        assert result.result == []
        assert result.reasoning == ""
        mock_openai.OpenAI().chat.completions.create.assert_called_once()
        mock_openai.OpenAI().chat.completions.create.assert_called_with(
            model="gpt-3.5-turbo",
            temperature=0.3,
            top_p=0.3,
            messages=mock.ANY,
            max_tokens=250,
            n=1,
        )

    if __name__ == "__main__":
        unittest.main()
