import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseFamiliarEvalModel(ABC):
    @abstractmethod
    def eval_familiar(self, summary, context, max_tokens=100):
        pass


class GPT3TurboFamiliarEvalModel(BaseFamiliarEvalModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def eval_familiar(self, summary, context, max_tokens=500):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"I am providing you with a summary of a long text and a specific excerpt from that text. \
                            Your task is to determine whether you are fully familiar with the content of the original text. \
                            Specifically, I want to know if you understand the story or topic well enough to identify the context and meaning of \
                            the provided excerpt within the full text. If you are completely familiar with the text and can clearly identify \
                            the excerpt's meaning and its connection to the full story, answer \"yes.\" \
                            If you are not familiar with the full text or unsure about the excerpt's context, answer \"no.\"\n \
                            Summary:\n \
                            {summary}\n \
                            Excerpt:\n \
                            {context}\n \
                            Please answer with \"yes\" or \"no\" only:"
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3FamiliarEvalModel(BaseFamiliarEvalModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def eval_familiar(self, summary, context, max_tokens=500):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"I am providing you with a summary of a long text and a specific excerpt from that text. \
                            Your task is to determine whether you are fully familiar with the content of the original text. \
                            Specifically, I want to know if you understand the story or topic well enough to identify the context and meaning of \
                            the provided excerpt within the full text. If you are completely familiar with the text and can clearly identify \
                            the excerpt's meaning and its connection to the full story, answer \"yes.\" \
                            If you are not familiar with the full text or unsure about the excerpt's context, answer \"no.\"\n \
                            Summary:\n \
                            {summary}\n \
                            Excerpt:\n \
                            {context}\n \
                            Please answer with \"yes\" or \"no\" only:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e