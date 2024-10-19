import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseInfoEvalModel(ABC):
    @abstractmethod
    def eval_info(self, summary, context, max_tokens=100):
        pass


class GPT3TurboInfoEvalModel(BaseInfoEvalModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def eval_info(self, summary, context, max_tokens=500):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Here is a summary of a long text and an excerpt from that text. \
                            Please evaluate whether the excerpt provides important information that is not included in the summary. \
                            Use an integer score from 1 to 4 to measure the amount of new information provided by the excerpt.\n \
                            Summary:\n \
                            {summary}\n \
                            Excerpt:\n \
                            {context}\n \
                            Please answer using Arabic numerals only 1, 2, 3, 4:"
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3InfoEvalModel(BaseInfoEvalModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def eval_info(self, summary, context, max_tokens=500):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Here is a summary of a long text and an excerpt from that text. \
                            Please evaluate whether the excerpt provides important information that is not included in the summary. \
                            Use an integer score from 1 to 4 to measure the amount of new information provided by the excerpt.\n \
                            Summary:\n \
                            {summary}\n \
                            Excerpt:\n \
                            {context}\n \
                            Please answer using Arabic numerals only 1, 2, 3, 4:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e