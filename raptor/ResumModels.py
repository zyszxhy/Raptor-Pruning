import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseResumModel(ABC):
    @abstractmethod
    def Re_sum(self, summary, context, max_tokens=100):
        pass


class GPT3TurboResumModel(BaseResumModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def Re_sum(self, summary, context, max_tokens=500):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Here is a summary of a long text and an excerpt from that text. \
                            Carefully read both and identify the important additional information in the excerpt that is not mentioned in the summary. \
                            Concisely summarize this additional information.\n \
                            Summary:\n \
                            {summary}\n \
                            Excerpt:\n \
                            {context}:"
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3ResumModel(BaseResumModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def Re_sum(self, summary, context, max_tokens=500):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Here is a summary of a long text and an excerpt from that text. \
                            Carefully read both and identify the important additional information in the excerpt that is not mentioned in the summary. \
                            Concisely summarize this additional information.\n \
                            Summary:\n \
                            {summary}\n \
                            Excerpt:\n \
                            {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e