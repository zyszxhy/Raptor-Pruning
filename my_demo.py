from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig, RetrievalAugmentation, \
                    BaseHypoQuestionModel, BaseInfoEvalModel, BaseFamiliarEvalModel

from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings

from pydantic import BaseModel, Field

import os
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = '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'
# os.environ['LANGCHAIN_PROJECT'] = 'quality_test_1'

class If_known(BaseModel):
    """Generate the bool answer"""
    answer: bool = Field(..., description="whether you know the title of this article")

class LanagchianFamiliarEval(BaseFamiliarEvalModel):
    def __init__(self, model):
        self.eval_model = model
        self.message = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "I am providing you with a summary of a long text and a specific excerpt from that text. \
                        What is the article title of the original text? \
                        If you know the title of the article from which this excerpt is taken, meaning you are familiar with the content of the original article, \
                        please assess how much important information this excerpt provides that you did not already know from the original.\n \
                        Use an integer score from 1 to 6 to measure the amount of new information provided by the excerpt.\n \
                        If you do not know the title of the article, answer '6'. If this excerpt provides no important information to you, answer '1'.\n\
                        Summary:\n \
                        {summary}\n \
                        Excerpt:\n \
                        {context}\n \
                        Please answer using Arabic numerals only 1, 2, 3, 4, 5, 6:"),

            # ("human", "The text below is an excerpt from an article. What the title of this article?\n \
            #             Text:\n \
            #             {context}"),
        ])
        # self.eval_chain = self.message | self.eval_model | StrOutputParser()
        self.eval_chain = self.message | self.eval_model.with_structured_output(InfoEvalScore)

    def eval_familiar(self, summary, context):   # summary, 
        return self.eval_chain.invoke({"summary":summary, "context":context})  # "summary":summary, 

class InfoEvalScore(BaseModel):
    """Generate the eval score."""
    score: int = Field(..., description="the eval score.")

class LangchainInfoEvalModel(BaseInfoEvalModel):
    def __init__(self, model):
        self.eval_model = model
        self.message = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "Here is a summary of a long text and an excerpt from that text. \
                        Please evaluate whether the excerpt provides important information that is not included in the summary. \
                        Use an integer score from 1 to 6 to measure the amount of new information provided by the excerpt.\n \
                        Summary:\n \
                        {summary}\n \
                        Excerpt:\n \
                        {context}\n \
                        Please answer using Arabic numerals only 1, 2, 3, 4, 5, 6:"),
        ])
        # self.eval_chain = self.message | self.eval_model | StrOutputParser()
        self.eval_chain = self.message | self.eval_model.with_structured_output(InfoEvalScore)

    def eval_info(self, summary, context):
        return self.eval_chain.invoke({"summary":summary, "context":context})


class HypotheticalQuestion(BaseModel):
    """Generate a hypothetical question."""
    question: str = Field(..., description="a question")

class LangchainHypoQuestionModel(BaseHypoQuestionModel):
    def __init__(self, model):
        self.chat_model = model
        self.message = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "Generate a hypothetical question for the following text \
                        that accurately and completely represents the meaning conveyed by the text: {context}:"),
        ])
        self.hypo_qs_chain = self.message | self.chat_model.with_structured_output(HypotheticalQuestion)
    
    def generate_hypo_qs(self, context:str, max_tokens):
        query = None
        while(query == None):
            query = self.hypo_qs_chain.invoke({"context":context})
        return query.question

class LangchainSummarizationModel(BaseSummarizationModel):
    def __init__(self, model):
        self.chat_model = model
        
        self.message = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "Write a summary of the following, including as many key details as possible: {context}:"),
        ])
        self.parser = StrOutputParser()
        self.summarize_chain = self.message | self.chat_model | self.parser
    
    def summarize(self, context:str, max_tokens):
        return self.summarize_chain.invoke({"context":context})


# TODO structure output choice number, avoid -6
class LangchainQAModel(BaseQAModel):
    def __init__(self, model):
        self.chat_model = model
        
        self.message = ChatPromptTemplate.from_messages([
            ("system", "You are Question Answering Portal."),
            # ("human", "Given Context: {context}\n Give the id of the best answer amongst the option to the question: {question}.\n \
            #  Options:\n \
            #  {options}\n \
            #  Please answer using Arabic numerals only 1, 2, 3, 4:"),

            ("human", "Using the folloing information \n{context}. \nAnswer the following question in less than 5-7 words, if possible: \n{question}"),
        ])
        self.parser = StrOutputParser()
        self.qa_chain = self.message | self.chat_model | self.parser
    
    def answer_question(self, context, question, options=None):    # for quality
        if options:
            return self.qa_chain.invoke({"context":context, "question":question, "options":options})
        else:
            return self.qa_chain.invoke({"context":context, "question":question})
    
    # def answer_question(self, context, question):   # for qasper
    #     return self.qa_chain.invoke({"context":context, "question":question})

class LangchainEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model):
        self.embed = model

    def create_embedding(self, text):
        return self.embed.embed_query(text)

class raptor_method:
    def __init__(self, text_path, text_tree, out_json):
        # summarization model
        # self.zhipu_model_sum = ChatZhipuAI(api_key="bc4bd9e0c76356b1ccd0be2f6bdde455.0WbiHJW92BLcV20J",
        #                         model="glm-4",
        #                         max_tokens=500)
        self.ollama_model_sum = ChatOllama(model="llama3.1",
                                  temperature=0.8,
                                  num_predict=500)
        self.summarization_model = LangchainSummarizationModel(model=self.ollama_model_sum)
        # sum_text = summarization_model.summarize('Right now we are passing a list of messages directly into the language model. \
        #                               Where does this list of messages come from? Usually, it is constructed from a combination of user input and application logic. \
        #                               This application logic usually takes the raw user input and transforms it into a list of messages ready to pass to the language \
        #                               model. Common transformations include adding a system message or formatting a template with the user input.')
        # print(sum_text)

        # hypo_question model
        self.hypo_qs_model = LangchainHypoQuestionModel(model=self.ollama_model_sum)

        # QA model
        # self.zhipu_model_qa = ChatZhipuAI(api_key="bc4bd9e0c76356b1ccd0be2f6bdde455.0WbiHJW92BLcV20J",
        #                         model="glm-4",
        #                         max_tokens=150)
        self.ollama_model_qa = ChatOllama(model="llama3.1",
                                          num_predict=100)
        self.qa_model = LangchainQAModel(model=self.ollama_model_qa)

        # info eval model
        self.info_eval_mdoel = LangchainInfoEvalModel(model=self.ollama_model_qa)

        # familiar eval model
        self.familiar_eval_model = LanagchianFamiliarEval(model=self.ollama_model_qa)

        # embeding model
        # self.zhipu_embed = ZhipuAIEmbeddings(
        #     model="embedding-2",
        #     api_key="bc4bd9e0c76356b1ccd0be2f6bdde455.0WbiHJW92BLcV20J"
        #     )
        self.ollama_embed = OllamaEmbeddings(model="nomic-embed-text:latest")
        self.embedding_model = LangchainEmbeddingModel(model=self.ollama_embed)

        self.RAC = RetrievalAugmentationConfig(summarization_model=self.summarization_model, 
                                               hypo_qs_model=self.hypo_qs_model, 
                                               info_eval_model=self.info_eval_mdoel,
                                               familiar_eval_model=self.familiar_eval_model,
                                               qa_model=self.qa_model, 
                                               embedding_model=self.embedding_model)
        
        if text_tree:
            self.RA = RetrievalAugmentation(config=self.RAC, tree=text_tree)
        else:
            self.RA = RetrievalAugmentation(config=self.RAC)
            if os.path.exists(text_path):
                with open(text_path, 'r') as file:
                    text = file.read()
                # construct the tree
                self.RA.add_documents(text, out_json)
                print('Load text from ' + text_path + '.')
            else:
                self.RA.add_documents(text_path, out_json)

    def answer(self, question, collapse_tree, options=None):
        return self.RA.answer_question(question, options, collapse_tree=collapse_tree)
    
    def save_tree(self, save_path):
        self.RA.save(save_path)


if __name__ == "__main__":
    model = raptor_method(text_path='/home/zhangyusi/raptor/demo/sample.txt',
                text_tree=None,
                out_json=None)
    
    question = "How did Cinderella reach her happy ending ?"
    print("Answer: ", model.answer(question))