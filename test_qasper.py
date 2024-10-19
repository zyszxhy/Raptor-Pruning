import os
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = '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'
# os.environ['LANGCHAIN_PROJECT'] = 'quality_test'

import json
from tqdm import tqdm
from pydantic import BaseModel, Field

import torch
import gc

from langchain_chroma import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_ollama import OllamaEmbeddings, ChatOllama

from my_demo import raptor_method

#============= Langchian RAG Method =============
# zhipu_model = ChatZhipuAI(api_key="bc4bd9e0c76356b1ccd0be2f6bdde455.0WbiHJW92BLcV20J",
#                             model="glm-4",
#                             max_tokens=100)

# zhipu_embed = ZhipuAIEmbeddings(
#         model="embedding-3",
#         api_key="bc4bd9e0c76356b1ccd0be2f6bdde455.0WbiHJW92BLcV20J"
#         )

ollama_embed = OllamaEmbeddings(model="nomic-embed-text:latest")
ollama_model = ChatOllama(model="llama3.1")

def get_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Question Answering Portal."),
            ("human", "Using the folloing information \n{context}. \nAnswer the following question in less than 5-7 words, if possible: \n{question}"),
        ])
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": RunnableLambda(lambda x : x['question']) | retriever | format_docs, 
         "question": lambda x : x['question']
         }
        | prompt
        | ollama_model
        | StrOutputParser()
    )

    return rag_chain

def RAG_QA_answer_chain(model, query):
    result = model.invoke({"question":query})

    # del model
    # gc.collect()
    # torch.cuda.empty_cache()

    return result

#============= Sematic Spliter =============
def RAG_QA_model_sematic_spliter(article):
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20, encoding_name="cl100k_base")
    splits = text_splitter.split_text(article)
    vectorstore = Chroma.from_texts(texts=splits, embedding=ollama_embed)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

    return get_rag_chain(retriever)

#============= Hypothetical Queries =============
class HypotheticalQuestion(BaseModel):
    """Generate a hypothetical question."""

    question: str = Field(..., description="a question")

def RAG_QA_model_Hypo_question(article):
    import uuid

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_text(article)
    query_chain = (
        ChatPromptTemplate.from_template("Generate a hypothetical question for the following text \
                                         that accurately and completely represents the meaning conveyed by the text:\n{doc}")
        | ollama_model.with_structured_output(HypotheticalQuestion)
        # | RunnableLambda(lambda x: x.question)
    )

    # sum_chain = (
    #     ChatPromptTemplate.from_template("Write a summary of the following, including as many key details as possible:\n{doc}")
    #     | ollama_model
    #     | StrOutputParser()
    # )
    # queries = sum_chain.batch(splits, {"max_concurrency": 5})

    queries = []
    for i in range(len(splits)):
        query = None
        while(query == None):
            query = query_chain.invoke(splits[i])
        queries.append(query.question)
    
    vectorstore = Chroma(collection_name="hypo-questions",
                     embedding_function=ollama_embed)
    store = InMemoryByteStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs={'k': 5}
    )

    doc_ids = [str(uuid.uuid4()) for _ in splits]
    queries_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(queries)
    ]
    splits_docs = [Document(page_content=s, metadata={}) for s in splits]

    retriever.vectorstore.add_documents(queries_docs)
    retriever.docstore.mset(list(zip(doc_ids, splits_docs)))

    # del query_chain
    # gc.collect()
    # torch.cuda.empty_cache()

    return get_rag_chain(retriever)

#============= ColBERT =============
def RAG_QA_model_ColBERT(article):
    text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_text(article)

    from ragatouille import RAGPretrainedModel
    RAG = RAGPretrainedModel.from_pretrained("/mnt/e/Desktop/SFTP_files/colbertv2.0")
    RAG.index(
        collection=splits,
        index_name="Article",
        max_document_length=100,
        split_documents=True,
        bsize=5
    )
    retriever = RAG.as_langchain_retriever(k=5)
    
    return get_rag_chain(retriever)

#============= HyDE =============
def RAG_QA_model_HyDE(article):
    text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_text(article)

    from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
    from langchain.chains.llm import LLMChain
    prompt_template = """Please write a passage of less than 100 words to answer the question.
    Question: {question}
    Answer:"""
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
    llm_chain = LLMChain(llm=ollama_model, prompt=prompt)
    embeddings = HypotheticalDocumentEmbedder(llm_chain=llm_chain, base_embeddings=ollama_embed)

    vectorstore = Chroma.from_texts(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

    return get_rag_chain(retriever)

#============= Raptor =============
def RAG_QA_model_raptor(article, out_json):
    return raptor_method(text_path=article, text_tree=None, out_json=out_json)

def RAG_QA_answer_raptor(model, query):
    return model.answer(question=query, collapse_tree=True)


#============= Eval =============
def evaluate_rag_accuracy(dataset_json, out_json):

    with open(dataset_json, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    f.close()

    with open(out_json, 'a', encoding='utf-8') as output_f:
        for item in tqdm(data[137:], desc="Article"):
            article = item["article"]
            query_list = item["query_list"]

            model = RAG_QA_model_raptor(article, out_json.replace('.json', '_cnt_reduce.json'))
            # model = RAG_QA_model_sematic_spliter(article)

            for query_info in query_list:
                query = query_info["question"]                
                predicted_answer = RAG_QA_answer_raptor(model, query)
                # predicted_answer = RAG_QA_answer_chain(model, query)
                
                result = {
                    "question_id": query_info["question_id"],
                    "predicted_answer": predicted_answer,
                }
                output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_f.flush()
            
            # del model
            # gc.collect()
            # torch.cuda.empty_cache()

if __name__ == "__main__":
    evaluate_rag_accuracy(dataset_json='/home/zhangyusi/RAG_data/qasper/dummy_data/qasper-test-and-evaluator-v0.3/qasper-test-v0.3_extract.json',
                        out_json='/home/zhangyusi/raptor/output_result/qasper-test-v0.3_llama3.1_nomic-embed-text/raptor_llm-familiar-filter-6-1.json')