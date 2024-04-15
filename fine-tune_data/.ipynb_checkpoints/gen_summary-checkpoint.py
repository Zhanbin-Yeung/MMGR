from langchain.document_loaders import PyPDFLoader, ImageCaptionLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI # the LLM model we'll use (CHatGPT)
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.mapreduce import MapReduceChain
from langchain.docstore.document import Document
from langchain import PromptTemplate
import os
import openai
import time
import json

os.environ["OPENAI_API_KEY"] = "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"


if __name__ == "__main__":
    with open('./ed_data_for_ingestion.json', 'r') as file:
        data = json.load(file)
    print(type(data))
    print(len(data))
    
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
        
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    summary_chain = load_summarize_chain(llm, chain_type="stuff")
    
    summary_list = list()
    for row in range(len(data)):
        content = data[row][:16380]
        texts = text_splitter.split_text(content)
        docs = [Document(page_content=t) for t in texts]
        summary = summary_chain.run(docs)
        summary_list.append(summary)
        
    print(len(summary_list))
    with open("./ed_data_summary.json", 'w') as file:
        json.dump(summary_list, file)