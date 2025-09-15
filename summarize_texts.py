# This script based on 
# https://github.com/debugverse/debugverse-youtube/tree/main/summarize_huge_documents_kmeans
# https://www.youtube.com/@DebugVerseTutorials
# https://www.youtube.com/watch?v=Gn64NNr3bqU
###
# https://github.com/mendableai/QA_clustering/blob/main/notebooks/clustering_approach.ipynb
#
# !install miniconda (https://www.anaconda.com)!
# mkdir -p ~/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm ~/miniconda3/miniconda.sh
# source ~/miniconda3/bin/activate
# conda init --all
# conda deactivate
#
# !prepare conda environment!
# conda create -n summarize python=3.12
# conda activate summarize
# pip install -r requirements.txt
# 
# !install ollama!
# wget https://ollama.com/install.sh
# sh install.sh
# Download models
# ollama pull llama3.2:latest
# ollama pull nomic-embed-text:latest

import numpy as np
import glob, sys, os
from pathlib import Path
import logging
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
import pandas as pd
import pyexcel as pe
import torch
import warnings
### ChatGPT ####
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)

# verbose output
set_debug(True)
set_verbose(True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##### Settings #####
## Also adjust the prompts to your needs (later in this script)##
## sys_template_str= and human_template_str ##
##
pdfs_folder="./input_files" # put your PDF oder TXT files here
output_folder = Path("output_rag") # Results folder
output_folder.mkdir(exist_ok=True) # create output folder
language="de" # language for prompts and output
chunk_size=1024
chunk_overlap=100
# ChatGPT
#chat_model="openai" # needs openai api key
# llm_model="gpt-5" 
#llm_model="gpt-3.5-turbo"  
#openai_api_key="<Your API key here>"
# ollama
chat_model="ollama" # self hosted model
# Replace with LLM of your choice
llm_model="llama3.2:latest"
# or
# llm_model = "llama3.1:8b"
# llm_model="deepseek-r1:7b"
embedding_model="nomic-embed-text:latest"
# or
# embedding_model = "BAAI/bge-m3"
# embedding_model = "BAAI/bge-base-en-v1.5"

##### Settings End #####

if chat_model=="openai":
    llm_model=llm_model
    temperature=0
else:    
    llm_model=llm_model 
    temperature=0
    embedding_model=embedding_model
    
def extract_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(pages)
    return texts

def extract_txt(file_path):
    loader = TextLoader(file_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(pages)
    return texts

def summarize_document_with_kmeans_clustering(type, file, llm, embeddings):
    if type=="pdf":
        texts = extract_pdf(file)
    if type=="txt":
        texts = extract_txt(file)

    print("Processing:", file)
    n_clusters = np.ceil(len(texts))
	
    print("N_Clusters: ", n_clusters)
    if n_clusters > 10:
        n_clusters=10
    print("num_cluster set to: ", n_clusters)
    filter = EmbeddingsClusteringFilter(embeddings=embeddings, num_clusters=n_clusters)
    if chat_model=="openai":
        template = """
            {system_prompt}
            {user_prompt}
        """  

    if (llm_model=="deepseek-r1:7b"):
        template = """
            <|begin of sentence|>
            <|User|>{user_prompt}<|Assistant|>{system_prompt}
            <|end of sentence|>
        """
                
    if (llm_model=="llama3.2:latest"):
        template = """
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            {system_prompt}
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {user_prompt}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """
    if (llm_model=="llama3.1:8b"):
        template = """
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            {system_prompt}
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {user_prompt}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """             
    if language=="de":
        sys_template_str = """
            Du bist ein hilfreicher Assistent und kannst Texte perfekt zusammenfassen. Antworte auf Deutsch oder übersetze die Antworten auf Deutsch.
        """
    else:
         sys_template_str = """
            You are a helpful assistant that summarizes a PDF. Analyse the document.
            """
    
    if language=="de":
        human_template_str = """
            Verfasse eine kurze Zusammenfassung des folgenden Textes auf DEUTSCH. Gib eine Antwort in Stichpunkten, die die wichtigsten Punkte des Textes abdecken.
            "{context}"
            KURZE ZUSAMMENFASSUNG AUF DEUTSCH:
        """
    else:
        human_template_str = """
            Write a concise summary of the following text. Give an answer in bullet points covering the most important points of the text.
            "{context}"
            CONCISE SUMMARY:
        """        

    prompt = PromptTemplate.from_template(template.format(system_prompt = sys_template_str, user_prompt = human_template_str))

    try:
        filter_result = filter.transform_documents(documents=texts)
        checker_chain = create_stuff_documents_chain(llm, prompt)
        summary = checker_chain.invoke({"context": filter_result}) 
        return file, summary
    except Exception as e:
        return str(e)


if chat_model=="openai":
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model=llm_model,
        temperature=temperature,
    )
    embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
else:
    llm = ChatOllama(
        model=llm_model,
        temperature=temperature,
        validate_model_on_init=True,
    )    

    embeddings = OllamaEmbeddings(
    model=embedding_model,
    validate_model_on_init=True,
    )

summaries = []
# get files from input folder
for file in sorted(glob.glob(pdfs_folder + "/*.*")):
    type = file.split('.')[-1].lower()
    if type == 'pdf' or type == 'txt':
        # the magic happens
        summary=summarize_document_with_kmeans_clustering(type, file, llm, embeddings)
        summaries.append(summary)
    else:
        print("Unsupported file type", file)


# Save all summaries into one .txt file
with open(output_folder / "summaries.txt", "w", encoding="utf-8") as f:
     for summary in summaries:
         f.write("# Dateiname: " + summary[0] + "\n")     
         f.write(summary[1] + "\n"*3)    

# Save all summaries into one .csv, .ods, .xslx file
# Comment the exports you don't need
df = pd.DataFrame(summaries, columns=["Filename", "Summary"])
csv_path=output_folder / "summaries.csv"
ods_path = output_folder / "summaries.ods" 
df.to_csv(csv_path, index=False) # to csv
records = pe.get_records(file_name=csv_path)
pe.save_as(records=records, dest_file_name=ods_path) # to ods
excel_path = output_folder / "summaries.xlsx"
df.to_excel(excel_path, index=False) # to xlsx

