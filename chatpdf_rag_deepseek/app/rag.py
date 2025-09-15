# based on https://github.com/paquino11/chatpdf-rag-deepseek-r1
import os
import shutil
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

set_debug(False)
set_verbose(False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
### Settings ###
#embedding_model="mxbai-embed-large"
embedding_model="nomic-embed-text:v1.5"
llm="deepseek-r1:latest"
language="de" # or "en"
res_cont=""
temperature=0
reasoning=True
# Define persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join("data", "chroma_db")
### Settings end ###

class LoggingHandler(BaseCallbackHandler):
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        print("Chat model started")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        global res_cont
        
        generations = response.generations
        response_message = generations[0][0].message
        ai_message_kwargs = response_message.model_dump()
        
        res_cont=ai_message_kwargs["additional_kwargs"]["reasoning_content"]
        self.queue.put(res_cont)    
        
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        print(f"Chain {serialized.get('name')} started")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print(f"Chain ended, outputs: {outputs}")

class ChatPDF:
    """A class for handling PDF ingestion and question answering using RAG."""

    def __init__(self, llm_model: str = llm, embedding_model: str = embedding_model):
        """
        Initialize the ChatPDF instance with an LLM and embedding model.
        """
        self.model = ChatOllama(model=llm_model, reasoning=reasoning, temperature=temperature)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        if language=="de":
            self.prompt = ChatPromptTemplate.from_template(
                """
                Du bist ein hilfsbereiter Assistent, der Fragen NUR auf Grundlage des hochgeladenen Dokuments beantwortet.
                Kontext:
                {context}
                
                Frage:
                {question}
                
                Beantworte die Frage auf deutsch prägnant, präzise und ausführlich mit Aufzählungspunkten.
                """
            )
        else:
            self.prompt = ChatPromptTemplate.from_template(
                """
                You are a helpful assistant answering questions based on the uploaded document.
                Context:
                {context}
                
                Question:
                {question}
                
                Answer concisely and accurately in three sentences or less.
                """
            )
        
        self.vector_store = None
        self.retriever = None

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF file, split its contents, and store the embeddings in the vector store.
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=Settings(allow_reset = True, is_persistent=True, anonymized_telemetry=False),
        )
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }
    
        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the LLM.")        
        callbacks = [LoggingHandler()]        
        output=chain.invoke(formatted_input,config={"callbacks": callbacks})
        result=[output, res_cont]
        return result        

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        if self.vector_store is not None:
            logger.info("Clearing vector store and retriever.")
            self.vector_store._client.reset()
            self.vector_store._client.clear_system_cache()
            self.vector_store = None
            self.retriever = None
            if os.path.exists(PERSIST_DIRECTORY):
                shutil.rmtree(PERSIST_DIRECTORY)
                os.makedirs(PERSIST_DIRECTORY)
                logger.info("Vector DB removed from disk")


