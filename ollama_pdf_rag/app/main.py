# based on: https://github.com/tonykipkemboi/ollama_pdf_rag/, https://www.youtube.com/watch?v=ztBJqzBU5kc
# This project is open source and available under the MIT License.
"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # Disables problematic inspection
import streamlit as st
import logging
import tempfile
import shutil
import pdfplumber
import ollama
import warnings
import json
from pathlib import Path
from typing import List, Tuple, Any
# Suppress torch warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')
warnings.filterwarnings('ignore', category=UserWarning, message='torch.classes')
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

#### Settings ####
# language of the pdf content
pdf_languages=["de"]
embed_model="nomic-embed-text:v1.5"
temperature=0
#### Settings End ####

# Set protobuf environment variable to avoid error messages
# This might cause some issues with latency but it's a tradeoff
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Define persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join("data", "vectors")
# Define history store file
DB_FILE = 'data/db.json'

avatars = {
    "assistant": "🤖",
    "user": "👤"
}

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info: Response from ollama.list()

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    try:
        # The new response format returns a list of Model objects
        if hasattr(models_info, "models"):
            # Extract model names from the Model objects
            model_names = tuple(model.model for model in models_info.models)
        else:
            # Fallback for any other format
            model_names = tuple()
            
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()

def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path, languages=pdf_languages)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    # Updated embeddings configuration with persistent storage
    embeddings = OllamaEmbeddings(model=embed_model)
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=Settings(allow_reset = True, is_persistent=True, anonymized_telemetry=False),
        collection_name=f"pdf_{hash(file_upload.name)}"  # Unique collection name per file
    )
    
    st.session_state["collection_name"]=f"pdf_{hash(file_upload.name)}"
    logger.info("Vector DB created with persistent storage")
    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    # Initialize LLM
    llm = ChatOllama(
        model=selected_model,
        temperature=temperature,
        validate_model_on_init=True,
        )
    
    # Query prompt template multiple versions
    #QUERY_PROMPT = PromptTemplate(
    #    input_variables=["question"],
    #    template="""You are an AI language model assistant. Your task is to generate 2
    #    different versions of the given user question to retrieve relevant documents from
    #    a vector database. By generating multiple perspectives on the user question, your
    #    goal is to help the user overcome some of the limitations of the distance-based
    #    similarity search. Provide these alternative questions separated by newlines.
    #    Original question: {question}""",
    #)
    #QUERY_PROMPT = PromptTemplate(
    #input_variables=["question"],
    #template="""Du bist ein KI Sprachmodell-Assistent. Deine Aufgabe ist, drei unterschiedliche Versionen
    #der Frage des Benutzers zu erstellen, die relevante Informationen aus den Dokumenten in der
    #Vektor-Datenbank enthalten. Durch die Generierung mehrerer Perspektiven auf die Benutzerfrage soll
    #dem Benutzer geholfen werden, einige der Einschränkungen der distanzbasierten
    #Ähnlichkeitssuche zu überwinden. Geben Sie diese alternativen Fragen durch Zeilenumbrüche getrennt an.
    #Original Frage: {question}""",
#)
    QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Du bist ein KI Sprachmodell-Assistent. Deine Aufgabe ist, die Frage des Benutzers präzise
    NUR mit den relevante Informationen aus dem Dokument in der
    Vektor-Datenbank zu beantworten.
    Original Frage: {question}""",
)

    # Set up retriever
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )
    # en
    # RAG prompt template
    #template = """Answer the question based ONLY on the following context:
    #context}
    #uestion: {question}
    #""
    # RAG prompt template
    
    template = """Beantworte die Frage NUR basierend auf dem folgenden Kontext:
    {context}
    Frage: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages
    
# fill selectbox with values from database
def populate_selectbox():
    with open(DB_FILE, 'r') as file:
        db = json.load(file)
    selectbox_data=[]
    for i, hist_prompt in enumerate(db.get('chat_history', [])):
        selectbox_data.append(hist_prompt[0]["content"])
    return selectbox_data

# set choosen value from selectbox
def selectbox_changed():    
    st.session_state.user_prompt=st.session_state.hist

# store prompt in database
def input_submit():    
    prompt_input=st.session_state.chat_input
    entry_found=False
    with open(DB_FILE, "r") as file:
        db = json.load(file)
    # search for duplicate        
    for i, hist_prompt in enumerate(db.get('chat_history', [])):
        if hist_prompt[0]["content"] == prompt_input:
            entry_found=True
    # store the new prompt in database
    if not entry_found:
        db_entry = [{
            "role": "user",
            "content": st.session_state.chat_input
        }]
        
        db['chat_history'].insert(0, db_entry)
        
        with open(DB_FILE, 'w') as file:
            json.dump(db, file)
        with open(DB_FILE, "r") as file:
            db = json.load(file)
        selectbox_data=[]
        for i, hist_prompt in enumerate(db.get('chat_history', [])):
            selectbox_data.append(hist_prompt[0]["content"])
        st.session_state.history_list = selectbox_data
    
def process_prompt(prompt: str, message_container, selected_model):
    try:
        # Add user message to chat
        st.session_state["messages"].append((prompt, True))
        
        # Process and display assistant response        
        with message_container:
            with st.spinner(":green[processing...]"):
                if st.session_state["vector_db"] is not None:
                    response = process_question(
                        prompt, st.session_state["vector_db"], selected_model
                    )
                    st.session_state["messages"].append((response, False))
                else:
                    st.warning("Please upload a PDF file first.")
        
        for i, (msg, is_user) in enumerate(st.session_state["messages"]):
            if is_user:
                message_container.chat_message("user",avatar=avatars["user"]).markdown(msg)        
            else:
                message_container.chat_message("assistent",avatar=avatars["assistant"]).markdown(msg)        
        
    except Exception as e:
        st.error(e, icon="⛔️")
        logger.error(f"Error processing prompt: {e}")
    else:
        if st.session_state["vector_db"] is None:
            st.warning("Upload a PDF file or use the sample PDF to begin chat...")    

def delete_vector_db():
    st.session_state.toggle = False
    st.session_state.toggle_key = 0
    if st.session_state["vector_db"] is not None:
        st.session_state["vector_db"].delete_collection()
        st.session_state["vector_db"]._client.reset()
        st.session_state["vector_db"]._client.clear_system_cache()

        st.session_state["vector_db"] = None
        st.session_state["pdf_pages"] = None
        st.session_state.messages = []
        st.session_state.user_prompt = ""
        st.session_state["file_names"] = ""
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        os.makedirs(PERSIST_DIRECTORY)
        logger.info("Vector DB removed from disk")

def process_file():
    uploader=f"file_uploader_{st.session_state.get('file_key')}" 
    if uploader in st.session_state:
        file_upload=st.session_state[uploader]
        if file_upload is not None:
            delete_vector_db()
            st.session_state["file_key"] += 1
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing uploaded PDF..."):
                    st.session_state["vector_db"] = create_vector_db(file_upload)
                    # Store the uploaded file in session state
                    st.session_state["file_upload"] = file_upload                    
                    # Extract and store PDF pages
                    with pdfplumber.open(file_upload) as pdf:
                        st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]

def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)
   
    # Get available models
    models_info = ollama.list()
    available_models = extract_model_names(models_info)
   
    # Initialize session state for controlling the visibility
    if 'clicked' not in st.session_state:
            st.session_state.clicked = False   
                        
    #render_sidebar()
    with st.sidebar:
        st.subheader("Chat history")
        
    if 'history_list' not in st.session_state:
        st.session_state['history_list'] = []
            
    # load history from db
    st.session_state.history_list=populate_selectbox() 
    
    selection = st.sidebar.selectbox('Choose an option:',
                                     options=st.session_state.history_list,
                                     on_change=selectbox_changed,
                                     key='hist',
                                     index=None,
                                     )
    
    delete_history=st.sidebar.button('Clear history', type="primary")
    
    if delete_history:
        # Clear chat history in db.json        
        db['chat_history'] = []
        with open(DB_FILE, 'w') as file:
            json.dump(db, file)
        # Clear chat messages in session state
        st.session_state.messages = []
        st.session_state.user_prompt = ""
        st.rerun()        

    # Create layout
    col1, col2 = st.columns([1.5, 2])
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False
    if "user_prompt" not in st.session_state:
        st.session_state["user_prompt"] = ""
    if "file_names" not in st.session_state:        
        st.session_state["file_names"] = ""

    # Model selection
    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", 
            available_models,
            key="model_select"
        )

    # Add toggle for sample PDF
    if "toggle" not in st.session_state:
        st.session_state.toggle = False  # Default to off
    if "toggle_key" not in st.session_state:
        st.session_state.toggle_key = 1

    use_sample = col1.toggle(
        "Use sample PDF (KI-im-Alltag-LinuxWelt)", 
        value=st.session_state.toggle,
        key=st.session_state["toggle_key"]
    )
    
    # Clear vector DB if switching between sample and upload
    if use_sample != st.session_state.get("use_sample"):
        if st.session_state["vector_db"] is not None:
            st.session_state["vector_db"].delete_collection()
            st.session_state["vector_db"]._client.reset()
            st.session_state["vector_db"]._client.clear_system_cache()
            st.session_state["vector_db"] = None
            st.session_state["pdf_pages"] = None
            st.session_state.user_prompt = ""
            st.session_state["file_names"] = ""
        st.session_state["use_sample"] = use_sample
    
    if use_sample:
        # Use the sample PDF
        sample_path = "data/pdfs/sample/KI-im-Alltag-LinuxWelt.pdf"
        if os.path.exists(sample_path):
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing sample PDF..."):
                    loader = UnstructuredPDFLoader(file_path=sample_path, languages=pdf_languages)
                    data = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                    chunks = text_splitter.split_documents(data)
                    st.session_state["vector_db"] = Chroma.from_documents(
                        documents=chunks,
                        embedding=OllamaEmbeddings(model=embed_model),
                        persist_directory=PERSIST_DIRECTORY,
                        client_settings=Settings(allow_reset = True, is_persistent=True, anonymized_telemetry=False),
                        collection_name="sample_pdf"
                    )
                    
                    # Open and display the sample PDF
                    with pdfplumber.open(sample_path) as pdf:
                        st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]

        else:
            st.error("Sample PDF file not found.")
    else:
        # Regular file upload with unique key
        if "file_key" not in st.session_state: st.session_state["file_key"] = 0
        file_upload = col1.file_uploader(
            "Upload a PDF file ↓", 
            type="pdf", 
            accept_multiple_files=False,
            on_change=process_file(),
            key=f"file_uploader_{st.session_state.get('file_key')}",
           
        )
        with col1:            
            file_names = st.empty()
            st.session_state["file_names"] = file_names
            if st.session_state["vector_db"] is not None:
                if "file_upload" in st.session_state:
                    file_names.markdown("**File uploaded:** " + st.session_state["file_upload"].name)
            else:
                file_names.markdown("")
            
    # Display PDF if pages are available
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        # PDF display controls
        zoom_level = col1.slider(
            "Zoom Level", 
            min_value=100, 
            max_value=500, 
            value=500, 
            step=50,
            key="zoom_slider"
        )

        # Display PDF pages
        with col1:
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

    # Delete collection button
    delete_collection = col1.button(
        "⚠️ Delete collection and vector db", 
        type="secondary",
        key="delete_button"
    )

    if delete_collection:
        delete_vector_db()
        st.rerun()

    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)
        
        if st.session_state.get('user_prompt'):
            use_history=True
        else:
            use_history=False  
        
        if use_history:
            prompt=st.session_state.user_prompt
            dummy = st.chat_input("Enter a prompt here...", key="chat_input", on_submit=input_submit)   
            process_prompt(prompt, message_container, selected_model)
            st.session_state.user_prompt=""
        else:
            st.session_state.user_prompt=""
            use_history=False
            if prompt := st.chat_input("Enter a prompt here...", key="chat_input", on_submit=input_submit):
                process_prompt(prompt, message_container, selected_model)

if __name__ == "__main__":
# if the DB_FILE chat-history not exists, create it
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w') as file:
            db = {
                'chat_history': []
            }
            json.dump(db, file)
    # load the database
    else:
        with open(DB_FILE, 'r') as file:
            db = json.load(file)
    main()
