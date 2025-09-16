# based on https://github.com/paquino11/chatpdf-rag-deepseek-r1
import os
import tempfile
import json
import time
import subprocess
from pathlib import Path
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

# Streamlit page configuration
st.set_page_config(
    page_title="RAG with Local DeepSeek R1",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define history store file
DB_FILE = 'data/db_deepseek.json'

avatars = {
    "assistant": "🤖",
    "user": "👤"
}

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
    use_history=True

# store prompt in database
def input_submit(prompt_input:str):
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
            "content": prompt_input
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

def display_messages(message_container, file_names):
    """Display the chat history."""
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        if is_user:
            message_container.chat_message("user",avatar=avatars["user"]).markdown(msg)        
        else:
            message_container.chat_message("assistent",avatar=avatars["assistant"]).markdown(msg)

    file_list=""
    for file in st.session_state["file_upload"]: 
        file_list = file_list + file.name + " "
    file_names.markdown("**File(s) loaded:** " + file_list + " ")

def process_input(prompt:str):
    """Process the user input and generate an assistant response."""
    user_text = prompt.strip()

    if len(user_text) > 0:
        input_submit(user_text)
        with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
            try:                
                agent_text = st.session_state["assistant"].ask(
                    user_text,
                    k=st.session_state["retrieval_k"],
                    score_threshold=st.session_state["retrieval_threshold"],
                )
            except ValueError as e:
                agent_text = str(e)

        st.session_state["messages"].append((user_text, True))
        
        if st.session_state["show_think"]:
            if agent_text[1] != "":
                st.session_state["messages"].append(("<think>" + agent_text[1] + "</think>", False))
        st.session_state["messages"].append((agent_text[0], False))


def read_and_save_file():
    """Handle file upload and ingestion."""
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    st.session_state.user_prompt=""

    #for file in st.session_state["file_uploader"]:
    for file in st.session_state["file_upload"]:
    
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}..."):
            t0 = time.time()
            st.session_state["assistant"].ingest(file_path)
            t1 = time.time()

        st.session_state["messages"].append(
            (f"Ingested {file.name} in {t1 - t0:.2f} seconds", False)
        )
        os.remove(file_path)

def page():
    """Main app page layout."""
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    st.subheader("🧠 RAG with Local DeepSeek R1", divider="gray", anchor=False)
    # Create layout
    col1, col2 = st.columns([1.5, 2])
    
    with col1:
        st.subheader("Upload a Document")
        st.session_state["ingestion_spinner"] = st.empty()
        if "file_key" not in st.session_state: st.session_state["file_key"] = 0
        file_upload = col1.file_uploader(
            "Upload a PDF file ↓", 
            type="pdf", 
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"file_uploader_{st.session_state.get('file_key')}",
        )
        if file_upload:
            st.session_state["file_key"] += 1
            st.session_state["file_upload"] = file_upload
            read_and_save_file()
        file_names = st.empty()

        # Retrieval settings
        st.subheader("Settings")
        st.toggle("Show 'think'", key="show_think", value=True)    
                
        st.session_state["retrieval_k"] = st.slider(
            "Number of Retrieved Results (k)",
            help="The parameter k specifies how many documents (or chunks) the retriever should return after the similarity has been calculated (taking into account score_threshold, if set).",
            min_value=1,
            max_value=10,
            value=5
        )
        st.session_state["retrieval_threshold"] = st.slider(
            "Similarity Score Threshold",
            help="The score_threshold parameter determines a minimum similarity value that a vector (or a document or chunk) must have in order to be returned in a query.",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05
        )

            # Clear chat
        if st.button("Clear Chat and Vector DB",type="secondary"):
            st.session_state["messages"] = []
            st.session_state["assistant"].clear()
            st.rerun()                
    #render_sidebar()
    with st.sidebar:
        st.subheader("Stored questions")
        
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
    
    delete_history=st.sidebar.button('Clear questions', type="primary")    
    
    if delete_history:
        # Clear chat history in db.json        
        db['chat_history'] = []
        with open(DB_FILE, 'w') as file:
            json.dump(db, file)
        st.session_state.user_prompt = ""
        st.rerun()        

    with col2:
        message_container = st.container(height=500, border=True)
        st.session_state["thinking_spinner"] = st.empty()
        
        if st.session_state.get('user_prompt'):
            use_history=True
        else:
            use_history=False 
        
        if use_history:
            dummy=st.chat_input("Enter a prompt here...", key="user_input")
            process_input(st.session_state.user_prompt)
            display_messages(message_container, file_names)
            st.session_state.user_prompt=""
        else:
            st.session_state.user_prompt=""
            use_history=False

            if prompt:=st.chat_input("Enter a prompt here...", key="user_input"):
                process_input(prompt)
                display_messages(message_container, file_names)

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
    page()
