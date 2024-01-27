from modules.CustomLoader import CustomLoader
from modules.VectorDB import VectorDB
import streamlit as st
import yaml
import logging

@st.cache_resource
def configure_logging(file_path=None, streaming=None, level=logging.INFO):
    '''
    Initiates the logger, runs once due to caching
    '''
    # streamlit_root_logger = logging.getLogger(st.__name__)

    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if not len(logger.handlers):
        # Add a filehandler to output to a file
        if file_path:
            file_handler = logging.FileHandler(file_path, mode='a')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info('Added filer_handler')

        # Add a streamhandler to output to console
        if streaming:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            logger.info('Added stream_handler')
    

    return logger

def initialize_session_state():
    '''
    Handles initializing of session_state variables
    '''
    # Load config if not yet loaded
    if 'config' not in st.session_state:
        with open('config.yml', 'r') as file:
            st.session_state.config = yaml.safe_load(file)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
@st.cache_resource
def get_resources(namespace):
    '''
    Initializes the customer modules
    '''
    vector_db = VectorDB(st.session_state.config)
    vector_db.create_retriever(namespace=namespace)
    vector_db.create_chain()

    return vector_db

def main():
    '''
    Main Function for streamlit interface
    '''
    # Load configs, logger, classes
    st.set_page_config(page_title="Conversation RAG Bot")
    initialize_session_state()    
    if st.session_state.config['local']:
        logger = configure_logging('app.log')
    else: 
        logger = configure_logging(streaming=True)
    
    namespace = st.text_input('Enter user', value="")
    
    if namespace in st.secrets['users']:
        vector_db = get_resources(namespace)  
        if st.button('Clear chat history', type='primary'):
            vector_db.clear_memory()

        #------------------------------------- MAIN PAGE -----------------------------------------#
        st.markdown("## :rocket: Health Hack: Conversation RAG Bot")    
        st.caption('by Jeremy, Mark, Kenny and Sien Long')
        with st.chat_message("assistant"):
            st.write("Hello 👋, please remember to clear my history first, otherwise I will remember what we talked about (even if it is not on the screen hahaha)")

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What is diabetes?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            answer = vector_db.query(prompt)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
        
        # st.info('Most recent source used', icon='📚')
        # for field in metadata:
        #     st.write()
        #     st.write('-----------------------------------')

if __name__ == '__main__':
    main()

