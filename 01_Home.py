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

    if "valid_user" not in st.session_state:
        st.session_state.valid_user = False
        
@st.cache_resource
def get_resources(namespace):
    '''
    Initializes the customer modules
    '''
    vector_db = VectorDB(st.session_state.config)
    vector_db.create_retriever(namespace=namespace)
    vector_db.create_chain(k=1, return_source=True)

    return vector_db

@st.cache_data
def login(namespace):
    '''
    Side bar for login
    '''
    if namespace in st.secrets['users']:
        return True, namespace
    else:
        return False, None

def main():
    '''
    Main Function for streamlit interface
    '''
    # Load configs, logger, classes
    st.set_page_config(page_title="Conversational Medical Report Explainer")
    initialize_session_state()    
    if st.session_state.config['local']:
        logger = configure_logging('app.log')
    else: 
        logger = configure_logging(streaming=True)
    
    # ----------------------------------- SIDE BAR ------------------------------------------ #
    with st.sidebar:
        namespace = st.text_input('Enter username', value="")
        start = st.button('Start', type='primary')
        if start:
           st.session_state['valid_user'], namespace = login(namespace)
    
    #------------------------------------- MAIN PAGE -----------------------------------------#
    st.markdown("## :rocket: Conversational Medical Report Explainer")    
    st.caption('Created for Health Hack, by Jeremy, Mark, Kenny and Sien Long')
    st.caption(f"Powered by LLMs: {st.session_state.config['llm']}, {st.session_state.config['embedding_options']['model']}")
    if not st.session_state['valid_user']:
        st.warning('Enter valid username on the sidebar to begin', icon='⚠')
    else:               
        # Initialise vector db
        vector_db = get_resources(namespace)  

        # Button to clear history
        if st.button('Clear chat history', type='primary'):
            with st.status('Clearing chat history') as status:
                    logger.info(f"Saving conversation history before clearing:\n{st.session_state.messages}")
                    vector_db.clear_memory()
                    st.session_state.messages.clear()
                    status.update(label='Chat history cleared!', state='complete')

        with st.sidebar:
            if st.button('Get sample medical report', type='primary'):
                st.write(vector_db.sample_medical_report('healthhack_sample'))

        # Starting message
        with st.chat_message("assistant"):
            st.write("Hello 👋, please remember to clear the chat history first")
            st.write("Otherwise I might remember what we talked about (even if it is not on the screen!)")

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Enter question about something in the medical report."):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner():
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                answer = vector_db.query(prompt, return_source=True)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
                # st.markdown('Here's my source', icon='📚')

if __name__ == '__main__':
    main()

