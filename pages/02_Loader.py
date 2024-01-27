from modules.CustomLoader import CustomLoader
import streamlit as st
import yaml
import pinecone

def initialize_session_state():
    '''
    Handles initializing of session_state variables
    '''
    # Load config if not yet loaded
    if 'config' not in st.session_state:
        with open('config.yml', 'r') as file:
            st.session_state.config = yaml.safe_load(file)

@st.cache_resource
def get_resources():
    '''
    Initializes the customer modules
    '''
    pc = pinecone.Pinecone(api_key=st.secrets['PC_API_KEY'])
    loader = CustomLoader(st.session_state.config)
    loader.index_db(pc, 'healthhack')
    return loader

def main():
    st.set_page_config(page_title="Admin page")
    initialize_session_state()
    namespace = st.text_input('Enter user', value="")
    
    if namespace in st.secrets['users']:
        loader = get_resources()        
        weblinks = st.text_area(label = 'Retrieve from website (Enter every link on a new line)').split('\n')
        if st.button('Retrieve', type='primary'):
            with st.status('Retrieving texts and metadata', expanded=True) as status:
                try:
                    loader.load_webpages(weblinks)
                except Exception as e:
                    # logger.error('Exception during Splitting / embedding', exc_info=True)
                    status.update(label='Error occured.', state='error', expanded=False)
                else:
                    status.update(label='Splitting complete, expand to preview!', state='complete', expanded=False)
                    chunk_no = 0
                    for chunk, metadata in zip(loader.texts, loader.metadatas):
                        st.markdown(f'### Document {chunk_no}')
                        st.markdown(f'{chunk[:1000]}.....')
                        for field, info in metadata.items():
                            if field != 'text':
                                st.caption(f'{field}: {info}')
                        st.divider()
                        chunk_no +=1
        st.write(loader)
        if len(loader.texts) > 0:
            if st.button('Clear documents', type='secondary'):
                loader.clear_docs()
            docs_to_delete = st.multiselect('Documents to delete', [i for i in range(len(loader.texts))])
            if st.button('Delete items', type='secondary'):
                loader.delete_docs(docs_to_delete)
            if st.button('Upload', type='primary'):
                with st.status('Uploading', expanded=True) as status:
                    loader.upload_to_server(namespace)
                    st.write('Uploaded')
                    st.write(loader)
                    


if __name__ == "__main__":
    main()