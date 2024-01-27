import pinecone
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4
import streamlit as st
import logging.config
import re
import os
logger = logging.getLogger(__name__)


class CustomLoader():
    '''
    Object class that handles loading of webpages and uploading to PineCone vectorDB
    '''
    def __init__(self, config: dict)-> None:
        '''
        # Parameters
        -------------
        config : dictionary containing the following info
            chunk_size
            chunk_overlap
            chunk_separators
            embedding_model

        '''
        self.vector_db = None
        if config['local']:
            openai_api_key = os.environ['OPENAI_API_KEY']
        else:
            openai_api_key = st.secrets['OPENAI_API_KEY']
            
        self.embedding_function = OpenAIEmbeddings(
            deployment="SL-document_embedder",
            model=config['embedding_options']['model'],
            show_progress_bar=True,
            openai_api_key= openai_api_key)
        
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=config['splitter_options']['chunk_size'],
                    chunk_overlap=config['splitter_options']['chunk_overlap'],
                    separators = config['splitter_options']['chunk_separators']
                    )
        self.clean_regex = config['splitter_options']['clean_regex']
        self.regex_patterns = config['splitter_options']['regex_patterns']

        self.texts = []
        self.metadatas = []
        logger.info(f'Initialized loader, current no. of chunks: {len(self.texts), len(self.metadatas)}')

    def __str__(self) -> str:
        '''Returns the length of currently loaded data and metadata, both should match'''
        return f'Texts: {len(self.texts)}, Metadatas: {len(self.metadatas)}'

    def load_webpages(self, url)-> None:
        '''
        Function to load webpage and split into raw texts with metadata
        # Parameters
        ------------
        url : str or list of strings containing urls
        '''
        loader = WebBaseLoader(url)
        for idx, record in enumerate(loader.load()):
            # Remove messy delimiters first:
            if self.clean_regex:
                for regex_pattern in self.regex_patterns:
                    record.page_content = re.sub(regex_pattern, ' ', record.page_content)
                    
            # Extract metadata, split text and append to main list to upload
            metadata = record.metadata
            record_texts = self.splitter.split_text(record.page_content)
            record_metadatas = [{"chunk": chunk_num, "text": text, **metadata} for chunk_num, text in enumerate(record_texts)]
            self.texts.extend(record_texts)
            self.metadatas.extend(record_metadatas)
            logger.info((len(self.texts), len(self.metadatas)))

    def index_db(self, vector_db: pinecone.Pinecone, index_name: str)-> None:
        '''
        Function to index the vector database
        # Parameters
        -------------
        vector_db (str) : vector_db instance
        index_name (str) : name of index
        '''
        self.index = vector_db.Index(index_name)
        logger.info(f'Index called: {index_name}')
        # logger.info(vector_db.describe_index(index_name))
        

    def upload_to_server(self, namespace : str)-> None:
        '''
        Function to upload any loaded data and metadata to the vector_db index, clears the uploaded text and metadata from instance once completed
        # Parameters
        ------------
        namespace (str) : a namespace for partitioning items in the index
        '''
        ids = [str(uuid4()) for _ in range(len(self.texts))]
        embeddings = self.embedding_function.embed_documents(self.texts)
        self.index.upsert(vectors=zip(ids, embeddings, self.metadatas), namespace=namespace)
        self.texts.clear()
        self.metadatas.clear()
    
    def clear_docs(self):
        '''
        Clears all currently loaded data
        '''
        self.texts.clear()
        self.metadatas.clear()

    def create_embeddings(self):
        '''
        Creates and returns all embeddings (vectors) of current texts
        '''
        return self.embedding_function.embed_documents(self.texts)
    
    def delete_docs(self, docs_to_delete: list):
        '''
        Deletes specified texts and corresponding metadata
        # Parameters
        -------------
        docs_to_delete (list): a list of indices to delete
        '''
        for idx in docs_to_delete:
            del self.texts[idx]
            del self.metadatas[idx]
