from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import Pinecone
import pinecone
import logging.config
import os
import streamlit as st

logging.config.fileConfig(fname='../logging_config.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class VectorDB():
    '''
    Class object for retrieval and querying
    '''
    def __init__(self, config: dict):
        '''
        # Parameters
        -------------
        config : dictionary containing the following info
            chunk_size
            chunk_overlap
            chunk_separators
            embedding_model

        '''
        index_name = config['db_options']['index_name']
        if config['local']:
            pc_api_key = os.environ['PC_API_KEY']
            openai_api_key = os.environ['OPENAI_API_KEY']
        else:
            pc_api_key = st.secrets['PC_API_KEY']
            openai_api_key = os.environ['OPENAI_API_KEY']
        
        pc = pinecone.Pinecone(api_key=pc_api_key)
        self.index = pc.Index(index_name)
        self.embedding_function = OpenAIEmbeddings(
            deployment="SL-document_embedder",
            model=config['embedding_options']['model'],
            show_progress_bar=True)
        logger.info(f"\n{pc.describe_index(index_name)}")
        self.llm = ChatOpenAI(
            model_name=config['llm'],
            temperature=1,
            openai_api_key = openai_api_key
            )
        logger.info(f"llm model: {config['llm']}")
      
    def create_retriever(self, namespace):
        self.retriever = Pinecone(self.index, self.embedding_function, 'text', namespace=namespace)
        
    def create_chain(self):
        '''
        Creates the conversation chain from scratch
        '''

        # First need to summarise the history and create a standalone question using the LLM
        _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""

        self.CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(_template)

        _inputs = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(
                chat_history=lambda x: get_buffer_string(x["chat_history"])
            )
            | self.CONDENSE_QUESTION_PROMPT
            | self.llm
            | StrOutputParser(),
        )

        # Next part is similar to a normal RAG, the difference is in the context pipe, we will pipe in the standalone_question

        langchain_retriever = self.retriever.as_retriever(
            search_type="similarity", # mmr, similarity_score_threshold, similarity
            search_kwargs = {"k": 1}
        )

        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

        _context = {
            "context": itemgetter("standalone_question") | langchain_retriever ,
            "question": lambda x: x["standalone_question"],
        }

        # First we add a step to load memory
        # This adds a "memory" key to the input object
        self.memory = ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )

        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
        )

        self.final_chain = loaded_memory | _inputs | _context | ANSWER_PROMPT | self.llm
        logger.info('Chain created')
        
    def query(self, question):
        input = {'question': question}
        with get_openai_callback() as cb:
            result = self.final_chain.invoke(input)
        self.memory.save_context(input, {"answer": result.content})
        logger.info(result.content)
        logger.info(f"\n{cb}")
        return result.content
    
    def clear_memory(self):
        self.memory.clear()
        logger.info('Memory cleared')
        