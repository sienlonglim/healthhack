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
import requests
logger = logging.getLogger(__name__)

class VectorDB():
    '''
    Class object for retrieval and querying
    '''
    def __init__(self, config: dict):
        '''
        ## Parameters
        -------------
        config : dictionary containing the following info
            chunk_size
            chunk_overlap
            chunk_separators
            embedding_model

        '''
        # Read config
        index_name = config['db_options']['index_name']
        embedding_model = config['embedding_options']['model']

        if config['local']:
            self.pc_api_key = os.environ['PC_API_KEY']
            openai_api_key = os.environ['OPENAI_API_KEY']
        else:
            self.pc_api_key = st.secrets['PC_API_KEY']
            openai_api_key = os.environ['OPENAI_API_KEY']
        
        # Create pinecone connection, embedding model, llm
        pc = pinecone.Pinecone(api_key=self.pc_api_key)
        self.index = pc.Index(index_name)
        self.embedding_function = OpenAIEmbeddings(
            deployment="SL-document_embedder",
            model=embedding_model,
            show_progress_bar=True)
        self.llm = ChatOpenAI(
            model_name=config['llm'],
            temperature=0.5,
            openai_api_key = openai_api_key
            )
        # Log info
        logger.info(f"Host: {pc.describe_index(index_name)['host']}")
        logger.info(f"llm model: {config['llm']}")
      
    def create_retriever(self, namespace):
        self.retriever = Pinecone(self.index, self.embedding_function, 'text', namespace=namespace)
        
    def create_chain(self, k: int=1, return_source: bool=False)-> None:
        '''
        Creates the conversation chain from scratch, chain consist of two parts:
        1. Chain to summarise chat history and new question to a standalone question
        2. Chain to combine standalone question with context (from semantic search) and query for answer
        ## Parameters
        ------------
        k : int
            number of documents to retrieve from vectorDB
        return_source : bool
            whether chain will return source documents
        '''

        # ------------ CHAIN 1 --------------------
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
            search_kwargs = {"k": k}
        )

        # ------------ CHAIN 2 --------------------
        # template = """Use the following piece of context to answer the user question. You should use only the facts from the context to answer.
        # If the context not contain the answer, just say that 'I don't know', don't try to make up an answer, use the context.
        # Don't address the context directly, but use it to answer the user question like it's your own knowledge.
        # {context}

        # Question: {question}
        # """

        template_medical_report = """Use the following piece of context to explain the medical report question in layman terms. You should use only the facts from the context to answer.
        If the context not contain the answer, just say that 'Sorry, I do not have the information to answer that.', don't try to make up an answer, use the context. 
        Do not give any diagnosis for the medical report. If the question asks for a diagnosis, just say that 'Sorry, I am not allowed to give any diagnosis.', don't give any diagnosis.
        Don't address the context directly, but use it to simply explain the medical report as if it's your own knowledge, remember to always use layman terms.

        Context:
        {context}

        Medical report question: {question}
        """
        ANSWER_PROMPT = ChatPromptTemplate.from_template(template_medical_report)

        _context = {
            "context": itemgetter("standalone_question") | langchain_retriever ,
            "question": lambda x: x["standalone_question"],
        }

        # ------------- Memory -------------------
        # First we add a step to load memory
        # This adds a "memory" key to the input object
        self.memory = ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )

        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
        )

        # -------------- FINAL CHAIN -----------
        # And finally, we do the part that returns the answers
        if not return_source:
            self.final_chain = loaded_memory | _inputs | _context | ANSWER_PROMPT | self.llm
        else:
            answer = {
                "answer": ANSWER_PROMPT | self.llm,
                "docs": itemgetter("context"),
            }
            self.final_chain = loaded_memory | _inputs | _context | answer
        logger.info('Chain created')
        
    def query(self, question: str, return_source: bool=False)-> str:
        '''
        Function to query the chain
        ## Parameters
        -------------
        question : str
            questions
        return_source : bool
            whether to return source metadata links
        ## Returns
        Answer (str)

        '''
        input = {'question': question}
        with get_openai_callback() as cb:
            result = self.final_chain.invoke(input)
        
        # Save and log question, answer and cost
        answer = result['answer'].content
        self.memory.save_context(input, {"answer": answer})
        logger.info(f"Question: {question}\nAnswer: {answer}")
        logger.info(f"\n{cb}")

        source = None
        if return_source and (answer not in ['Sorry, I do not have the information to answer that.', 'Sorry, I am not allowed to give any diagnosis.']):
            if result['docs']:
                source = result['docs'][0].metadata['source']
                logger.info(f'Source: {source}')
                answer = answer + f'\nHere is my source: {source}'
            else:
                logger.info('No source retrieved')
        return answer
    
    
    def clear_memory(self):
        self.memory.clear()
        logger.info('Memory cleared')
    
    def create_embeddings(self, query: str):
        '''
        Creates and returns embeddings (vectors) of an input
        ## Parameters
        --------------
        query (str): query to get the vector of
        '''
        return self.embedding_function.embed_documents(query)

    def sample_medical_report(self, sample):
        sample_dict = {}
        sample_dict['healthhack_sample'] = '''Medical Report:

US HEPATOBILIARY SYSTEM - ABOVE 16YRS Comparison is made with the study dated 19.1.23.
The liver is normal in size and shape. Its surface appears smooth. The liver parenchyma shows diffusely increased echogenicity. 
No focal hepatic lesion is detected. The gallbladder wall is not thickened and Murphy's sign is negative. 
A small echogenic focus is seen within the gallbladder (3mm). The biliary tree and common duct are not dilated. 
The visualized pancreas appears normal. The spleen is not enlarged and no focal lesion is seen in it. 
COMMENTS Fatty liver noted. A small echogenic focus seen within the gallbladder may be due to polyp or adherent soft stone.'''
        
        return sample_dict.get(sample)

    # def get_vector_list(self):
    #     '''
    #     API call to get index stats, but doesnt work
    #     '''
    #     url = "https://healthhack-2nyyyl3.svc.gcp-starter.pinecone.io/describe_index_stats"

    #     headers = {
    #         "accept": "application/json",
    #         "content-type": "application/json",
    #         "Api-Key": self.pc_api_key
    #     }
    #     response = requests.post(url, headers=headers)
    #     return response.text
    
    