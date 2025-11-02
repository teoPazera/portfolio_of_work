from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.in_memory import InMemoryDocstore
from langchain_core.runnables import ConfigurableField



DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an expert in sustainability reports analysis. 
    Based on the following question, generate a detailed, informative and consice document that addresses the question comprehensively.
    If the question asks about emissions it is likely the consice document should contain some markdown table.\n\n"
    Question: {question}\n\n"
    Hypothetical Document:""",
)

class HyDERetriever:
    def __init__(self, llm: BaseLanguageModel, vector_store: VectorStore, filter: dict[str,str]) -> None:
        """Class which acts as a Hypothetical document retriever, retriever which generates synthetic document 
        where answer to underlying question would lie

        Args:
            llm (BaseLanguageModel): LLM used in to generate the hypothetical document
            vector_store (VectorStore): vector store mainly chromadb object to use when retrieving text chunks
            filter (dict[str,str]): filter to use when retrieving the text chunks
        """
        self.llm = llm
        self.llm_chain = LLMChain(llm=self.llm, prompt=DEFAULT_QUERY_PROMPT)
        self.vector_store = vector_store
        self.filter = filter 

    def generate_hypothetical_document(self, question: str) -> str:
        """function which generates the Hypothetical document

        Args:
            question (str): question to which we want to generate hypothetical document

        Returns:
            str: Hypothetical docuemtn generated
        """
        response = self.llm_chain.invoke(question)
        return response["text"]

    def get_relevant_documents(self, question: str, k: int) -> list[Document]:
        """function to retrieve the relevant text chunks to the question

        Args:
            question (str): question to which we search relevant documents
            k (int): number of text chuks to be retrieved

        Returns:
            list[Document]: list of text chunks to return in Document wrapper
        """
        hypothetical_doc: str = self.generate_hypothetical_document(question)
        similar_docs: list[Document]
        if self.filter:
            similar_docs = self.vector_store.similarity_search(query=hypothetical_doc, filter=self.filter, k=k)
        else:
            similar_docs = self.vector_store.similarity_search(query=hypothetical_doc, k=k)    
        return similar_docs

class HybridRetriever:
    def __init__(self, vector_store: VectorStore, filter: dict, weights: list[float]=[0.5, 0.5]) -> None:
        """
        Retriever class which serves the purpose of finding relevant text chunk to given query
        it uses combination of vector similarity search and word matching(BM25)
        Args:
            vector_store (VectorStore): vector store fromw which to retrieve relevant text chunks
            filter (dict): filter to use when retrieving text chunks
            weights (list[float], optional): weights to set for reciproral rank fusion of bm25 retieval and vector similarity search. Defaults to [0.5, 0.5].
        """
        self.filter = filter
        self.vector_store = vector_store
        self.weights = weights
    

    def get_relevant_documents(self, question: str, k: int) -> list[Document]:
        """function to search for relevant text chunk, retrieving first 90 text chunks which are reranked using BM25 search

        Args:
            question (str): question to use when searching for relevant documents
            k (int): number of text chunks to retrieve

        Returns:
            list[Document]: list of text chunks to return in Document wrapper 
        """
        dense_retriever = self.vector_store.as_retriever(search_kwargs={'filter': self.filter, "k": 90})
        docs = dense_retriever.get_relevant_documents(query=question)
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k
        dense_retriever.search_kwargs={'filter': self.filter, "k": k}
        retriever = EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever],
                                       weights=self.weights, k=k)
        
        result_docs: list[Document] = retriever.invoke(input=question)
        return result_docs[:k]

    
if __name__ == "__main__":
    load_dotenv()
    # Initialize your LLM and vector store
    llm = AzureChatOpenAI(deployment_name="gpt-4",
                                    temperature=0,
                                    azure_endpoint=os.getenv("AZURE_ENDPOINT_COMPLETION")
        )
    embedding_model = AzureOpenAIEmbeddings(
            api_key=os.getenv('OPENAI_API_KEY'),
            azure_endpoint=os.getenv("AZURE_ENDPOINT_EMBEDDING"),
            model="text-embedding-3-large"
        )
    vectordb = Chroma(persist_directory="basic_m_enriched_vectors",
                               embedding_function=embedding_model)
    # docs = [
    # # Relevant documents
    # Document(page_content="To claim car insurance, you must report the accident, gather evidence, and submit a claim form to your insurer.", metadata={"category": "claims"}),
    # Document(page_content="You can file an insurance claim online by logging into your provider’s portal and submitting necessary documents.", metadata={"category": "claims"}),
    # Document(page_content="Most insurers require a police report, photos of the damage, and proof of policy to process a car insurance claim.", metadata={"category": "claims"}),
    # Document(page_content="Insurance claims for accidents typically take 7 to 14 days to process, depending on the complexity of the case.", metadata={"category": "claims"}),

    # # Irrelevant but still insurance-related documents
    # Document(page_content="Life insurance policies vary based on coverage types such as term life and whole life insurance.", metadata={"category": "policy"}),
    # Document(page_content="An actuary calculates insurance risks and determines premium amounts based on statistical models.", metadata={"category": "actuarial"}),
    # Document(page_content="Insurance fraud can lead to severe penalties, including fines and legal action.", metadata={"category": "fraud"}),
    # Document(page_content="The underwriting process in insurance involves assessing the risk of insuring an individual or business.", metadata={"category": "underwriting"}),
    # Document(page_content="Health insurance policies often cover preventive care, hospitalization, and prescription medications.", metadata={"category": "policy"}),
    # Document(page_content="Reinsurance helps insurance companies manage large claims by distributing the risk to other insurers. Car can be reinsured too Car car", metadata={"category": "reinsurance"}),
    # ]
    '''
    vector_store = Chroma.from_documents(documents=docs,
                               embedding=embedding_model)

    # Create an instance of HyDERetrieval
    hyde_retrieval = HyDERetriever(llm, vector_store, {})

    # Retrieve documents similar to the user's query
    similar_documents = hyde_retrieval.get_relevant_documents(question=query, k=4)
    print(similar_documents)
    '''
    hybrid_retrieval = HybridRetriever(vector_store=vectordb, filter={"company_name": "ABF"}, weights=[0.5, 0.5])
    query = "What are the scope 3 emmissions in the year 2021?"
    similar_documents = hybrid_retrieval.get_relevant_documents(question=query, k=6)
    print(len(similar_documents))