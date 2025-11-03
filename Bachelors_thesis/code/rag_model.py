import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from utils.prompts import *
from utils.token_counter import count_tokens
from parsing_model import Parser
from langchain_core.documents.base import Document
from KG_entity_types_prompts import PREDICTION_PROMPT, COT_PREDICTION_PROMPT
import json
from networkx.readwrite import json_graph
import tiktoken
from retrievers import HybridRetriever, HyDERetriever
from typing import Literal
from rerankers import Rerankers


class RAG:
    vecdb_directory: str
    embedding_model: AzureOpenAIEmbeddings
    llm: AzureChatOpenAI
    chunk_documents: list[Document]
    vectordb: Chroma
    parser: Parser
    def __init__(self) -> None:
        """RAG class which serves as a model in sustainability reports automation
        flexible usage with multiple heuristics 
        """
        load_dotenv() # loading keys to be able to use LLMs
        self.embedding_model = AzureOpenAIEmbeddings(
            api_key=os.getenv('OPENAI_API_KEY'),
            azure_endpoint=os.getenv("AZURE_ENDPOINT_EMBEDDING"),
            model="text-embedding-3-large"
            )

        self.llm = AzureChatOpenAI(deployment_name="gpt-4o-mini-20240718",
                                    temperature=0.1,
                                    azure_endpoint=os.getenv("AZURE_ENDPOINT_COMPLETION")
        ) # defining llm model to be used to answer queries that are under 16000 tokens
        self.llm_w_json = AzureChatOpenAI(deployment_name="gpt-4o-mini-20240718",
                                    temperature=0.1,
                                    azure_endpoint=os.getenv("AZURE_ENDPOINT_COMPLETION"),
                                    model_kwargs={"response_format": {"type": "json_object"}}
        )
        self.filter = dict()
        self.rerankers = Rerankers()
        self.tokens = 0

    def load_db(self, company_name: str, vec_db_path: str) -> None:
        """method to load the vector database, must be called everytime after creating an instance
        Args: 
            company_name (str): name of the company for which we want to fill out the report
            vec_db_path (str): path to chroma db vector database
        """
    
        self.vectordb = Chroma(persist_directory=vec_db_path,
                               embedding_function=self.embedding_model)
        self.filter = {"company_name": company_name} 
        if vec_db_path == "KG_vectors":                  
            with open(f"KGs\graph_{company_name}_final.json", "r") as f:
                data = json.load(f)

            # Convert JSON data to a NetworkX graph
            self.graph = json_graph.node_link_graph(data)
        

 

    def answer_query(self, query: str, form_of_answer: str, explanation: str, k: int, 
        heuristic: list[Literal["basic","KG", "HyDE", "Hybrid", "RerankGPT", "RerankPeripheral", "RerankCrossEncoder", "CoT"]]) -> dict:
        """function used to answer to a question on the report with the ability to use multiple heuristics at once
        Args:
            query: str -> input question that is used to find find relevant context
            form_of_answer: str -> predefined format we specify to the llm in which it should answer us
            explanation: str -> extra context for the llm to understand given query better
            heuristic: list[Literal[...]] -> options to use some of the heuristic while answering query
        
        Returns:
            dict -> answer to given question and context used to answer it

        """
        if "KG" in heuristic:
            return self.answer_query_kg(query=query, form_of_answer=form_of_answer,
                        explanation=explanation, topk_nodes=k, heuristic=heuristic)
        if "basic" in heuristic:
            retriever = self.vectordb.as_retriever(search_kwargs={'filter': self.filter, "k":k})
        elif "HyDE" in heuristic:
            retriever = HyDERetriever(llm=self.llm, vector_store=self.vectordb, filter=self.filter)
        elif "Hybrid" in heuristic:
            # add kwargs to for example change weights in hybrid
            retriever = HybridRetriever(vector_store=self.vectordb, filter=self.filter, weights=[0.5,0.5])

        relevant_documents: list[Document]
        if "RerankGPT" in heuristic:
            if "basic" in heuristic:
                retriever.search_kwargs = {'filter': self.filter, "k":30}
            relevant_documents = retriever.get_relevant_documents(query, k=30)
            m = -k + 12
            relevant_documents, tokens = self.rerankers.gpt(docs=relevant_documents, query=query, n=12, m=m)
            self.tokens += tokens

        elif "RerankCrossEncoder" in heuristic:
            if "basic" in heuristic:
                retriever.search_kwargs = {'filter': self.filter, "k":30}
            relevant_documents = retriever.get_relevant_documents(query, k=30)
            relevant_documents = self.rerankers.cross_encoder(docs=relevant_documents, query=query, k=k)

        else:
            relevant_documents = retriever.get_relevant_documents(query, k=k)

        print(len(relevant_documents), heuristic) 
        if "RerankPeripheral":
            relevant_documents = self.rerankers.peripheral(docs=relevant_documents)

        context: str = "\n---\n".join([d.page_content for d in relevant_documents])
        if "CoT" in heuristic:
            prompt = CoT_JSON_prompt.replace("{question}", query)
        else:
            prompt = basic_prompt.replace("{question}", query)
        
        prompt = prompt.replace("{form_of_answer}", form_of_answer)
        prompt = prompt.replace("{context}", context)
        prompt = prompt.replace("{explanation}", explanation)
        tok_count: int = count_tokens(prompt)
        #print(tok_count)
        self.tokens += tok_count
        
        if tok_count < 150000:
            if "CoT" in heuristic:
                answer = self.llm_w_json.invoke(prompt)
                try: 
                    json_dict = json.loads(answer.content)
                    out = {"answer": json_dict["Answer"], "context": context}

                except: 
                    out = {"answer": answer.content, "context": context}

            else:
                answer = self.llm.invoke(prompt)
                out = {"answer": answer.content, "context": context}
            self.tokens += count_tokens(answer.content)
        else:
            print("too many tokens needed to be used model would overhault")
            out =  {"answer": "", "context": context}
        return out

        

    def answer_query_kg(self, query: str, form_of_answer: str, explanation: str, 
                  heuristic: list[Literal["RerankGPT", "CoT"]], 
                  topk_nodes: int=12, topk_internal_rel: int=12, topk_external_rel: int=12) -> dict:     
        """function to generate answer to given question using Graph RAG approach

        Args:
            query (str): input question that is used to find find relevant context
            form_of_answer (str): desired format of the output 
            explanation (str): additional context provided to better understand given question
            heuristic (list[Literal[&quot;RerankGPT&quot;, &quot;CoT&quot;]]): additional heuristics that can be used
            topk_nodes (int, optional): number of nodes to retrieve initialy. Defaults to 12.
            topk_internal_rel (int, optional): number of strongest internal relationships to find when searching from the retrieved nodes. Defaults to 12.
            topk_external_rel (int, optional): number of strongest external relationshipts to find outside of the 12 initial nodes. Defaults to 12.

        Returns:
            dict -> answer to given question and context used to answer it

        """
        nodes = self.vectordb.similarity_search(query=query, k=topk_nodes, filter=self.filter)
        
        chunks: list[str] = []
        for i, node in enumerate(nodes):
            try:
                chunks.extend(self.graph.nodes[node.metadata["source"]]['chunks'])
            except:
                print(i, 1)
                pass

        chunks_selected: list[str] = list(set(chunks))
        if "RerankGPT" in heuristic:
            # when reranking chunks just select top 12 
            n = 24 # 24
            m = 12 # -topk_nodes + n 12
            # num of nodes on output = n-m
            before = len(chunks_selected)
            chunks_selected, tokens = self.rerankers.gpt(docs=chunks_selected, query=query, n=n, m=m, g_RAG=True)
            self.tokens += tokens
            after = len(chunks_selected)
            print(before, after)
        print(len(chunks_selected))

        # getting top k internal and external relationships 
        within_relationships = []
        between_relationships = []
        nodes_info = []
        nodes_set = set([x.metadata["source"] for x in nodes])

        for node1, node2, data in self.graph.edges(data=True):
            relationship_info = {
                "start": node1,
                "end": node2,
                "description": data["relationship"],
                "score": data["score"]
            }
            if node1 in nodes_set and node2 in nodes_set:
                within_relationships.append(relationship_info)
            elif (node1 in nodes_set and node2 not in nodes_set) or (node2 in nodes_set and node1 not in nodes_set):
                between_relationships.append(relationship_info)

        within_relationships = sorted(within_relationships, key=lambda x: x["score"], reverse=True)[:topk_internal_rel]
        between_relationships = sorted(between_relationships, key=lambda x: x["score"], reverse=True)[:topk_external_rel]

        all_nodes = set()
        relationships = within_relationships + between_relationships

        for rel in relationships:
            all_nodes.add(rel["start"])
            all_nodes.add(rel["end"])

        for node in all_nodes:
            if node in self.graph.nodes:
                node_data = self.graph.nodes[node]
                node_info = {
                    "id": node,
                    "description": node_data["description"]
                }
                nodes_info.append(node_info)

        relationships_selected = {
            "nodes": nodes_info,
            "relationships": relationships
        }

        # getting immediate summaries
        summaries_selected = []
        for i, node in enumerate(nodes):
            try:
                summaries_selected.append(self.graph.nodes[node.metadata["source"]]['community_summaries'][0])
            except KeyError:
                print(i,2)
        # generating prompt
        
       
        context = "CHUNK TEXT: \n" + "\n".join(chunks_selected) + \
                "\n\nNODES: \n" + str(relationships_selected["nodes"]) + \
                "\n\nRELATIONSHIPS: \n" + str(relationships_selected["relationships"]) + \
                "\n\nCOMMUNITY SUMMARIES: \n" + str(summaries_selected)
        
        if "CoT" in heuristic:
            prompt = COT_PREDICTION_PROMPT.replace("{question}", query)
        else:
            prompt = PREDICTION_PROMPT.replace("{question}", query)

        prompt = prompt.replace("{context}", context)
        prompt = prompt.replace("{form_of_answer}", form_of_answer)
        prompt = prompt.replace("{explanation}", explanation)
        tokens_used = count_tokens(prompt)
        self.tokens += tokens_used
        print(tokens_used)

        # generating response

        if "CoT" in heuristic:
            answer = self.llm_w_json.invoke(prompt)
            try: 
                json_dict = json.loads(answer.content)
                out = {"answer": json_dict["Answer"], "context": context}

            except: 
                out = {"answer": answer.content, "context": context}

        else:
            answer = self.llm.invoke(prompt)
            out = {"answer":answer.content, "context": context}
        
        self.tokens += count_tokens(answer.content)

        return out
    

if __name__ == "__main__":
    import os
    os.environ['CURL_CA_BUNDLE'] = r'C:\Users\TEO.PAZERA\OneDrive - Zurich Insurance\Desktop\bakalarka_cody\Bakalarka_rag\huggingface.co.crt'
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    rag = RAG()
    rag.load_db(company_name="ABF", vec_db_path="semantically_split_vectors")
    question = "Is the company committed to the Paris (Climate) Agreement?"
    form_of_answer = "yes/no"
    explanation = "The Paris agreement asks to limit global warming to well below 2°C and pursuing efforts to limit it to 1.5°C. While our ambition is to align to a 1.5° pathway, some companies also commit to a less onerous 2°C target."
    # answer = rag.answer_query(query=question, form_of_answer=form_of_answer, explanation=explanation, filter={"company_name": "RWE"})
    # print(answer)
    answer = rag.answer_query(query=question, form_of_answer=form_of_answer, explanation=explanation, k=6, heuristic=["CoT", "basic"])
    print(answer)