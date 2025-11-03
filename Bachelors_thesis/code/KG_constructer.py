import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import networkx as nx
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from parsing_model import Parser
from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrEngine, TableFormerMode
from openai import BadRequestError
import asyncio 
import aiohttp
import re
from KG_entity_types_prompts import (ENTITY_TYPE_GENERATION_JSON_PROMPT, SUMMARIZE_PROMPT, 
                                     RELATIONSHIP_SUMMARIZATION_PROMPT, COMMUNITY_SUMMARIZATION_PROMPT)
from KG_relationship_prompts import ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT, GEMINI_ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT
from utils.token_counter import count_tokens
from networkx.readwrite import json_graph
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import copy
from graspologic.partition import hierarchical_leiden
from langchain_community.vectorstores import Chroma
import time

def convert_sets_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj

class KG_construct:

    def __init__(self) -> None:
        """Class which takes in parsed out chunks of text from Sustainability reports and creates a Knowledge graph on top of them
        """
        load_dotenv()
        self.llm_json = AzureChatOpenAI(
            deployment_name="gpt-4o-mini-20240718",
            temperature=0.1,
            azure_endpoint=os.getenv("AZURE_ENDPOINT_COMPLETION"),
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        self.llm_regular = AzureChatOpenAI(
            deployment_name="gpt-4o-mini-20240718",
            temperature=0.1,
            azure_endpoint=os.getenv("AZURE_ENDPOINT_COMPLETION")
        )
        self.embedding_model = AzureOpenAIEmbeddings(
            api_key=os.getenv('OPENAI_API_KEY'),
            azure_endpoint=os.getenv("AZURE_ENDPOINT_EMBEDDING"),
            model="text-embedding-3-large"
        )
        self.tokens_spent = 0 
       

    async def extract_entities_relations(self, text_chunk: str) -> json.decoder:
        """asynchronous call to the llm to find all entities present in text chunk from the entity types defined below

        Args:
            text_chunk: str: text from which the entities will be extracted

        Returns:
            json.decoder: json dictionary of the entities and relationships extracted
        """
        #call to extract entities in the chunk
        # defining entity types
        entity_types = ["Scope 1 Emmission", "Scope 2 Emmission", "Scope 3 Emmission", "Energy reduction", "Co2", 
                "Operational emmision", "Carbon",
                "Carbon Emmision reduction", "Target Announcement Year", "Target Baseline Year", 
                "Target End/ Horizon Year", "Percentage Reduction (%)", 
                "Percentage Reduction comprising Carbon Offsets or Carbon Removals (%)", 
                "Details of Carbon Offsets or Carbon Removals to be used", 
                "Target Type (Absolute, Intensity, or Other)(Please specify further detail if Other)", 
                "Target Units (Relevant for 'Intensity' target type)",
                "Actions to meet targets", "Interim target",
                "Capex for transitional plan", "Investments to achieve targets",
                "Climate action 100 assesment", "TPI assesment", "SBTi assesment",
                "Alignment with Global Reporting Initiative (GRI)", 
                "Alignment with Sustainability Accounting Standards Board (SASB)", 
                "Alignemnt with Taskforce for Climate Related Disclosures (TCFD)"
                "Alignemtn with International Sustainability Standards Board (ISSB)"]
        
        relationships_extraction_prompt = re.sub("{entity_types}", ",".join(entity_types), ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT)
        relationships_extraction_prompt = re.sub("{input_text}", text_chunk, relationships_extraction_prompt)
        self.tokens_spent += count_tokens(relationships_extraction_prompt)
        # call to find the relationship between entities
        try:
            response = await self.llm_json.ainvoke(relationships_extraction_prompt) # , config=config
        except BadRequestError as e:
            print(f"Error: {e}")
            return None
        try:
            return json.loads(response.content)  # response.content Ensure the response is parsed as JSON
            #print(json.loads(response2.content))
        except json.JSONDecodeError:
            print(response)
            print("Error: Response is not valid JSON")
            return None  # Handle unexpected response format
        

    async def construct(self, docs: list[Document]) -> nx.Graph:
        """function takes in list of document chunks which it passes to LLM asynchronously 
        to get entities and relationships extracted

        Args:
            docs (list[Document]): list of document text chunks Document contains page_content=text and metadata 

        Returns:
            nx.Graph: outputs nx Knowledge Graph populated by the llm entities and relationships
        """
        G = nx.Graph()
        results: list = []
        grouped_together = 100
        for index in range(0, len(docs), grouped_together):
            try:
                tasks = [self.extract_entities_relations(chunk.page_content) for chunk in docs[index:index+grouped_together]]
                inter_mediate_results = await asyncio.gather(*tasks)  # Run all async calls concurrently
                print(index+grouped_together)
                results.extend(inter_mediate_results)
            except TypeError:
                time.sleep(45)
                print("error in creation of entities")
                try:
                    tasks = [self.extract_entities_relations(chunk.page_content) for chunk in docs[index:index+grouped_together]]
                    inter_mediate_results = await asyncio.gather(*tasks)  # Run all async calls concurrently
                    print(index+grouped_together)
                    results.extend(inter_mediate_results)
                except:
                    pass
        print("tokens spent for construction: ", self.tokens_spent)
        for idx, data in enumerate(results):
            if data:
                chunk: str = docs[idx].page_content
                file_name: str = docs[idx].metadata["file_name"]
                page_number: int = docs[idx].metadata["page_number"]
                if "entities" in data:
                    entity: dict
                    for entity in data["entities"]:
                        try:
                            if G.has_node(entity["name"]):
                                G.nodes[entity["name"]]['chunks'].add(chunk)
                                G.nodes[entity["name"]]['description'].add(entity["description"])
                                G.nodes[entity["name"]]["file_name"].add(file_name)
                                G.nodes[entity["name"]]["page_number"].add(page_number)
                            else:
                                G.add_node(entity["name"], description={entity["description"]}, type=entity["type"],
                                        file_name={file_name}, page_number={page_number}, chunks={chunk})
                        except KeyError as e:
                            print(idx)
                            print(data)
                            print(chunk)
                            print(entity)
                            print(G.nodes[entity["name"]])
                
                            
                
                if "relationships" in data:
                    rel: dict
                    for rel in data["relationships"]:
                        try:
                            if G.has_edge(rel["entity1"], rel["entity2"]):
                                existing_data = G[rel["entity1"]][rel["entity2"]]
                                existing_data['relationship'] += f"\n{rel['relationship']}"
                                existing_data['score'] = max(existing_data['score'], rel["relationship_strength"])
                            else:
                                G.add_edge(rel["entity1"], rel['entity2'], relationship=rel['relationship'], score=rel['relationship_strength'])
                        except:
                            print(rel)
        for node in G.nodes:
            if 'chunks' not in G.nodes[node]:
                G.nodes[node]['chunks'] = []
            else:
                G.nodes[node]['chunks'] = list(G.nodes[node]['chunks'])

            if 'description' not in G.nodes[node]:
                G.nodes[node]['description'] = ""
            else:
                G.nodes[node]['description'] = "\n".join(list(G.nodes[node]['description']))

            if 'file_name' not in G.nodes[node]:
                G.nodes[node]['file_name'] = []
            else:
                G.nodes[node]['file_name'] = list(G.nodes[node]['file_name'])
            
            if 'page_number' not in G.nodes[node]:
                G.nodes[node]['page_number'] = []
            else:
                G.nodes[node]['page_number'] = list(G.nodes[node]['page_number'])
                
        return G  # Return the constructed knowledge graph
    

    def deduplicate(self, G: nx.Graph) -> nx.Graph:
        """function to deduplicate entities from preconstructed nx Knowledge graph

        Args:
            G (nx.Graph): input knowledge graph 

        Returns:
            nx.Graph: outputs knowledge graph with contained nodes whic
        """
        lst = []
        nodes_label_mapping_lst = []
        for node, data in G.nodes(data=True):
            entity = "\nName: " + node + " \nDescription: " + data['description']
            embed =  self.embedding_model.embed_query(entity)
            lst.append(embed)
            nodes_label_mapping_lst.append(node)

        #print(len(lst), len(nodes_label_mapping_lst)) 
        X = np.array(lst)
        cosine_sim = cosine_similarity(X)
        
        np.fill_diagonal(cosine_sim, 0)
        cosine_sim = np.where(cosine_sim >= 0.925, 1, 0)

        # creating adjacency list from above matrix
        adjacency_list = defaultdict(list)
        for i in range(cosine_sim.shape[0]):
            not_connected = True
            for j in range(cosine_sim.shape[0]):
                if cosine_sim[i,j]==1:
                    not_connected = False
                    adjacency_list[i].append(j)
            if not_connected:
                adjacency_list[i].append(i)
        # creating list of connected nodes by clustering above connectivity data using BFS
        connected_components = []
        for key in adjacency_list.keys():
            if key not in [x for y in connected_components for x in y]:
                visited = []
                queue = [key]
                while len(queue)>0:
                    node = queue.pop(0)
                    if node not in visited:
                        visited.append(node)
                        for neighbour in adjacency_list[node]:
                            queue.append(neighbour)
                connected_components.append(visited)

        nodes_to_merge_lst = [set(nodes_label_mapping_lst[y] for y in x) for x in connected_components]
        
        
        G_copy = copy.deepcopy(G)

        for nodes_to_merge in nodes_to_merge_lst:
            if len(nodes_to_merge)>1:
                # Create a new node as the merged node
                merged_node = "\n".join(list(nodes_to_merge))
                G_copy.add_node(merged_node, description=set(), chunks=set(), file_name=set(), page_number=set())

                # Merge node properties
                for node in nodes_to_merge:
                    if G_copy.has_node(node):
                        node_data = G_copy.nodes[node]
                        G_copy.nodes[merged_node]['description'].add(node_data.get('description'))
                        for chunk in node_data.get('chunks'):
                            G_copy.nodes[merged_node]['chunks'].add(chunk)

                        for file_name in node_data.get("file_name"):
                            G_copy.nodes[merged_node]['file_name'].add(file_name)

                        for page_number in node_data.get('page_number'):
                            G_copy.nodes[merged_node]['page_number'].add(page_number)

                    
                G_copy.nodes[merged_node]['chunks'] = list(G_copy.nodes[merged_node]['chunks'])
                G_copy.nodes[merged_node]['description'] = "\n".join(list(G_copy.nodes[merged_node]['description']))
            
                # Update edges
                for node1, node2, data in G.edges(data=True):
                    if node1 in nodes_to_merge or node2 in nodes_to_merge:
                        new_node1 = merged_node if node1 in nodes_to_merge else node1
                        new_node2 = merged_node if node2 in nodes_to_merge else node2
                        
                        if new_node1 != new_node2:
                            if G_copy.has_edge(new_node1, new_node2):
                                existing_data = G_copy[new_node1][new_node2]
                                existing_data['relationship'] += f"\n{data['relationship']}"
                                existing_data['score'] = max(existing_data['score'], data['score'])
                            else:
                                G_copy.add_edge(new_node1, new_node2, relationship=data['relationship'], score=data['score'])

        # Remove old nodes
        for nodes_to_merge in nodes_to_merge_lst:
            if len(nodes_to_merge) > 1:                  
                G_copy.remove_nodes_from(nodes_to_merge)

        # checking if merging has been successful
        assert len(G.nodes) - len([y for x in nodes_to_merge_lst for y in x if len(x)>1]) + \
            len([x for x in nodes_to_merge_lst if len(x)>1]) == len(G_copy.nodes), "Merging not sucessful"
        
    
        # deleting earlier graph
        del G
        G = G_copy
        return G_copy

    def describe_nodes_and_edges(self, G: nx.Graph) -> nx.Graph:
        """
        function to summarize description of given nodes or relationships if they exceed treshold of tokens to not preserve lengthy summaries
        Args:
            G (nx.Graph): input KG

        Returns:
            nx.Graph: KG with summarized descriptions
        """
        for node, data in G.nodes(data=True):
            if count_tokens(data['description']) > 300:
                print(f"Description summerized for Node_{node}")
                prompt = SUMMARIZE_PROMPT.replace("{entity_name}", node)
                prompt = prompt.replace("{description_list}", str([data['description']]))
                desc =  self.llm_regular.invoke(prompt)
                self.tokens_spent += count_tokens(prompt) + count_tokens(desc.content)
                data['description'] = desc.content

        # summarizing relationship description
        for node1, node2, data in G.edges(data=True):
            if count_tokens(data['relationship']) > 300:
                print(f"Relationship summerized bw Node1 ({node1}) and Node2 ({node2})")
                prompt = RELATIONSHIP_SUMMARIZATION_PROMPT.replace("{entity_name1, entity_name2}", f"{node1}, {node2}")                                                          
                prompt = prompt.replace("{relationships_list}", str([data['relationship']]))
                desc = self.llm_regular.invoke(prompt)
                self.tokens_spent += count_tokens(prompt) + count_tokens(desc.content)

                data['relationship'] = desc.content
        print("Tokens spent after node, rel desc: ", self.tokens_spent)
        return G
    
    def create_communities(self, G: nx.Graph) -> nx.Graph:
        """function to generate community summaries on the KG

        Args:
            G (nx.Graph): input KG

        Returns:
            nx.Graph: KG with community summaries on top of the clusters in the KG
        """
        # creating prompt for community summarization
        def community_summary_prompt_generator(cluster_id: int, cluster_nodes: list[dict]) -> dict:

            """
            Args:
                cluster_id: int
                cluster_nodes: list[dict]: nodes which make up given cluster
            
            Returns:
                dict: nodes which make up a cluster with the relationships
            """
            cluster_info = {
            "communityId": cluster_id,
            "nodes": [],
            "relationships": []
            }

            for node in cluster_nodes:
                node_data = G.nodes[node]
                node_info = {
                    "id": node,
                    "description": node_data["description"]
                }
            cluster_info["nodes"].append(node_info)

            for node1, node2, data in G.edges(data=True):
                if node1 in cluster_nodes and node2 in cluster_nodes:
                    relationship_info = {
                    "start": node1,
                    "description": data["relationship"],
                    "end": node2
                    }
                    cluster_info["relationships"].append(relationship_info)

            return cluster_info

        # creating hierarchial clusters using community detection algo
        communities = hierarchical_leiden(G, max_cluster_size=10)

        # generating community summaries
        node_cluster_dct = defaultdict(list)
        for community in communities:
            node_cluster_dct[community.node].append((community.cluster, community.level))

        cluster_node_dct = defaultdict(list)
        for community in communities:
            cluster_node_dct[community.cluster].append(community.node)

        community_summary = {}
        for key, val in cluster_node_dct.items():
            prompt = COMMUNITY_SUMMARIZATION_PROMPT.replace("{community_info}", 
                                                            str(community_summary_prompt_generator(key, val)))
            if count_tokens(prompt)>20000:
                prompt = " ".join(prompt.split()[:20000])
                print(f"prompt truncated for Cluster_{key}")
            
            summary = self.llm_regular.invoke(prompt)
            self.tokens_spent += count_tokens(prompt)+count_tokens(summary.content)
            community_summary[key] = summary.content
            #print(summary.content)
        print("Tokens spent after community summary: ", self.tokens_spent)

        # storing all community summaries at different heirarchial level for each node
        for node, data in G.nodes(data=True):
            if node in node_cluster_dct.keys():
                node_level_summary = []
                for level in sorted(list(set([x[1] for x in node_cluster_dct[node]]))):
                    associated_communities = [y for y in node_cluster_dct[node] if y[1]==level]
                    associated_communities_summaries = [community_summary[y[0]] for y in associated_communities]
                    node_level_summary.append(("\n".join(associated_communities_summaries), level))
                data["community_summaries"] = [y[0] for y in sorted(node_level_summary, key = lambda x:x[1], reverse=True)]
            else:
                data["community_summaries"] = " "

        return G
    
    def store_node_embedding(self, G: nx.Graph, company_name: str) -> None:
        """function to save embedings for individual nodes of a Knowledge graph

        Args:
            G (nx.Graph): KG to be embedded
            company_name (str): name of the company for which the KG was made for to save in metadata
        """
        docs = []
        for node, data in G.nodes(data=True):
            entity = "\nName: " + node + " \nDescription: " + data['description']
            doc = Document(
                page_content=entity,
                metadata={"source": node,
                          "company_name": company_name}
            )
            docs.append(doc)
        print(len(docs))
        for index in range(0, len(docs), 10):
            Chroma.from_documents(documents=docs[index:index+10], 
                                   embedding=self.embedding_model, 
                                   persist_directory="KG_vectors")
        
        return


    def entire_graph_creation(self, company_name: str) -> None: 
        """function which does all of the steps in KG creation 
        from the LLM picking out entities to the community summaries

        Args:
            company_name (str): name of the company for which we create a KG
        """
        vector_store = Chroma(
            collection_name="langchain",
            embedding_function=self.embedding_model,
            persist_directory="basic_vectors"  
        )

        results = vector_store.get(where={"company_name": company_name}, include=["documents", "metadatas"])
        documents: list[Document] = []
        extra_doc: str = "" 
        for idx, page_content in enumerate(results["documents"]):
            if len(page_content) < 200:
                extra_doc += page_content
            else:
                if extra_doc:
                    documents.append(Document(page_content=extra_doc + "\n" + page_content,
                                                metadata=results["metadatas"][idx])) 
                    extra_doc = ""
                else:
                    documents.append(Document(page_content=page_content,
                                                metadata=results["metadatas"][idx])) 
  
        print(len(documents))                              
        G: nx.Graph = asyncio.run(self.construct(docs=documents))
        data = json_graph.node_link_data(G)
        data = convert_sets_to_lists(data)
        with open(f"KGs\graph_{company_name}_first.json", "w") as f:
            json.dump(data, f, indent=2)
        
        print("graph_created")

        G = self.deduplicate(G=G)
        data = json_graph.node_link_data(G)
        data = convert_sets_to_lists(data)
        with open(f"KGs\graph_{company_name}_deduplicated.json", "w") as f:
            json.dump(data, f, indent=2)
        
        print("graph_deduplicate")

        # # Convert JSON data to a NetworkX graph
        G = json_graph.node_link_graph(data)
        G = self.describe_nodes_and_edges(G=G)
        data = json_graph.node_link_data(G)
        data = convert_sets_to_lists(data)
        with open(f"KGs\graph_{company_name}_described.json", "w") as f:
            json.dump(data, f, indent=2)
        
        print("graph_described")

        G = self.create_communities(G=G)

        data = json_graph.node_link_data(G)
        data = convert_sets_to_lists(data)
        with open(f"KGs\graph_{company_name}_final.json", "w") as f:
            json.dump(data, f, indent=2)
        
        print("communities_created")

if __name__ == "__main__":
    os.environ['CURL_CA_BUNDLE'] = r'C:\Users\TEO.PAZERA\OneDrive - Zurich Insurance\Desktop\bakalarka_cody\Bakalarka_rag\huggingface.co.crt'
    knowledge_graph_creator = KG_construct() 
    for company_name in ["ABF", "BASF", "Bosch", "BP", "Holcim", "Marriott", "RWE"]: # , "Marriott"
        with open(f"KGs\graph_{company_name}_final.json", "r") as f:
            data = json.load(f)

        G = json_graph.node_link_graph(data)
        knowledge_graph_creator.store_node_embedding(G=G, company_name=company_name)
        
    '''
    with open("graph_Caesar_w_communities.json", "r") as f:
        data = json.load(f)

    # Convert JSON data to a NetworkX graph
    G = json_graph.node_link_graph(data)
    
    knowledge_graph_creator.store_node_embedding(G=G)

    folder = "pdf_files"
    parser = Parser(pdf_documents_directory=folder, chunk_lengths=3000, embed_model="text-embedding-large-3")

    pipeline_options = PdfPipelineOptions(
        ocr_engine=OcrEngine.TESSERACT,  # Choose OCR engine: TESSERACT, EASY_OCR, etc.
        do_table_structure=True,         # Enable table structure recognition
        table_structure_options={
            'mode': TableFormerMode.ACCURATE  # Use accurate mode for better table parsing
        },
        enable_remote_services=False     # Disable remote services for local processing
    )

    docs = parser.get_documents(pdf_pipeline_options=pipeline_options, chunker_type="Recursive", 
                                text_format='md', split_by_pages=True)
    print(len(docs))
    graph = asyncio.run(knowledge_graph_creator.construct(docs=docs))

    print("Nodes:")
    print(graph.nodes(data=True))
    print("Edges:")
    print(graph.edges(data=True))


    # Optionally, save graph to a JSON file
    data = json_graph.node_link_data(graph)
    with open("graph_Caesar.json", "w") as f:
        json.dump(data, f, indent=2)



    graph = knowledge_graph_creator.deduplicate(G)
    
    data = json_graph.node_link_data(graph)
   
        
    data = convert_sets_to_lists(data)
    with open("graph_Caesar_deduplicate.json", "w") as f:
        json.dump(data, f, indent=2)
    '''

