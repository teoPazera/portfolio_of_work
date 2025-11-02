from sentence_transformers import CrossEncoder
from langchain_core.documents.base import Document
import torch
from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
import tiktoken
from langchain_core.messages import BaseMessage
from typing import Union
from utils.token_counter import count_tokens

class Rerankers:

    def __init__(self) -> None:
        """Rerankers class is used in RAG pipeline to reorder/compress number of docs
         to be selected in final Question-Answering call"""
        load_dotenv()
        self.llm = AzureChatOpenAI(deployment_name="gpt-4o-mini-20240718",
                                    temperature=0.1,
                                    azure_endpoint=os.getenv("AZURE_ENDPOINT_COMPLETION"),
                                    max_tokens=150
        ) 
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2", activation_fn=torch.nn.Sigmoid())


    def cross_encoder(self, docs: list[Document], query: str, k: int=6) -> list[Document]:
        """Reranking function that compresses the number of documents and selects
        top few according to transformer-based model which rates how given query and chunk
        rank. 
        docs: List[Document] -> A list of Documents to be reranked.
        query: str -> query based on which the documents will be reranked
        
        Returns:
            List[Documents]: A reranked list of Documents
        """

        query_document_pair: list[tuple[str, str]] = [(query, doc.page_content) for doc in docs]
        scores: list[float] = self.model.predict(query_document_pair)
        doc_score_pairs: tuple[Document, float] = list(zip(docs, scores))

        sorted_doc_score_pairs: tuple[Document, float] = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        sorted_docs: list[Document] = [doc for doc, score in sorted_doc_score_pairs]

        return sorted_docs[:k]

        
    def gpt(self,  docs: list[Union[str, Document]], query: str, n: int=12, m: int=6, g_RAG: bool=False) -> list[Union[str, Document]]:
        """List-wise reranking function that compresses the number of documents and selects
        top few documents according to LLM that decides which are important to 
        answer given query correctly.
        docs: list[Union[str, Document]] -> A list of Documents to be reranked.
        query: str -> query based on which the documents will be reranked
        n: int -> num of docs to be ranked at a time in a sliding window ranking
        m: int -> num of docs to be swapped in the sliding window
        g_RAG: bool -> whether the ranked passages are text chunks or nodes
        n-m -> n - m num of docs to be selected on the output

        Returns:
            list[Union[str, Document]]: A reranked list of Documents or strings of KG rag
        """

        def generate_ranking_prompt(passages: list[tuple[int, str]], query: str) -> str:
            """
            Generates a formatted ranking prompt for an LLM call.
            passages: A list of tuples, where each tuple contains (identifier, content).
            query: The search query to rank passages against.

            return: A formatted string prompt.
            """
            template = (
            "USER: I will provide you with {num} passages, each indicated by a numerical identifier []. "
            "Rank the passages based on their relevance to the search query: ({query}).\n\n"
            "{passages_list}\n"
            "Rank the {num} passages above based on their relevance to the search query. "
            "All the passages should be included and listed using identifiers, in descending order of relevance. "
            "The output format should be []. e.g., [4] > [2] > [3]. Only respond with the ranking results, "
            "do not say any word or explain."
            )

            passages_str = "\n".join(
                "[{}] {}".replace("{}", str(identifier), 1)
                .replace("{}", content, 1)
                for identifier, content in passages
            )

            prompt  = template.replace("{num}", str(len(passages))).replace("{query}", query).replace("{passages_list}", passages_str)

            return prompt
        
        tokens_spent: int = 0
        passages_to_rank: list[tuple[int, str]]
        if g_RAG: 
            passages_to_rank  = [(index, doc) for index, doc in enumerate(docs, start=1)]
            entire_text = "\n---\n".join( docs)

        else:
            passages_to_rank  = [(index, doc.page_content) for index, doc in enumerate(docs, start=1)]
            entire_text = "\n---\n".join([d.page_content for d in docs])
          
        if count_tokens(entire_text) > 100000: 
            print("rerank sliding")
            # sliding window strategy
            subset: list[tuple[int, str]] = passages_to_rank[:n]

            answer: BaseMessage 
            index: int
            for index in range(n, len(passages_to_rank), m):
                prompt = generate_ranking_prompt(subset, query=query)
                tokens_spent += count_tokens(prompt)
                answer = self.llm.invoke(prompt)
                ranking: str = answer.content
                ranking: list[str] = ranking.split(">")
                ranking_list: list[int] = []
                for rank in ranking:
                    num_rank = rank.strip().replace("]", "").replace("[", "").replace(".", "")
                    if num_rank.isnumeric():
                        ranking_list.append(int(num_rank))
                subset = []
                for rank in ranking_list[:n-m]:
                    subset.append(passages_to_rank[rank-1])
                subset += passages_to_rank[index:index+(n-m)]

        else:
            prompt = generate_ranking_prompt(passages_to_rank, query=query)
            tokens_spent += count_tokens(prompt)
            answer: BaseMessage = self.llm.invoke(prompt)
            tokens_spent += count_tokens(answer.content)
            ranking: str = answer.content
            ranking: list[str] = ranking.split(">")
            ranking_list: list[int] = []
            for rank in ranking:
                num_rank = rank.strip().replace("]", "").replace("[", "").replace(".", "")
                if num_rank.isnumeric():
                    ranking_list.append(int(num_rank))
            
        if g_RAG:
            output_docs: list[str] = []
        else:
            output_docs: list[Document] = []
        
        index: int
        print(n-m)
        for index in ranking_list[:n-m]: #preserving n - m 
            output_docs.append(docs[index-1])

        return output_docs, tokens_spent

    def peripheral(self, docs: list[Document]) -> list[Document]:
        """Rerank to avoid Lost In the Middle (LIM) problem,
        where LLMs pay more attention to items at the ends of a list,
        rather than the middle. So we re-rank to make the best passages
        appear at the periphery of the list.
        Example reranking:
        1 2 3 4 5 6 7 8 9 ==> 1 3 5 7 9 8 6 4 2
        docs: List[Document] -> A list of Documents to be reranked.

        Returns:
            List[Documents]: A reranked list of Documents.
        """
        # Splitting items into odds and evens based on index, not value
        odds = docs[::2]   #taking odd indexed documents
        evens = docs[1::2][::-1]   #taking even indexed documents and than reverse them

        # Merging them back together
        return odds + evens