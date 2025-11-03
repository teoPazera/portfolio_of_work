import spacy
import os
from openai import AzureOpenAI
from openai.types.embedding import Embedding
from langchain_core.documents.base import Document
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from utils.prompts import *
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd 
from sentence_transformers import SentenceTransformer
os.environ['CURL_CA_BUNDLE'] = r'C:\Users\TEO.PAZERA\OneDrive - Zurich Insurance\Desktop\bakalarka_cody\Bakalarka_rag\huggingface.co.crt'
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
import time
import spacy
from utils.token_counter import count_tokens

def decompose_text(text: str) -> list[str]:
    """decomposes text into sentences using spacy

    Args:
        text (str): text to be split into individual sentences

    Returns:
        list[str]: list of sentences 
    """
    nlp = spacy.load("en_core_web_sm")
    if text:
        text = str(text)
        doc = nlp(text)
        return [sent.text for sent in doc.sents]
    else:
        return [""]

class Test:
    def __init__(self) -> None:
        """
        Test class is used to evaluate the performance of the model on the test set.
        It uses the AzureChatOpenAI model to evaluate the performance of the model on the test set.
        """
        load_dotenv()
        self.eval_llm = AzureChatOpenAI(deployment_name="gpt-4o-mini-20240718",
                                    temperature=0,
                                    azure_endpoint=os.getenv("AZURE_ENDPOINT_COMPLETION")
        )
        load_dotenv(dotenv_path=".env2")
        self.eval_llm2 = AzureChatOpenAI(deployment_name="gpt-4o-mini-20240718",
                                    temperature=0,
                                    azure_endpoint=os.getenv("AZURE_ENDPOINT_COMPLETION")
        )
        self.client = AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version = os.getenv("OPENAI_API_VERSION"),
            azure_endpoint = os.getenv("AZURE_ENDPOINT_EMBEDDING") 
        )
        self.sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokens_spent = 0
        self.last_pause_time = time.time()
        self.request = 0 


    def check_and_pause(self) -> None:
        """Checks token count over the last minute 
        and the requests to api over the last minute 
        if the quota is exceeded pauses to prevent openai errors.
        Args:
            None
        Returns:
            None
        """
        if self.tokens_spent >= 380000 or self.request >= 180:
            time_since_last_pause = time.time() - self.last_pause_time
            if time_since_last_pause < 60:
                pause_duration = 60 - time_since_last_pause
                print(f"Pausing for {pause_duration:.2f} seconds to avoid rate limiting...") # Corrected message
                time.sleep(pause_duration)
            print("check_performed")
            self.tokens_spent = 0  # Reset token count after pausing
            self.request = 0
            self.last_pause_time = time.time()    

    def test_all_dataset(self, answers_datafrane: pd.DataFrame) -> None:
        """function to iterate over entire set of question and answers and evaluate all of the metrics for them

        Args:
            answers_datafrane (pd.DataFrame): DataFrame containing the questions and answers to be evaluated

        Returns:
            None
        """
        results: list[dict] = []
        for index, row in answers_datafrane.iterrows():
            # if index % 21 == 0:
            print(index) 
            
            question: str = row["question"]
            answer: str = row["answer_of_model"]
            ground_truth_text: str = row["ground_truth_context"]
            retrieved_chunks: str = row["retrieved_context"]
            company: str = row["company"]
            manual_answer: str = row["manual_answer"]
            faithfulness: float = asyncio.run(self.faithfulness_metric(answer=answer, context_chunks=retrieved_chunks.split("\n---\n")))
            answer_relevancy: float = self.answer_relevance_metric(answer=answer, question=question, manual_answer=manual_answer)
            context_relevancy: float = asyncio.run(self.context_relevancy_metric(question=question, context_chunks=retrieved_chunks.split("\n---\n")))
            correctness: float = self.answer_correctness(question=question, manual_answer=manual_answer, model_answer=answer)
            precision: float
            recall: float
            precision, recall = tester.calculate_precision_recall(retrieved_chunks=retrieved_chunks, ground_truth_text=ground_truth_text)
            f1_score: float = self.f1_score(precision=precision, recall=recall)
            results.append({
                "question": question,
                "answer": answer,
                "manual_answer": manual_answer,
                "company": company,
                "correctness": correctness,
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_relevancy": context_relevancy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            })

        output_df = pd.DataFrame(results)
        return output_df


    def answer_correctness(self, question:str, manual_answer: str, model_answer: str) -> float:
        """
        Evaluate the correctness of the model's answer using a single LLM call.
        
        Args:
            question     (str): Question asked
            ground_truth (str): The manually provided correct answer.
            model_answer (str): The answer produced by the model.        
        Returns:
            float: A score between 0 and 1 indicating answer correctness.
        """
        prompt: str = f"""You are an expert RAG evaluator. Below are two answers for the same question:
        Question: '{question}'
        Ground truth: '{manual_answer}'
        Model answer: '{model_answer}'
        Evaluate the correctness of the model's answer compared to the ground truth. Consider factual accuracy, relevance, and completeness.
        If ground truth is that something is not mentioned, negative answer is as well at least partially correct.
        Provide a score from 0 to 1 where 1 is perfectly correct and 0 is completely incorrect. Return only the score.
        """
        try: 
            self.tokens_spent += count_tokens(text=prompt)
            self.request += 1 
            self.check_and_pause()
            output_text = self.eval_llm.invoke(prompt)
            self.tokens_spent += output_text.response_metadata["token_usage"]["completion_tokens"]
            score = float(output_text.content)
            return score
        
        except Exception as e:
            # Handle any parsing errors or API exceptions
            print("Error during evaluation:", e)
            return 0.0
        
    async def verify_claim_against_chunk(self, claim: str, chunk: str) -> str:
        """Check if a claim is supported, contradicted, or unknown against a specific chunk.
        Args:
            claim: str = claim to be verified
            chunk: str = chunk of text that has been retrieved to help answer the question
        Returns:    
            str: "yes" if the claim is supported, "no" if contradicted, or "idk" if uncertain.
        """
       
       
        prompt = f"""Given the reference text below, does the claim agree with it?
        \nReference Text: {chunk}\nClaim: {claim}
        \nRespond only with 'yes', 'no', or 'idk'."""
        try:
            self.tokens_spent += count_tokens(text=prompt)
            self.request += 1
            self.check_and_pause()
            response = await self.eval_llm.ainvoke(prompt)
            answer = response.content.strip().lower()
            self.tokens_spent += response.response_metadata["token_usage"]["completion_tokens"]
            #print("ne", self.tokens_spent, self.request, sep="-")
            return answer if answer in ["yes", "no", "idk"] else "idk"
        
        except Exception as e:
            print("faithfulness_metric")
            print("e", self.tokens_spent, self.request, sep="-")
            return "idk"

    async def faithfulness_metric(self, answer: str, context_chunks: list[str]) -> float:
        """function outputs the faithfulness score of the model's answer

        Args:
            answer (str): answer provided by the model
            context_chunks (list[str]): chunks of text that have been retrieved to help answer the question

        Returns:
            float: faithfulness score of the model's answer
        """
        claims = decompose_text(answer)
    
        claim_results = {}

        # Parallel execution for claim verification
        tasks = [
            self.verify_claim_against_chunk(claim, chunk)
            for claim in claims
            for chunk in context_chunks
        ]

        results = await asyncio.gather(*tasks)

        # Aggregate results
        idx = 0
        for claim in claims:
            claim_results[claim] = []
            for _ in context_chunks:
                result = results[idx]
                claim_results[claim].append(result)
                idx += 1

        truthful_claims = 0
        for claim, results in claim_results.items():
            if "yes" in results:
                truthful_claims += 1
              # Contradicted claim (ignored) all(res == "no" for res in results)
            else:
                continue  # "IDK" cases

        faithfulness_score = truthful_claims / len(claims)
        return faithfulness_score
    
    def generate_questions_to_answer(self, answer: str, manual_answer: str) -> list[str]:
        """
        Function to generate questions based on the model's answer and the manually created answer.
        Args:
            answer (str): answer provided by the model
            manual_answer (str): manual answer submited by the sustainability team
        Returns:    
                list[str]: list of generated questions
        """

        prompt = f"""Given these answers: ANSWER of model :{answer}
        MANUALY created ANSWER to the question: {manual_answer}
        Generate 3 questions that these answers would be appropriate for. 
        Separate the questions with "|" symbol 
        Make the questions specific and directly related to the content.
        """
        try:
            self.tokens_spent += count_tokens(prompt)
            self.request += 1 
            self.check_and_pause()
            response = self.eval_llm.invoke(prompt)
            self.tokens_spent += response.response_metadata["token_usage"]["completion_tokens"]
            return response.content.split("|")
        
        except Exception as e:
            print(f"Error creating questions from answer {answer}")
            return [""]*3
        

    def answer_relevance_metric(self, answer: str, question: str, manual_answer: str) -> float:
        """
        Function to evaluate the relevance of the answer to the question using cosine similarity.
        Args:
            answer (str): answer provided by the model
            question (str): _description_
            manual_answer (str): manual answer submited by the sustainability team

        Returns:
            float: average cosine similarity of the generated questions from the answers and actual answer
        """
        potential_questions = self.generate_questions_to_answer(answer, manual_answer)

        all_texts = [question] + potential_questions
        embeddings = self.sentence_embedding_model.encode(all_texts, normalize_embeddings=True)

        og_question_embedding = embeddings[0]  
        pot_question_embeddings = embeddings[1:] 

        similarities = cosine_similarity([og_question_embedding], pot_question_embeddings)[0]
        mean_similarity = np.mean(similarities)
        
        return mean_similarity
    
    def generate_embeddings(self, docs: list[str], model: str) -> list[float]: 
        """function to generate embeddings from list of strings
        Args:
            docs: list[str]= chunks to be embedded
            model: str= name of embedding model for example: text-embedding-3-large
        Returns: 
            list[Embedding] list of embeddings is returned on the output in the same order as docs on input"""
        response: list[Embedding] = self.client.embeddings.create(input = docs, model=model).data
        
        return [embedding.embedding for embedding in response]


    async def verify_relevancy(self, context_chunk: str, question: str) -> str:
        """Check if a from context is relevant to answering given question
        Args:
            context_chunk: str = chunk of text that has been retrieved to help answer the question
            question: str = question being asked by user
        Returns:
            str: "yes" if the context is relevant to the question, "no" if not, or "idk" if uncertain.
        """
                
        prompt = f"""Please extract only the relevant sentences from the provided context that can potentially help answer the following question.
                 If no relevant sentences are found, return the phrase "Insufficient Information."
                You are not allowed to make any changes to the sentences from the given context. 
                Separate the relevant sentences with "$|$" in the output. 

                Question to which sentences should be relevant: {question} 
                Context chunk: {context_chunk} """
        try:
            self.tokens_spent += count_tokens(prompt)
            self.request += 1 
            self.check_and_pause()
            response = await self.eval_llm2.ainvoke(prompt)
            self.tokens_spent += response.response_metadata["token_usage"]["completion_tokens"]
            answer = response.content.strip().lower()
            return answer
        
        except Exception as e:
            print("relevancy")
            print("e", self.tokens_spent, self.request, sep='-')
            return "Insufficient Information."

    async def context_relevancy_metric(self, question: str, context_chunks: list[str]) -> float:
        """
        Function used to get score of how much relevant are the context chunks to the question
        Args:
             question: str = question being asked by user
             context_chunks: list[str] = chunks of text that have been retrieved to help answer the question
        Returns:
            float: proportion of relevant sentences from all sentences
        """
        
        claims_from_chunks: list[list[str]] = []
        total_num__senteces: int = 0

        for chunk in context_chunks:
            sentences: list[str] = decompose_text(chunk)
            claims_from_chunks.append(sentences)
            total_num__senteces += len(sentences)

        tasks = [self.verify_relevancy("$|$".join(chunk), question)
            for chunk in claims_from_chunks]

        results = await asyncio.gather(*tasks)
        num_relevant_sentences: int = 0
        for i in range(len(results)):
            if not "insufficient information" in results[i]:
                relevant_sentences: list[str] = results[i].split("$|$")
                num_relevant_sentences += len(relevant_sentences)

        if num_relevant_sentences/total_num__senteces > 1:
            return 1
        else:
            return num_relevant_sentences/total_num__senteces

    def jaccard_score(self, set1: set, set2: set) -> float:
        """
        Calculates the Jaccard score between two sets.
        Args:
            set1: set of words from one sentence
            set2: set of words from second sentence
        Returns:
            float: jaccard score for the overlapping of the two sets
        """
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def measure_hits(self, retrieved_chunks_sentences: list[str], ground_truth_sentences: list[str], threshold: float) -> int:
        """
        Measures the number of hits based on Jaccard score between retrieved chunks and 
        sentences from the ground truth text.

        Args:
            retrieved_chunks: A list of strings, where each string is a retrieved chunk.
            ground_truth_text: A single string containing the entire ground truth text.
            threshold: The Jaccard score threshold for a hit.

        Returns:
            The number of hits.
        """
        hits: int = 0
        for retrieved in retrieved_chunks_sentences:
            retrieved_set = set(retrieved.split())  # Convert chunk to a set of words

            for ground_truth_sentence in ground_truth_sentences:
                ground_truth_set = set(ground_truth_sentence.split())  # Convert sentence to a set of words
                score = self.jaccard_score(retrieved_set, ground_truth_set)

                if score > threshold:
                    hits += 1
                    break  # Move to the next retrieved chunk if a hit is found

        return hits

    def calculate_precision_recall(self, retrieved_chunks: list[str], ground_truth_text: str, threshold: float = 0.7) -> tuple[float]:
        """
        Calculates precision and recall based on the number of hits.

        Args:
            retrieved_chunks: A list of strings, where each string is a retrieved chunk.
            ground_truth_text: A single string containing the entire ground truth text.
            threshold: The Jaccard score threshold for a hit.

        Returns:
            A tuple containing precision and recall.
        """
        retrieved_chunks_sentences: list[str] = decompose_text(retrieved_chunks)
        ground_truth_sentences: list[str] = decompose_text(ground_truth_text)

        hits = self.measure_hits(retrieved_chunks_sentences, ground_truth_sentences, threshold)
        total_retrieved = len(retrieved_chunks_sentences)  # Total documents retrieved
        
        # Estimate total relevant documents:  Number of sentences in the ground truth.
        total_relevant = len(ground_truth_sentences)  

        # Calculate precision and recall
        precision = hits / total_retrieved if total_retrieved > 0 else 0.0
        recall = hits / total_relevant if total_relevant > 0 else 0.0
        
        return precision, recall

    def f1_score(self, precision: float, recall: float) -> float:
        """return harmonic mean from precision and recall

        Args:
            precision (float): percentual amount of relevant retrieved sentences from all the retrieved sentences
            recall (float): percentual amount of relevant retrieved sentences from ground truth sentences which should be retrieved

        Returns:
            float: harmonic mean of precision and recall
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

if __name__== "__main__":
    tester = Test()

    excel_path: str = "evaluation\Evaluation_interactions_KG_rerankGPT.xlsx"
    with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement

        df1 = pd.read_json("graph_hypothesis\combined_KG_vectors_['KG', 'RerankGPT']_8_data.json")
        out1 = tester.test_all_dataset(answers_datafrane=df1) # Corrected variable name
        out1.to_excel(writer, index=False, sheet_name="KG-rerankGPT-8") # Corrected variable name
        del df1 #clean memory
        del out1 #clean memory

    # with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement

    #     df_KG_8 = pd.read_json("graph_hypothesis\combined_KG_vectors_['basic', 'KG']_8_data.json")
    #     output_KG_8 = tester.test_all_dataset(answers_datafrane=df_KG_8) # Corrected variable name
    #     output_KG_8.to_excel(writer, index=False, sheet_name="KG-8") # Corrected variable name
    #     del df_KG_8 #clean memory
    #     del output_KG_8 #clean memory
        
    # excel_path: str = "evaluation\Evaluation_opt_hybrid.xlsx"
    # with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement
    #     # df_hybrid8 = pd.read_json("opt_hypothesis\combined_basic_m_enriched_vectors_['Hybrid', '0.5, 0.5', 'RerankGPT', 'CoT']_8_data.json")
    #     # output_hybrid8 = tester.test_all_dataset(answers_datafrane=df_hybrid8) # Corrected variable name
    #     # output_hybrid8.to_excel(writer, index=False, sheet_name="hybrid8-CoT-RerankGPT") # Corrected variable name
    #     # del df_hybrid8 #clean memory
    #     # del output_hybrid8 #clean memory

    #     df_hybrid6 = pd.read_json("opt_hypothesis\combined_basic_m_enriched_vectors_['Hybrid', '0.5, 0.5', 'RerankGPT', 'CoT']_6_data.json")
    #     output_hybrid6 = tester.test_all_dataset(answers_datafrane=df_hybrid6) # Corrected variable name
    #     output_hybrid6.to_excel(writer, index=False, sheet_name="hybrid6-CoT-RerankGPT") # Corrected variable name
    #     del df_hybrid6 #clean memory
    #     del output_hybrid6 #clean memory
        
    # excel_path: str = "evaluation\Evaluation_hybrid.xlsx"
    # with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement
    #     df_hybrid05 = pd.read_json("advanced_search_hypothesis\combined_basic_m_enriched_vectors_['Hybrid', '0.5, 0.5']_6_data.json")
    #     output_hybrid05 = tester.test_all_dataset(answers_datafrane=df_hybrid05) # Corrected variable name
    #     output_hybrid05.to_excel(writer, index=False, sheet_name="hybrid05") # Corrected variable name
    #     del df_hybrid05 #clean memory
    #     del output_hybrid05 #clean memory

    #     df_hybrid1 = pd.read_json("advanced_search_hypothesis\combined_basic_m_enriched_vectors_['Hybrid', '1,0']_6_data.json")
    #     output_hybrid1 = tester.test_all_dataset(answers_datafrane=df_hybrid1) # Corrected variable name
    #     output_hybrid1.to_excel(writer, index=False, sheet_name="hybrid1") # Corrected variable name
    #     del df_hybrid1 #clean memory
    #     del output_hybrid1 #clean memory

    # excel_path: str = "evaluation\Evaluation_opt8.xlsx"

    # with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement
    #     df_opt8 = pd.read_json("opt_hypothesis\combined_basic_m_enriched_vectors_['CoT', 'RerankGPT', 'HyDE']_8_data.json")
    #     output_opt8= tester.test_all_dataset(answers_datafrane=df_opt8) # Corrected variable name
    #     output_opt8.to_excel(writer, index=False, sheet_name="CoT-RerankGPT-Hyde8") # Corrected variable name
    #     del df_opt8 #clean memory
    #     del output_opt8 #clean memory

    #     df_CoT12 = pd.read_json("CoT_Graph_hypothesis\combined_KG_vectors_['CoT', 'KG']_12_data.json")
    #     output_CoT12 = tester.test_all_dataset(answers_datafrane=df_CoT12) # Corrected variable name
    #     output_CoT12.to_excel(writer, index=False, sheet_name="CoT-KG12") # Corrected variable name
    #     del df_CoT12 #clean memory
    #     del output_CoT12 #clean memory
    # excel_path: str = "evaluation\Evaluation_search_3rd.xlsx"
    # with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement
    #     # df_Hybrid = pd.read_json("search_hypothesis\combined_basic_m_enriched_vectors_['Hybrid', '2ndAttempt']_6_data.json")
    #     # output_Hybrid = tester.test_all_dataset(answers_datafrane=df_Hybrid)
    #     # output_Hybrid.to_excel(writer, index=False, sheet_name="Hybrid")
    #     # del df_Hybrid  # clean memory
    #     # del output_Hybrid  # clean memory

    #     df_HyDE = pd.read_json("search_hypothesis\combined_basic_m_enriched_vectors_['HyDE', '2ndAttempt']_6_data.json")
    #     output_HyDE = tester.test_all_dataset(answers_datafrane=df_HyDE) # Corrected variable name
    #     output_HyDE.to_excel(writer, index=False, sheet_name="HyDE") # Corrected variable name
    #     del df_HyDE #clean memory
    #     del output_HyDE #clean memory

    # excel_path: str = "evaluation\Evaluation_2nd_rerankers.xlsx"
    # with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement

    #     df_gpt = pd.read_json("rerankers_hypothesis\combined_basic_m_enriched_vectors_['Rerank_GPT', '2ndAttempt', 'basic']_data.json")
    #     output_gpt = tester.test_all_dataset(answers_datafrane=df_gpt) # Corrected variable name
    #     output_gpt.to_excel(writer, index=False, sheet_name="gpt") # Corrected variable name
    #     del df_gpt #clean memory
    #     del output_gpt #clean memory

    # excel_path: str = "evaluation\Evaluation_fix_Ks.xlsx"
    # with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement
    #     df_4 = pd.read_json("K-hypothesis_answers\combined_basic_vectors_['basic']_4_data.json")
    #     output_4 = tester.test_all_dataset(answers_datafrane=df_4)
    #     output_4.to_excel(writer, index=False, sheet_name="four")
    #     del df_4  # clean memory
    #     del output_4  # clean memory

    #     df_6 = pd.read_json("K-hypothesis_answers\combined_basic_vectors_['basic']_6_data.json")
    #     output_6 = tester.test_all_dataset(answers_datafrane=df_6)
    #     output_6.to_excel(writer, index=False, sheet_name="six")
    #     del df_6  # clean memory
    #     del output_6 # clean memory

    #     df_8 = pd.read_json("K-hypothesis_answers\combined_basic_vectors_['basic']_8_data.json")
    #     output_8 = tester.test_all_dataset(answers_datafrane=df_8)
    #     output_8.to_excel(writer, index=False, sheet_name="eight")
    #     del df_8  # clean memory
    #     del output_8 # clean memory

    #     df_10 = pd.read_json("K-hypothesis_answers\combined_basic_vectors_['basic']_10_data.json")
    #     output_10 = tester.test_all_dataset(answers_datafrane=df_10)
    #     output_10.to_excel(writer, index=False, sheet_name="ten")
    #     del df_10  # clean memory
    #     del output_10 # clean memory

    #     df_12 = pd.read_json("K-hypothesis_answers\combined_basic_vectors_['basic']_12_data.json")
    #     output_12 = tester.test_all_dataset(answers_datafrane=df_12)
    #     output_12.to_excel(writer, index=False, sheet_name="twelve")
    #     del df_12  # clean memory
    #     del output_12 # clean memory

    # excel_path: str = "evaluation\Evaluation_graph.xlsx"
    # with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement
    #     df_graph = pd.read_json("graph_hypothesis\combined_KG_vectors_['KG']_12.json_data.json")
    #     output_graph = tester.test_all_dataset(answers_datafrane=df_graph)
    #     output_graph.to_excel(writer, index=False, sheet_name="Graph_12k")
    #     del df_graph  # clean memory
    #     del output_graph # clean memory
    # excel_path: str = "evaluation\Evaluation_LargeK.xlsx"
    # with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement
    #     df_largeK = pd.read_json("K-hypothesis_answers\combined_basic_m_enriched_vectors_['basic']_30.json_data.json")
    #     output_largeK = tester.test_all_dataset(answers_datafrane=df_largeK)
    #     output_largeK.to_excel(writer, index=False, sheet_name="30-k")
    #     del df_largeK  # clean memory
    #     del output_largeK # clean memory


    # excel_path: str = "evaluation\Evaluation_rerankers.xlsx"
    # with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement
    #     df_cross = pd.read_json("rerankers_hypothesis\combined_basic_m_enriched_vectors_['basic', 'Rerank_CrossEncoder']_data.json")
    #     output_cross = tester.test_all_dataset(answers_datafrane=df_cross)
    #     output_cross.to_excel(writer, index=False, sheet_name="cross_encoder")
    #     del df_cross  # clean memory
    #     del output_cross # clean memory

    #     df_gpt = pd.read_json("rerankers_hypothesis\combined_basic_m_enriched_vectors_['basic', 'Rerank_GPT']_data.json")
    #     output_gpt = tester.test_all_dataset(answers_datafrane=df_gpt) # Corrected variable name
    #     output_gpt.to_excel(writer, index=False, sheet_name="gpt") # Corrected variable name
    #     del df_gpt #clean memory
    #     del output_gpt #clean memory

    # excel_path: str = "evaluation\Evaluation_peripheral.xlsx"
    # with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement
    #     # df_basic_peripheral = pd.read_json("peripheral_hypothesis\combined_basic_m_enriched_vectors_['basic', 'Rerank_Peripheral']_data.json")
    #     # output_basic_peripheral = tester.test_all_dataset(answers_datafrane=df_basic_peripheral)
    #     # output_basic_peripheral.to_excel(writer, index=False, sheet_name="basic_peripheral")
    #     # del df_basic_peripheral  # clean memory
    #     # del output_basic_peripheral # clean memory

    #     df_hybrid_peripheral = pd.read_json("peripheral_hypothesis\combined_basic_m_enriched_vectors_['Hybrid', 'Rerank_Peripheral']_data.json")
    #     output_hybrid_peripheral = tester.test_all_dataset(answers_datafrane=df_hybrid_peripheral) # Corrected variable name
    #     output_hybrid_peripheral.to_excel(writer, index=False, sheet_name="hybrid_peripheral") # Corrected variable name
    #     del df_hybrid_peripheral #clean memory
    #     del output_hybrid_peripheral #clean memory
        
    # excel_path: str = "evaluation\Evaluation_search.xlsx"
    # with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement
    #     df_Hybrid = pd.read_json("search_hypothesis\combined_basic_m_enriched_vectors_['Hybrid']_6.json_data.json")
    #     output_Hybrid = tester.test_all_dataset(answers_datafrane=df_Hybrid)
    #     output_Hybrid.to_excel(writer, index=False, sheet_name="Hybrid")
    #     del df_Hybrid  # clean memory
    #     del output_Hybrid  # clean memory

        # df_basic = pd.read_json("combined_answers\combined_basic_vectors_data.json")
        # output_basic = tester.test_all_dataset(answers_datafrane=df_basic)
        # output_basic.to_excel(writer, index=False, sheet_name="basic")
        # del df_basic  # clean memory
        # del output_basic  # clean memory

        # df_HyDE = pd.read_json("search_hypothesis\combined_basic_m_enriched_vectors_['HyDE']_6.json_data.json")
        # output_HyDE = tester.test_all_dataset(answers_datafrane=df_HyDE) # Corrected variable name
        # output_HyDE.to_excel(writer, index=False, sheet_name="HyDE") # Corrected variable name
        # del df_HyDE #clean memory
        # del output_HyDE #clean memory

    # excel_path: str = "evaluation\Evaluation_chunking.xlsx"
    # with pd.ExcelWriter(excel_path) as writer:  # Use ExcelWriter in a 'with' statement
    #     df_enriched = pd.read_json("Chunking_hypothesis\combined_basic_m_enriched_vectors_['basic']_6_data.json")
    #     output_enriched = tester.test_all_dataset(answers_datafrane=df_enriched)
    #     output_enriched.to_excel(writer, index=False, sheet_name="enriched")
    #     del df_enriched  # clean memory
    #     del output_enriched  # clean memory

    #     # df_basic = pd.read_json("combined_answers\combined_basic_vectors_data.json")
    #     # output_basic = tester.test_all_dataset(answers_datafrane=df_basic)
    #     # output_basic.to_excel(writer, index=False, sheet_name="basic")
    #     # del df_basic  # clean memory
    #     # del output_basic  # clean memory

    #     df_semantic = pd.read_json("Chunking_hypothesis\combined_semantically_split_vectors_['basic']_6_data.json")
    #     output_semantic = tester.test_all_dataset(answers_datafrane=df_semantic) # Corrected variable name
    #     output_semantic.to_excel(writer, index=False, sheet_name="semantic") # Corrected variable name
    #     del df_semantic #clean memory
    #     del output_semantic #clean memory

    
        
    


    # excel_path  = "question_answer_dataset.xlsx"
    # input_df = pd.read_excel(excel_path, sheet_name="to_be_tested")
    # out = tester.test_all_dataset(input_df)
    # out.to_excel(excel_path, index=False, sheet_name="results")

    #     question = "What is the difference between BBA and VFA?"
    #     correct_context = """18.  The VFA measurement model is a modification of the General BBA model reflecting the fact that the consideration which the reporting entity receives can be considered to be a Variable Fee. 
    #     The General BBA model, which is applicable to non-par contracts, forms the basis of the VFA model (and of the BBA indirect par model for participating contracts that fail the VFA scope test)"""
        
    #     answer = """The General BBA and VFA measure fulfillment cash flows in the same way, but differences arise for changes in fulfillment cash flows due to changes in discount rates and other financial variables. 
    #             All changes are reported in P&L and/or OCI for the General BBA model, while for VFA, the CSM is adjusted to reflect changes in the Variable Fee, which includes changes in discount rates and some other financial and operating variables.
    #             The VFA measurement model is a modification of the General BBA model reflecting the fact that the consideration which the reporting entity receives can be considered a Variable Fee. (Sources: Accounting Policies 1.5.1, Analysis of Change 7.1)"""

        
    #     # Supporting context chunks
    #     context = ["""17.  Participating contracts are insurance contracts which, in addition to benefits guaranteed at 
    # contract inception, offer policyholders an opportunity to participate in the return of a pool of 
    # assets and/or surplus generated on other sources of profit (e.g. mortality or expenses). The 
    # Standard differentiates between participating contracts with and without direct participating 
    # features based on whether the contracts pass the test for the Variable Fee approach eligibility 
    # at inception. 

    # 18.  The VFA measurement model is a modification of the General BBA model reflecting the fact 
    # that the consideration which the reporting entity receives can be considered to be a Variable 
    # Fee. The General BBA model, which is applicable to non-par contracts, forms the basis of the 
    # VFA model (and of the BBA indirect par model for participating contracts that fail the VFA 
    # scope test).""", """19. Insurance contracts with direct participation features are primarily insurance contracts that are 
    # substantially investment-related service contracts under which a reporting entity promises to 
    # pay policyholders an amount that is equal to the fair value of underlying items, less a Variable 
    # Fee (insurer’s fee in exchange for the service provided under the contract)."""
    # ,"""20. Obligations under contracts with direct participating features according to paragraph
    # IFRS17.B104 consist of:
    # • Underlying items: Obligation to pay the policyholder an amount equal to the fair value of the 
    # underlying items. 
    # • Variable Fee: According to paragraph IFRS17.B104(b) Variable Fee is defined as:
    # The amount of the entity’s share of the underlying items; less
    # Fulfillment Cash Flows that do not vary based on the returns of the underlying items.""",
    # """25. For with-profit non-unit linked contracts, which may participate in other results (e.g. risk and 
    # expenses) and unit linked contracts which participate in other results, the underlying item does 
    # not need to be a portfolio of financial assets. In practice, the underlying item for participating 
    # contracts is often connected to how the policyholder participation is derived: either as a result 
    # of local regulation or management decisions or common market practices. Amongst others this 
    # could comprise of 
    # (a) statutory gross of tax profits generated (investment, mortality, lapse, expense etc. results) 
    # adjusted for relevant items when deemed to be necessary (e.g. bonus fund reserve),
    # excluding any IFRS17 non-qualified expenses as these by definition should not be included 
    # in variable fee and fulfillment cash flows (non-qualifying expenses are reflected in the 
    # policyholder benefits through profit sharing mechanism, but not included in fulfilment cash 
    # flows as expenses).
    # (b) risk, cost and investment processes comprising of inflows and outflows by source,
    # (c) guaranteed benefits less premiums"""]
#     # print(asyncio.run(tester.faithfulness_metric(answer=answer, context_chunks=context)))
#     # print(tester.answer_relevance_metric(answer=answer, question=question))
    # print(asyncio.run(tester.context_relevancy_metric(question=question, context_chunks=context)))
    
#     precission, recall = tester.calculate_precision_recall(retrieved_chunks="\n\n".join(context), ground_truth_text=correct_context)
#     print(precission, recall)
#     print(tester.f1_score(precision=precission, recall=recall))