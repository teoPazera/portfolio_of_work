import pandas as pd
from rag_model import RAG
from typing import Literal
import os
import json

class Answer_generator:
    companies: list[str]
    rag_model: RAG
    def __init__(self) -> None:
        """class to help with generating testing question answer dataset to evaluate models settings
        """
        self.rag_model = RAG()
        self.companies = ["ABF", "BASF", "Bosch", "BP", "Holcim", "Marriott", "RWE"]
    
    def answers_one_company(self, k: int, company_name: Literal["ABF", "BASF", "Bosch", "BP", "Holcim", "Marriott", "RWE"], vec_db_path: str, 
                            heuristic: list[Literal["basic", "HyDE", "Hybrid", "Rerank-GPT", "Rerank-Peripheral", "Rerank-CrossEncoder"]],
                            folder: str) -> pd.DataFrame:
        """Function to generate answer for all of the questions in report using given heuristic
        k: int -> how many docs to retrieve for one question answering
        company_name: str -> company for which we generate answer
        vec_db: str -> path to vec db we want to use 
        heuristic: list -> name of the heuristics we want to use to try and get better result
        folder: str -> path to folder where to store the answers"""
        #print(vec_db_path)
        self.rag_model.load_db(company_name=company_name, vec_db_path=vec_db_path)
        questions_df = pd.read_excel("questions_and_answers\questions_sheet.xlsx")
        answers_df = pd.DataFrame(columns=["question", "answer", "context"])
        for index, row in questions_df.iterrows():
            question: str = row["Questions"]
            form_of_answer: str = row["Format of answer"] 
            explanation: str = row["Explanation"]  
            if pd.isna(explanation):
                explanation = ""
            answer_dict: dict[str,str] = self.rag_model.answer_query(query=question, 
                                        form_of_answer=form_of_answer, explanation=explanation, heuristic=heuristic,
                                        k=k)
            answer_dict["question"] = question
            answers_df = pd.concat([answers_df, pd.DataFrame([answer_dict])], ignore_index=True)
            
    
        answers_df.to_json(f"{folder}\{company_name}-{vec_db_path}-{heuristic}-{str(k)}.json")
        return answers_df
    
    def answer_all_companies(self, k: int,  vec_db_path: str, 
                            heuristic: list[Literal["basic", "KG","HyDE", "Hybrid",
                             "Rerank-GPT", "Rerank-Peripheral", "Rerank-CrossEncoder"]],
                            folder: str) -> None:
        """iterates through all companies available to fill out the report

        Args:
            k (int): number of chunks to retrieve
            vec_db_path (str): vector database to use
            heuristic (list[Literal[&quot;basic&quot;, &quot;KG&quot;,&quot;HyDE&quot;, &quot;Hybrid&quot;, &quot;Rerank): heuristic to use when creating answers
            folder (str): folder where to store json output files
        """
        for company in self.companies: 
            self.answers_one_company(company_name=company, vec_db_path=vec_db_path, heuristic=heuristic, k=k, folder=folder)
        print("average num of tokens used per call", self.rag_model.tokens/147)
    
    def create_combined_dataframe(self, folder: str):
        """
        Combines data from multiple JSON files into a single Pandas DataFrame, merging answers with the same heuristic and k,
        and saves it as JSON. Handles missing 'correct_answers.xlsx' gracefully.

        Args:
            directory (str): The directory containing the JSON files.
            companies (list): A list of company names corresponding to sheets in correct_answers.xlsx

        Returns:
            dict: A dictionary of the paths to the created JSON files, keyed by vector type.
        """

        data = {}  # Store data grouped by vector type, heuristic, and k
        correct_answers_dfs = {} # Store correct answers DataFrames for each company

        # Load correct answers from each sheet in the Excel file
        try:
            excel_file = pd.ExcelFile("questions_and_answers\correct_answers.xlsx")
            for company in self.companies:
                if company in excel_file.sheet_names:
                    correct_answers_dfs[company] = pd.read_excel(excel_file, sheet_name=company)
                    correct_answers_dfs[company] = correct_answers_dfs[company].rename(
                        columns={
                            "Question": "question",
                            "Manual output": "manual_answer",
                            "Desired context": "ground_truth_context",
                        }
                    )
                    correct_answers_dfs[company] = correct_answers_dfs[company][
                        ["question", "manual_answer", "ground_truth_context"]
                    ]
                else:
                    print(f"Warning: Sheet '{company}' not found in correct_answers.xlsx.")
                    correct_answers_dfs[company] = None # Handle missing sheet
        except FileNotFoundError:
            print("Warning: correct_answers.xlsx not found.  Continuing without it.")
            correct_answers_dfs = {company: None for company in self.companies} # All companies are None
        except Exception as e:
            print(f"Warning: Error reading correct_answers.xlsx: {e}. Continuing without it.")
            correct_answers_dfs = {company: None for company in self.companies} # All companies are None


        for filename in os.listdir(folder):
            if not filename.endswith(".json"):
                continue  # Skip non-JSON files

            parts = filename.split("-")
            if len(parts) != 4:  # Expecting 4 parts (company-vector_type-heuristic-k.json)
                continue  # Skip files that don't match the expected naming pattern

            company = parts[0]
            vector_type = parts[1]
            heuristic = parts[2]
            k = parts[3].replace(".json", "")  # Remove the file extension


            key = (vector_type, heuristic, k) # Create a tuple key

            if key not in data:
                data[key] = []

            try:
                with open(os.path.join(folder, filename), "r") as f:
                    answers_data = json.load(f)  # Assuming it's a list of dictionaries
                    answers_df = pd.DataFrame(answers_data)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON in {filename}. Skipping.")
                continue
            except Exception as e:
                print(f"Warning: Error reading {filename}: {e}. Skipping.")
                continue

            # Merge with correct answers if available
            if company in correct_answers_dfs and correct_answers_dfs[company] is not None:
                merged_df = pd.merge(
                    answers_df, correct_answers_dfs[company], on="question", how="left"
                )
            else:
                merged_df = answers_df.copy()
                merged_df["manual_answer"] = None
                merged_df["ground_truth_context"] = None

            # Rename columns to match desired output
            merged_df = merged_df.rename(
                columns={"answer": "answer_of_model", "context": "retrieved_context"}
            )

            # Add company name
            merged_df["company"] = company

            data[key].append(merged_df)

        # Concatenate DataFrames for each vector type, heuristic, and k
        combined_data = {}
        for key, dfs in data.items():
            vector_type, heuristic, k = key
            combined_data[key] = pd.concat(dfs, ignore_index=True)

        # Save to JSON files
        output_files = {}
        for key, df in combined_data.items():
            vector_type, heuristic, k = key
            output_filename = f"combined_{vector_type}_{heuristic}_{k}_data.json"
            output_path = os.path.join(folder, output_filename)
            df.to_json(output_path, orient="records")
            output_files[vector_type] = output_path

        return output_files
    
if __name__ == "__main__":
    import os
    os.environ['CURL_CA_BUNDLE'] = r'C:\Users\TEO.PAZERA\OneDrive - Zurich Insurance\Desktop\bakalarka_cody\Bakalarka_rag\huggingface.co.crt'
    ag = Answer_generator()
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    k = 8
    folder="graph_hypothesis"
    vec_db_path="KG_vectors"
    h = ["KG", "RerankGPT"]
    ag.answer_all_companies(k=k, vec_db_path=vec_db_path, heuristic=h, folder=folder)

    ag.create_combined_dataframe(folder=folder)

    # ,
    # for company in ["ABF", "BASF", "Bosch", "BP", "Holcim", "Marriott", "RWE"]: # plain text, enriched text, semantic text comparison
    #     k = 6
    #     h = ["Rerank-GPT", "2ndAttempt", "basic"]
    #     ag.answers_one_company(company_name=company, vec_db_path="basic_m_enriched_vectors", heuristic=h, k=k, folder="rerankers_hypothesis")

    


