import os
import pandas as pd
import json

def create_combined_dataframe(directory: str, companies: list[str]):
    """
    Combines data from multiple files into a single Pandas DataFrame, merging answers with the same heuristic and k,
    and saves it as JSON.

    Args:
        directory (str): The directory containing the files.
        companies (list): A list of company names corresponding to sheets in correct_answers.xlsx

    Returns:
        dict: A dictionary of the paths to the created JSON files, keyed by vector type.
    """

    data = {}  # Store data grouped by vector type, heuristic, and k
    correct_answers_dfs = {} # Store correct answers DataFrames for each company

    # Load correct answers from each sheet in the Excel file
    try:
        excel_file = pd.ExcelFile("questions_and_answers\correct_answers.xlsx")
        for company in companies:
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
        correct_answers_dfs = {company: None for company in companies} # All companies are None
    except Exception as e:
        print(f"Warning: Error reading correct_answers.xlsx: {e}. Continuing without it.")
        correct_answers_dfs = {company: None for company in companies} # All companies are None

    for filename in os.listdir(directory):
        if filename.endswith(".xlsx") and filename == "correct_answers.xlsx":
            continue # Skip the correct_answers file itself.
        if filename.endswith(".xlsx") and filename == "questions_sheet.xlsx":
            continue

        parts = filename.split("-")
        if len(parts) < 4:  # Expecting at least 4 parts (company-vector_type-heuristic-k.xlsx)
            continue  # Skip files that don't match the expected naming pattern

        company = parts[0]
        vector_type = parts[1]
        heuristic = parts[2]
        k = parts[3].replace(".json", "")  # Remove the file extension
        print(k)
        key = (vector_type, heuristic, k) # Create a tuple key

        if key not in data:
            data[key] = []

        try:
            with open(os.path.join(directory, filename), "r") as f: # Changed to read excel
               answers_data = json.load(f)  # Assuming it's a list of dictionaries # Changed to read excel
               answers_df = pd.DataFrame(answers_data)  # Changed to read excel
        
        except json.JSONDecodeError: # Changed to read excel
           print(f"Warning: Could not decode JSON in {filename}. Skipping.")# Changed to read excel
           continue # Changed to read excel
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
        output_path = os.path.join(directory, output_filename)
        df.to_json(output_path, orient="records")
        output_files[vector_type] = output_path

    return output_files


# Example Usage:
directory_path = "opt_hypothesis"  
companies_list = ["ABF", "BASF", "Bosch", "BP", "Holcim", "Marriott", "RWE"]
output_files = create_combined_dataframe(directory_path, companies_list)

for vector_type, filepath in output_files.items():
    print(f"Combined data for {vector_type} saved to: {filepath}")