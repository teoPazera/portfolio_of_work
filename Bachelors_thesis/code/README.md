# RAG Model Evaluation for Sustainability Reports

This repository contains the implementation and evaluation of Retrieval-Augmented Generation (RAG) models for generating sustainability reports. The project compares different RAG models using various metrics and provides tools for generating answers, creating knowledge graphs, and evaluating model performance.

## Project Structure

- **`answer_generator.py`**: Script for generating answers using the RAG models with specific configurations.
- **`creator_of_testing_df.py`**: Utility for creating testing datasets for evaluation.
- **`custom_splitter.py`**: Custom logic for splitting text into meaningful chunks for processing.
- **`evaluation/`**: Contains evaluation results, including metrics and comparisons of different models.
- **`pdf_files/`**: Stores the pdf files used as sources of the information for reports creation
- **`Evaluation_analysis.ipynb`**: Jupyter notebook for analyzing evaluation results.
- **`KG_constructer.py`**: Code for constructing Knowledge Graphs used in Graph RAG models.
- **`KG_entity_types_prompts.py`** and **`KG_relationship_prompts.py`**: Scripts for generating entity types and relationship prompts for Knowledge Graphs.
- **`parsing_model.py`**: Handles parsing and preprocessing of input data.
- **`pdf_cutter.py`**: Utility for splitting PDF files into smaller sections for processing.
- **`basic_m_enriched_vectors/`, `basic_vectors/`, `KG_vectors/` , `semantically_split_vectors`**: Directories containing vectorized representations of data for retrieval.
- **`Chunking_hypothesis/`, `CoT_Graph_hypothesis/`, `CoT_hypothesis/`, `graphRAG_hypothesis/`, `interactions_hypothesis/`, `K-hypothesis_answers/`, `peripheral_hypothesis/`**: Folders containing hypothesis data and model-generated answers in JSON format.
- **`questions_and_answers/`**: Contains question-answer pairs used for testing and evaluation.
- **`images/`**: Stores images used in documentation or analysis.

## Key Features

- **Custom RAG Implementation**: Most of the code is written from scratch, with support for Knowledge Graphs and advanced heuristics.
- **Evaluation Framework**: Tools for comparing models using metrics and saving results in structured formats.
- **Knowledge Graph Support**: Enables enhanced retrieval using Knowledge Graphs.
- **PDF Processing**: Utilities for cutting and processing PDF files for input into the models.

## Installation and Setup

1. Ensure you have Python 3.10.10 installed.
2. Clone this repository.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your Azure OpenAI credentials to a .env file in the root directory.
```bash
OPENAI_API_KEY=<your_openai_api_key>
AZURE_ENDPOINT_COMPLETION=<your_azure_endpoint_completion>
AZURE_ENDPOINT_EMBEDDING=<your_azure_endpoint_embedding>
OPENAI_API_TYPE=azure
OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>
```

## Usage
### Generating Answers
Use the answer_generator.py script to generate answers with specific configurations. Example:
```python
from answer_generator import Answer_generator

ag = Answer_generator()
k = 8
folder = "graph_hypothesis"
vec_db_path = "KG_vectors"
heuristic = ["KG", "RerankGPT"]

ag.answer_all_companies(k=k, vec_db_path=vec_db_path, heuristic=heuristic, folder=folder)
ag.create_combined_dataframe(folder=folder)
```
### Evaluation Models 
Use the creator_of_testing_df.py and evaluation/ folder to evaluate models. Example:
```python 
from testing_framework import Test
import pandas as pd

tester = Test()
excel_path = "evaluation/Evaluation_interactions_KG_rerankGPT.xlsx"

with pd.ExcelWriter(excel_path) as writer:
    df = pd.read_json("graph_hypothesis/combined_KG_vectors_['KG', 'RerankGPT']_8_data.json")
    results = tester.test_all_dataset(answers_dataframe=df)
    results.to_excel(writer, index=False, sheet_name="KG-rerankGPT-8")
```
### Using the RAG Model
Refer to rag_model.py for constructing and using the RAG model. Example:
```python 
from rag_model import RAG

rag = RAG()
rag.load_db(company_name="ABF", vec_db_path="semantically_split_vectors")
question = "Is the company committed to the Paris (Climate) Agreement?"
form_of_answer = "yes/no"
explanation = "The Paris agreement asks to limit global warming to well below 2°C and pursuing efforts to limit it to 1.5°C."

answer = rag.answer_query(query=question, form_of_answer=form_of_answer, explanation=explanation, k=6, heuristic=["CoT", "basic"])
print(answer)
```
## Acknowledgments
- LangChain: Used for some RAG functionalities.
- ChromaDB: Used for vector database operations.
- Special thanks to my advisors and peers for their guidance and support throughout this project.