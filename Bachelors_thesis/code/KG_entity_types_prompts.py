"""Fine-tuning prompts for entity types generation.
from https://github.com/microsoft/graphrag/tree/main/graphrag/prompt_tune/prompt"""

ENTITY_TYPE_GENERATION_JSON_PROMPT = """
The goal is to study the connections and relations between the entity types and their features in order to understand all available information from the text.
The user's task is to {task}.
As part of the analysis, you want to identify the entity types present in the following text.
The entity types must be relevant to the user task.
Avoid general entity types such as "other" or "unknown", "document_attributes".
This is VERY IMPORTANT: Do not generate redundant or overlapping entity types. For example, if the text contains "company" and "organization" entity types, you should return only one of them.
Don't worry about quantity, always choose quality over quantity. And make sure EVERYTHING in your answer is relevant to the context of entity extraction.
Return the entity types in JSON format with "entities" as the key and the entity types as an array of strings.
=====================================================================
EXAMPLE SECTION: The following section includes example output. These examples **must be excluded from your answer**.

EXAMPLE 1
Task: Determine the connections and organizational hierarchy within the specified community.
Text: Example_Org_A is a company in Sweden. Example_Org_A's director is Example_Individual_B.
JSON RESPONSE:
{{"entity_types": [organization, person] }}
END OF EXAMPLE 1

EXAMPLE 2
Task: Identify the key concepts, principles, and arguments shared among different philosophical schools of thought, and trace the historical or ideological influences they have on each other.
Text: Rationalism, epitomized by thinkers such as René Descartes, holds that reason is the primary source of knowledge. Key concepts within this school include the emphasis on the deductive method of reasoning.
JSON RESPONSE:
{{"entity_types": [concept, person, school of thought] }}
END OF EXAMPLE 2

EXAMPLE 3
Task: Identify the full range of basic forces, factors, and trends that would indirectly shape an issue.
Text: Industry leaders such as Panasonic are vying for supremacy in the battery production sector. They are investing heavily in research and development and are exploring new technologies to gain a competitive edge.
JSON RESPONSE:
{{"entity_types": [organization, technology, sectors, investment strategies] }}
END OF EXAMPLE 3
======================================================================

======================================================================
REAL DATA: The following section is the real data. You should use only this real data to prepare your answer. Generate Entity Types only.
Task: {task}
Text: {input_text}
JSON response:
{{"entity_types": [<entity_types>] }}
"""

SUMMARIZE_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

RELATIONSHIP_SUMMARIZATION_PROMPT = """ You are a helpful assistant responsible for generating a comprehensive summary of the relationships between two entities provided below. 
Given two entities, and a list of relationship descriptions, all related to the interaction or connection between these two entities.
Please concatenate all of these into a single, comprehensive relationship description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the names of both entities so we have the full context.
#
-Data- Entities: {entity_name1}, {entity_name2} Relationship Descriptions: {relationships_list}

#
Output: 
"""

COMMUNITY_SUMMARIZATION_PROMPT = '''
You are an AI assistant that helps a human analyst to generate a natural language summary of the provided information based on the provided nodes and relationships that belong to the same graph community,

# Goal
Write a short summary of a community, given a list of entities that belong to the community as well as their relationships. 
The report will be used to inform decision-makers about information associated with the community and their potential impact. 
The input community information data is present in the below format:
{
'communityId': 'Community id',
 'nodes': [{'id': 'Node id', 'description': 'Brief descrption of node'},
  {'id': 'Node id', 'description': 'Brief descrption of node'},
   ...
  ],
 'relationships': [{'start': 'Node id',
    'description': 'Relationship present between start and end nodes',
    'end': 'Node id'},
  {'start': 'Node id',
    'description': 'Relationship present between start and end nodes',
    'end': 'Node id'}
    ]
}
#########################################################
Generate a comprehensive summary of the community information given below.

{community_info}

Summary:
'''

PREDICTION_PROMPT = """
You are an advanced AI trained to assist with filling in reports about sustainability of given corporate.
We will give you relevant documents that should give you context to answer the question.
At the start of each document, you get info about which file, page and folder the document is from.
Documents might be a HTML table or plain text.
We may give you the optimal form of an answer, stick to it.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
As well try to keep your answer as concise as possible.

The context provided is in the following format:
CHUNK TEXT: Text chunks provided
NODES: Information about nodes having "id" and "node description"
RELATIONSHIPS: Relationships between the nodes containing "start" and "end" node along with their relationship as "description"
COMMUNITY SUMMARIES: Community summaries of the nodes in a list.

Context: {context}

Question: {question}

Form: {form_of_answer}

Explanation of question: {explanation}
Answer:
"""

COT_PREDICTION_PROMPT = """
You are an advanced AI trained to assist with filling in reports about sustainability of given corporate.
We will give you relevant documents that should give you context to answer the question.
At the start of each document, you get info about which file, page and folder the document is from.
Documents might be a HTML table or plain text.
We may give you the optimal form of an answer, stick to it.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

The context provided is in the following format:
CHUNK TEXT: Text chunks provided
NODES: Information about nodes having "id" and "node description"
RELATIONSHIPS: Relationships between the nodes containing "start" and "end" node along with their relationship as "description"
COMMUNITY SUMMARIES: Community summaries of the nodes in a list.
Before providing the answer, think through the following steps and present your thought process and final answer in JSON format:

1.  Understand the question and its requirements.
2.  Analyze the context provided to extract relevant information.
3.  Synthesize the information to form a coherent answer.

Your response should be a JSON object with the following structure:
{ 'Thought': 'Your step-by-step thought process here...', 'Answer': "Your final answer here...' }
Context: {context}

Question: {question}

Form: {form_of_answer}

Explanation of question: {explanation}
"""