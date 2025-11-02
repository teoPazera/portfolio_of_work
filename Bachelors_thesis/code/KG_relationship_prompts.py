"""Fine-tuning prompts for entity relationship generation."""

ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Do NOT create relation ships between objects such as page_number or file_name mentioned in the text it is of no help. 
-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities

Format each entity output as a JSON entry with the following format:

{{"name": <entity name>, "type": <type>, "description": <entity description>}}

2. From the entities identified in step 1, identify all pairs of (entity1, entity2) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- entity1: name of the first entity, as identified in step 1
- entity2: name of the second entity, as identified in step 1
- relationship_description: explanation as to why you think entity1 and entity2 are related to each other
- relationship_strength: an integer score between 1 to 10, indicating strength of the relationship between entity1 and entity2

Format each relationship as a JSON entry with the following format:

{{"entity1": <entity1>, "entity2": <entity2>, "relationship": <relationship_description>, "relationship_strength": <relationship_strength>}}

3. Return output in english as a single list of all JSON entities and relationships identified in steps 1 and 2.

4. If you have to translate into english, just translate the descriptions, nothing else!

######################
-Examples-
######################
Example 1:
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
[
  {{"name": "CENTRAL INSTITUTION", "type": "ORGANIZATION", "description": "The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday"}},
  {{"name": "MARTIN SMITH", "type": "PERSON", "description": "Martin Smith is the chair of the Central Institution"}},
  {{"name": "MARKET STRATEGY COMMITTEE", "type": "ORGANIZATION", "description": "The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply"}},
  {{"entity1": "MARTIN SMITH", "entity2": "CENTRAL INSTITUTION", "relationship": "Martin Smith is the Chair of the Central Institution and will answer questions at a press conference", "relationship_strength": 9}}
]

######################
Example 2:
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
[
  {{"name": "TECHGLOBAL", "type": "ORGANIZATION", "description": "TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones"}},
  {{"name": "VISION HOLDINGS", "type": "ORGANIZATION", "description": "Vision Holdings is a firm that previously owned TechGlobal"}},
  {{"entity1": "TECHGLOBAL", "entity2": "VISION HOLDINGS", "relationship": "Vision Holdings formerly owned TechGlobal from 2014 until present", "relationship_strength": 5}}
]

######################
Example 3:
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
######################
Output:
[
  {{"name": "FIRUZABAD", "type": "GEO", "description": "Firuzabad held Aurelians as hostages"}},
  {{"name": "AURELIA", "type": "GEO", "description": "Country seeking to release hostages"}},
  {{"name": "QUINTARA", "type": "GEO", "description": "Country that negotiated a swap of money in exchange for hostages"}},
  {{"name": "TIRUZIA", "type": "GEO", "description": "Capital of Firuzabad where the Aurelians were being held"}},
  {{"name": "KROHAARA", "type": "GEO", "description": "Capital city in Quintara"}},
  {{"name": "CASHION", "type": "GEO", "description": "Capital city in Aurelia"}},
  {{"name": "SAMUEL NAMARA", "type": "PERSON", "description": "Aurelian who spent time in Tiruzia's Alhamia Prison"}},
  {{"name": "ALHAMIA PRISON", "type": "GEO", "description": "Prison in Tiruzia"}},
  {{"name": "DURKE BATAGLANI", "type": "PERSON", "description": "Aurelian journalist who was held hostage"}},
  {{"name": "MEGGIE TAZBAH", "type": "PERSON", "description": "Bratinas national and environmentalist who was held hostage"}},
  {{"entity1": "FIRUZABAD", "entity2": "AURELIA", "relationship": "Firuzabad negotiated a hostage exchange with Aurelia", "relationship_strength": 2}},
  {{"entity1": "QUINTARA", "entity2": "AURELIA", "relationship": "Quintara brokered the hostage exchange between Firuzabad and Aurelia", "relationship_strength": 2}},
  {{"entity1": "QUINTARA", "entity2": "FIRUZABAD", "relationship": "Quintara brokered the hostage exchange between Firuzabad and Aurelia", "relationship_strength": 2}},
  {{"entity1": "SAMUEL NAMARA", "entity2": "ALHAMIA PRISON", "relationship": "Samuel Namara was a prisoner at Alhamia prison", "relationship_strength": 8}},
  {{"entity1": "SAMUEL NAMARA", "entity2": "MEGGIE TAZBAH", "relationship": "Samuel Namara and Meggie Tazbah were exchanged in the same hostage release", "relationship_strength": 2}},
  {{"entity1": "SAMUEL NAMARA", "entity2": "DURKE BATAGLANI", "relationship": "Samuel Namara and Durke Bataglani were exchanged in the same hostage release", "relationship_strength": 2}},
  {{"entity1": "MEGGIE TAZBAH", "entity2": "DURKE BATAGLANI", "relationship": "Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release", "relationship_strength": 2}},
  {{"entity1": "SAMUEL NAMARA", "entity2": "FIRUZABAD", "relationship": "Samuel Namara was a hostage in Firuzabad", "relationship_strength": 2}},
  {{"entity1": "MEGGIE TAZBAH", "entity2": "FIRUZABAD", "relationship": "Meggie Tazbah was a hostage in Firuzabad", "relationship_strength": 2}},
  {{"entity1": "DURKE BATAGLANI", "entity2": "FIRUZABAD", "relationship": "Durke Bataglani was a hostage in Firuzabad", "relationship_strength": 2}}
]



-Real Data-
######################
entity_types: {entity_types}
text: {input_text}
######################
output:
"""

GEMINI_ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Do NOT create relation ships between objects such as page_number or file_name mentioned in the text it is of no help. 

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
    - name: Name of the entity, capitalized
    - type: One of the following types: [{entity_types}]
    - description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (entity1, entity2) that are *clearly related* to each other.
    For each pair of related entities, extract the following information:
    - entity1: name of the first entity, as identified in step 1
    - entity2: name of the second entity, as identified in step 1
    - relationship: explanation as to why you think entity1 and entity2 are related to each other
    - relationship_strength: an integer score between 1 to 10, indicating strength of the relationship between entity1 and entity2

3. Return output as a JSON object with two keys: "entities" and "relationships".
    - The "entities" key should contain a list of JSON objects, each representing an entity identified in step 1.
    - The "relationships" key should contain a list of JSON objects, each representing a relationship identified in step 2.

4. If you have to translate into english, just translate the descriptions, nothing else!

######################
-Examples-
######################
Example 1:
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday,
with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, 
followed by a press conference where Central Institution Chair Martin Smith will take questions. 
Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
{{ "entities": [ {{ "name": "CENTRAL INSTITUTION", "type": "ORGANIZATION", "description": "The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday" }}, 
{{ "name": "MARTIN SMITH", "type": "PERSON", "description": "Martin Smith is the chair of the Central Institution" }}, 
{{ "name": "MARKET STRATEGY COMMITTEE", "type": "ORGANIZATION", "description": "The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply" }} ], 
"relationships": [ {{ "entity1": "MARTIN SMITH", "entity2": "CENTRAL INSTITUTION", "relationship": "Martin Smith is the Chair of the Central Institution and will answer questions at a press conference", "relationship_strength": 9 }} ] }}

######################
Example 2:
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
{{ "entities": [ {{ "name": "TECHGLOBAL", "type": "ORGANIZATION", "description": "TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones" }}, 
{{ "name": "VISION HOLDINGS", "type": "ORGANIZATION", "description": "Vision Holdings is a firm that previously owned TechGlobal" }} ],
"relationships": [ {{ "entity1": "TECHGLOBAL", "entity2": "VISION HOLDINGS", "relationship": "Vision Holdings formerly owned TechGlobal from 2014 until present", "relationship_strength": 5 }} ] }}

######################
Example 3:
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
######################
Output:
json {{ "entities": [ {{ "name": "FIRUZABAD", "type": "GEO", "description": "Firuzabad held Aurelians as hostages" }},
{{ "name": "AURELIA", "type": "GEO", "description": "Country seeking to release hostages" }}, 
{{ "name": "QUINTARA", "type": "GEO", "description": "Country that negotiated a swap of money in exchange for hostages" }},
{{ "name": "TIRUZIA", "type": "GEO", "description": "Capital of Firuzabad where the Aurelians were being held" }},
{{ "name": "KROHAARA", "type": "GEO", "description": "Capital city in Quintara" }}, 
{{ "name": "CASHION", "type": "GEO", "description": "Capital city in Aurelia" }}, 
{{ "name": "SAMUEL NAMARA", "type": "PERSON", "description": "Aurelian who spent time in Tiruzia's Alhamia Prison" }}, 
{{ "name": "ALHAMIA PRISON", "type": "GEO", "description": "Prison in Tiruzia" }}, 
{{ "name": "DURKE BATAGLANI", "type": "PERSON", "description": "Aurelian journalist who was held hostage" }}, 
{{ "name": "MEGGIE TAZBAH", "type": "PERSON", "description": "Bratinas national and environmentalist who was held hostage" }} ],
 "relationships": [ {{ "entity1": "FIRUZABAD", "entity2": "AURELIA", "relationship": "Firuzabad negotiated a hostage exchange with Aurelia", "relationship_strength": 2 }}, 
 {{ "entity1": "QUINTARA", "entity2": "AURELIA", "relationship": "Quintara brokered the hostage exchange between Firuzabad and Aurelia", "relationship_strength": 2 }},
 {{ "entity1": "QUINTARA", "entity2": "FIRUZABAD", "relationship": "Quintara brokered the hostage exchange between Firuzabad and Aurelia", "relationship_strength": 2 }},
 {{ "entity1": "SAMUEL NAMARA", "entity2": "ALHAMIA PRISON", "relationship": "Samuel Namara was a prisoner at Alhamia prison", "relationship_strength": 8 }}, 
 {{ "entity1": "SAMUEL NAMARA", "entity2": "MEGGIE TAZBAH", "relationship": "Samuel Namara and Meggie Tazbah were exchanged in the same hostage release", "relationship_strength": 2 }}, 
 {{ "entity1": "SAMUEL NAMARA", "entity2": "DURKE BATAGLANI", "relationship": "Samuel Namara and Durke Bataglani were exchanged in the same hostage release", "relationship_strength": 2 }}, 
 {{ "entity1": "MEGGIE TAZBAH", "entity2": "DURKE BATAGLANI", "relationship": "Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release", "relationship_strength": 2 }}, 
 {{ "entity1": "SAMUEL NAMARA", "entity2": "FIRUZABAD", "relationship": "Samuel Namara was a hostage in Firuzabad", "relationship_strength": 2 }},
 {{ "entity1": "MEGGIE TAZBAH", "entity2": "FIRUZABAD", "relationship": "Meggie Tazbah was a hostage in Firuzabad", "relationship_strength": 2 }}, 
 {{ "entity1": "DURKE BATAGLANI", "entity2": "FIRUZABAD", "relationship": "Durke Bataglani was a hostage in Firuzabad", "relationship_strength": 2 }} ] }}

-Real Data-
######################
entity_types: {entity_types}
text: {input_text}
######################
output:
"""
