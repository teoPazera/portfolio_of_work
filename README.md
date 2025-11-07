# My Portfolio of Projects

This repository presents a selection of my academic and professional work. It will guide the reader through the selection based of my judgement of importance of the work I did. 

---

## About Me

My name is Teo Pazera. I am a Master’s student in Data Science at Radboud University (Nijmegen, Netherlands) and currently work as a Data Science Intern at Zurich Insurance. I am based in the Netherlands (previously in the Bratislava Competence Center).

This repository highlights selected projects from my bachelor’s work, industry experience and lastly some of my school projects which are more general data science projects, which i was part of. In my role I implement large language models for workplace automation, develop task-focused AI agents (for example, automating report generation), and build RAG-based chatbots. I also worked on standard data science problems such as text classification using NLP methods or some other predictive modeling. I have over 1.5 years of experience at Zurich and hold a Bachelor’s degree in Data Science from Comenius University (Bratislava).


---

## Education

* **MSc in Data Science** (ongoing) — Radboud University, Nijmegen, Netherlands
* **BSc in Data Science** (2024) — Comenius University, Bratislava, Slovakia

  * Thesis: *Enhancement of RAG Chatbot Performance Using Various Heuristics* (in collaboration with Zurich Insurance)

---

## Professional Experience

* **Data Science Intern, Zurich Insurance** (2023–Present)

  * Hired to be part of newly formed AI working group in Bratislava, covering topics starting from RAG systems, usage of LLMs to automate processes, agentic AI systems.
  * Developed AI applications from scratch for different teams across entire company, starting as proofs-of-concept leading to operating products used to make processes more efficient
  * Contributed to an internal Agentic Hackathon and to prototyping agentic workflows.

---

## Projects

### Bachelor’s Thesis — *Enhancement of RAG Chatbot Performance Using Various Heuristics*

* **Folder**: `Bachelors_Thesis` (contains code for heuristics; proprietary data/knowledge bases excluded).
* **Description**: During my internship I studied how to improve answers produced by a RAG chatbot used for automating ESG reporting. The project tested a variety of heuristics across the retrieval and answer-generation pipeline: chunking strategies, methods for storing text (including LLM-assisted knowledge representations), retrieval and reranking approaches (cross-encoders and LLM-based rerankers), and prompting techniques such as chain-of-thought. The pipeline accepted new customer data from a sustainability analyst, retrieved the most relevant context, and generated structured answers for reporting.
* **Evaluation**: I measured retrieval metrics and had LLM-judged answer quality compared against a gold set. Statistical significance of improvements was tested with the Wilcoxon signed-rank test. The final approach prioritised explainability and efficiency: limiting the context visible to the LLM improved human verifiability, and more complex, costly methods (e.g., graph-based RAG) offered marginal gains that did not justify their cost. The optimized workflow used a multi-step process with an embedding based retrieval stage and an LLM reranker before final answer generation with chain-of-thought prompting.
* **Technologies**: Python, LangChain, LLM APIs, vector retrieval, knowledge representations, and statistical analysis.
* **Outcome**: The solution was adopted by the sustainability team at Zurich and is used in their reporting workflow. (Code in repository excludes proprietary data.)

---

### Professional Projects at Zurich Insurance


1. **Actuarial Domain RAG Chatbot (introductory project)**

   * **Description**: Built a chatbot to query an IFRS 17 guidebook. The system parsed documents for vector retrieval and included a simple user interface and feedback loop. The project provided practical lessons on prompt design, context construction, and the limitations of embedding search for complex domain queries, especially when LLM context windows were smaller.
   * **Technologies**: Python, Chroma vector DB, LLM APIs, parsers, LangChain.

2. **Follow-up RAG systems**

   * **Description**: Implemented several similar RAG pipelines for actuarial documentation. As dataset size increased, new requirements emerged (for example, automatic filtering to improve retrieval efficiency). Some heuristics tested here were later incorporated into my bachelor’s thesis.
   * **Technologies**: Python, Chroma, LLM APIs, parsers, LangChain.

3. **NLP-based Claims Classification**

   * **Description**: Developed an automated classifier to tag insurance claims into established categories. After evaluation, the solution used lighter weight NLP models rather than a pure LLM approach to obtain more reliable and explainable results for this task.
   * **Technologies**: Python, standard NLP toolkits and modeling approaches.

4. **Agentic Model for Hierarchical Insurance Reporting**

   * **Description**: Worked on an agentic system to aggregate and summarise country level presentation slides into regional or group level reporting. The agent used stored slide metadata and a knowledge base of the reporting hierarchy. Given a question, the agent decided whether to retrieve low level slides for a focused answer or to summarise across hierarchical levels and periods for a broader analysis. The agent applied filtering, summarisation, and guideline based reasoning to produce context-aware responses.
   * **Technologies**: LangGraph, LLMs, custom tools for data filtering and knowledge lookup.
   * **Outcome**: The project is in development and received positive feedback from analysts and management during demonstrations.

5. **Agentic Hackathon — Reserving Data Presentation Drafter**

   * **Description**: Team project to build a semi-autonomous agent that drafts presentations from hierarchical reserving data. The agent produced low level commentary and attempted to scale comments up to higher levels of aggregation, highlighting deltas and their origins. The prototype struggled with messy, synthetic data and with translating granular insights into clear high-level commentary, likely due to insufficient contextual signals in the mock dataset.
   * **Technologies**: LangGraph, numerical data processing tools.

## Notes on Repository and Usage

* Code related to Zurich projects is not included here because of NDAs. Where possible I include non-proprietary examples and the thesis codebase.


## Academic Projects 
These projects, completed during my Bachelor's at Comenius University, demonstrate foundational skills in data analysis, visualization, and predictive modeling. All were group efforts and are primarily in Slovak (translations available upon request). Source code and reports are in the `School_Projects` folder.

1. **COVID-19 Genome Sequence Visualization** 
   - **Description**: Analyzed and visualized a dataset of COVID-19 genome sequencing runs to uncover patterns in sequence variations and temporal trends.  
   - **Technologies**: Python, data visualization libraries (Pandas, Matplotlib, Seaborn)
   - **Outcomes**: Produced report investigating given data and trying to find relevant patterns in it. This project built my skills in exploratory data analysis, crucial for preprocessing in AI pipelines.  
  

2. **Slovak Parliamentary Election Prediction from Polls** 
   - **Description**: Used historical polling data to model trends in Slovak political parties, predicting outcomes for upcoming elections by characterizing party behaviors (e.g. some parties under perform in polls as people are not proud to show support).  
   - **Technologies**: Python, parsing of pdfs, time-series analysis, regression models
   - **Outcomes**: Found relevant factors that help build upon and predict the difference between last available poll and final election result.  


3. **Comparative Analysis of Slovak vs. Czech Railway Networks** 
   - **Description**: Evaluated railway systems using network theory metrics and simulations of traveling to assess if the Slovak network underperforms compared to the Czech one.  
   - **Technologies**: Python (NetworkX), graph algorithms
   - **Outcomes**: Quantified inefficiencies in Slovak railway system and showed vulnerabilities in network which might cause entire network to stop functioning

---

## Contact
- LinkedIn: [linkedin.com/in/teo-pazera](https://www.linkedin.com/in/teo-pazera-7520b1380/)
- Email: pazerateo@gmail.com  


