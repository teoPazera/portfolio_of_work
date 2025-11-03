# My Portfolio of projects

Welcome to my portfolio repository. This space showcases a selection of my academic and industry projects. As an aspiring researcher in agentic AI systems, I aim to demonstrate my expertise in building autonomous AI agents, Retrieval-Augmented Generation (RAG) systems, natural language processing (NLP), and process automation—skills directly relevant to collaborative, industry-driven research environments like the Agentic Systems Lab.

## About Me

I am Teo Pazera, a Master's student in Data Science at Radboud University in Nijmegen, Netherlands. Alongside my studies, I work as a Data Science Intern at Zurich Insurance, currently based in the Netherlands (previously in the Bratislava Competence Center). My role involves implementing large language models (LLMs) for workplace automation, developing AI agents for task-specific operations, creating RAG-based chatbots, and other more industry standard data science projects like classifications of text data using NLP techniques.

With over 1.5 years of experience at Zurich and a Bachelor's in Data Science from Comenius University in Bratislava, I have worked on my skills in translating complex AI concepts into real-world solutions. My work emphasizes ethical AI deployment, performance optimization through heuristics, and collaboration between technical and business teams—qualities essential for succeeding in interdisciplinary research internships.

This repository highlights key projects from my Bachelor's studies and professional tenure. Due to non-disclosure agreements (NDAs), Zurich-related code and data are not shared; instead, I provide high-level overviews, methodologies, and outcomes. For my Bachelor's thesis, open-source code (excluding proprietary data) is available in the dedicated folder.

## Education

- **Master's in Data Science** (Ongoing), Radboud University, Nijmegen, Netherlands  

- **Bachelor's in Data Science** (2024), Comenius University, Bratislava, Slovakia  
  Thesis: "Enhancement of RAG Chatbot Performance Using Various Heuristics" (in collaboration with Zurich Insurance).  
  

## Professional Experience

- **Data Science Intern, Zurich Insurance** (2023–Present)  
  - Developed AI solutions for insurance processes, including RAG systems, agentic models, and NLP classifiers.  
  - Collaborated on cross-functional teams to automate reporting, claims processing, and actuarial tasks.  
  - Participated in Zurich's Agentic Hackathon, contributing to innovative AI agent prototypes.

## Projects

### Bachelor's Thesis: Enhancement of RAG Chatbot Performance Using Various Heuristics
- **Folder**: Bachelors_Thesis (includes code for heuristics implementation; proprietary data and knowledge bases excluded).
- **Description**: Collaborated with Zurich Insurance to optimize RAG chatbots for automating ESG (Environmental, Social, Governance) reporting. Explored a range of optimization strategies, including retrieval enhancements, data preprocessing methods (like Graph RAG), and advanced prompting.
- **Final Model**: A final model was selected that prioritized **simplicity and efficiency**, balancing performance with practical resource constraints. More complex methods like Graph RAG were evaluated but not adopted, as their marginal gains did not justify the significant increase in computational cost. The optimized workflow employs a multi-step process involving hypothetical answer generation, embedding-based retrieval, and an LLM-powered reranker to select the most relevant context before generating the final answer.
- **Technologies**: Python, LangChain for RAG pipelines, LLMs, advanced retrieval techniques, knowledge graphs, and statistical analysis methods.
- **Outcomes**: Achieved **statistically significant improvements** in performance metrics, confirmed through statistical analysis. The resulting solution was **successfully implemented by the sustainability team at Zurich Insurance**, validating its practical effectiveness and streamlining their reporting workflow.

### Professional Projects at Zurich Insurance
Due to NDAs, these are described at a high level without code or data. They build on my RAG expertise, evolving toward agentic AI models for insurance automation.

1. **Actuarial Domain RAG Chatbot** (Introductory Project)  
   - **Description**: Built a chatbot to query an IFRS 17 guidebook, parsing documents for vector-based retrieval and evaluating pipeline performance. Developed a basic UI for user interaction and feedback loop.
   - **Technologies**: Python, vector databases(Chroma), LLMs through API, parsers, langchain framework

2. **File Parsing and Metadata-Filtered RAG Systems** (Multiple follow up projects)  
   - **Description**: Adapted RAG frameworks for repetitive file parsing tasks, incorporating metadata filters from user queries and testing heuristics for performance gains(which were later used in bachelors thesis project).  
   - **Technologies**: Python, vector databases(Chroma), LLMs through API, parsers, langchain framework
   
   
3. **NLP-Based Claims Classification**  
   - **Description**: Classified insurance claims into categories using NLP, opting for lightweight models over LLMs for efficiency and precision.  

4. **Agentic Model for Hierarchical Insurance Reporting**  
   - **Description**: Implementation of an agentic system using LangGraph to aggregate country-level presentations into regional/group-level reports. Agents used tools for filtering irrelevant/relevant data, summarizing hierarchies, and incorporating guidelines for context-aware reasoning.  
   - **Technologies**: LangGraph, LLMs, custom tools for data filtering and knowledge lookup.  
  

5. **Agentic Hackathon: Reserving Data Presentation Drafter**  
   - **Description**: Team project building a semi-autonomous agent to generate presentation drafts from hierarchical reserving data, starting with low-level commentary and scaling up. 
   - **Technologies**: LangGraph, numerical data processing tools. 

### Academic Projects 
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



## Contact
- LinkedIn: [linkedin.com/in/teo-pazera](https://www.linkedin.com/in/teo-pazera-7520b1380/)
- Email: pazerateo@gmail.com  
