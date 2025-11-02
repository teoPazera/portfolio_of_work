import os 
from docling.document_converter import DocumentConverter,  PdfFormatOption, ConversionResult
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrEngine, TableFormerMode
from docling.datamodel.settings import settings
from langchain_core.documents.base import Document
from typing import Literal
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import math
from collections import defaultdict
from custom_splitter import split_markdown_sentences_tables, split_markdown_to_chunks, split_large_table
from utils.token_counter import count_tokens

class Parser:
    pdf_documents_directory: str
    chunk_lengths: int
    embedding_model: AzureOpenAIEmbeddings
    company_name: str

    def __init__(self, pdf_documents_directory, chunk_lengths: int) -> None:
        """Parser class to be used when we reading the pdf documents and converting them to
        markdown text chunks. The class is used to convert the pdf documents to text chunks

        Args:
            pdf_documents_directory (_type_): _directory to the pdf documents
            chunk_lengths (int): length of the text chunks to be created
        """
        self.pdf_documents_directory = pdf_documents_directory
        self.company_name = pdf_documents_directory.split("\\")[-1]
        self.chunk_lengths = chunk_lengths

        self.markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            language="markdown",
            chunk_size=chunk_lengths,
            chunk_overlap=chunk_lengths//10, # 10 percent of chunk lengths
        )

        self.html_splitter = RecursiveCharacterTextSplitter.from_language(
            language="html",
            chunk_size=chunk_lengths,
            chunk_overlap=chunk_lengths//10,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_lengths,
            chunk_overlap=chunk_lengths//10,
        )

        load_dotenv() # loading keys to be able to use embedding models
        self.embedding_model = AzureOpenAIEmbeddings(
            api_key=os.getenv('OPENAI_API_KEY'),
            azure_endpoint=os.getenv("AZURE_ENDPOINT_EMBEDDING"),
            model="text-embedding-3-large"
        )
        self.sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            
    def store_docs(self, conv_result: ConversionResult, database: str, file_name: str) -> None:
        """function to store the documents in the vector database based 

        Args:
            conv_result (ConversionResult): output of docling pdf parsing
            database (str): type of parsing as well as the chunking strategy
            file_name (str): name of the file to be used in metadata
        Returns:
            None
        """
        documents: list[Document] = []

        if database == "basic_vectors":
            documents.extend(self.export_by_pages_md( conv_result=conv_result, file_name=file_name, add_metadata=False))
        elif database == "basic_m_enriched_vectors":
            documents.extend(self.export_by_pages_md(conv_result=conv_result, file_name=file_name, add_metadata=True))

        elif database == "semantically_split_vectors":
            documents.extend(self.semantic_chunking(conv_result, file_name))
        
        
        for index in range(0, len(documents), 10):
            try:
                Chroma.from_documents(
                    documents=documents[index:index+10],
                    embedding=self.embedding_model,
                    persist_directory=database
                )
            except:
                print("failed to save")

    
    def semantic_chunking(self, conv_result: ConversionResult, file_name: str) -> list[Document]:
        """function to split the text into semantically similar chunks 

        Args:
            conv_result (ConversionResult): results of docling pdf parsing
            file_name (str): name of the file from which the text was parsed

        Returns:
            list[Document]: list of text chunks in Document wrapper 
        """
        text = conv_result.document.export_to_markdown()
        pure_sentences, tables = split_markdown_sentences_tables(text=text)

        lengths = [len(sen)for sen in pure_sentences]
        average = sum(lengths)/len(lengths)

        embeddings = self.sentence_embedding_model.encode(pure_sentences, normalize_embeddings=True)

        num_clusters = math.ceil(len(pure_sentences)*(average/self.chunk_lengths))
        if num_clusters > len(pure_sentences):
            num_clusters = len(pure_sentences)
        print(num_clusters)

        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(embeddings)
        cluster_assignments = kmeans.labels_

        clusters = defaultdict(list)
        for sentence, cluster_id in zip(pure_sentences, cluster_assignments):
            clusters[cluster_id].append(sentence)

        merged_texts = [' '.join(cluster) for cluster in clusters.values()]
        
        #print([len(text) for text in merged_texts])
        #print([len(text) for text in tables])

        documents: list[Document] = []
        for table in tables:
            if count_tokens(table) > 6000:
                table_chunks: list[str] = split_large_table(table_text=table, chunk_length=8000)
                print([count_tokens(chunk) for chunk in table_chunks])
                for table_chunk in table_chunks:
                    doc: Document = Document(page_content=table_chunk, metadata={"file_name": file_name, "company_name": self.company_name, "table": True})
                    documents.append(doc)
            if len(table) < 50 and documents:
                documents[-1].page_content += table
            else:
                doc: Document = Document(page_content=table, metadata={"file_name": file_name, "company_name": self.company_name, "table": True})
                documents.append(doc)

        for merged_text in merged_texts:
            tokens = count_tokens(merged_text)
            if tokens > 6000:
                splits = split_markdown_to_chunks(merged_text, chunk_length=4000)
                print([count_tokens(split) for split in splits])
                for split in splits:
                    doc: Document = Document(page_content=split, metadata={"file_name": file_name, "company_name": self.company_name, "table": False})
                    documents.append(doc)
            else:
                doc: Document = Document(page_content=merged_text, metadata={"file_name": file_name, "company_name": self.company_name, "table": False})
                documents.append(doc)

        return documents
    
    def export_by_pages_md(self, conv_result: ConversionResult, file_name: str, add_metadata: bool) -> list[Document]:
        """Function to export the text from the pdf document to markdown format

        Args:
            conv_result (ConversionResult): results of docling pdf parsing
            file_name (str): name of the file from which the text was parsed
            add_metadata (bool): whether to add metadata to the text chunks

        Returns:
            list[Document]: list of text chunks in Document wrapper
        """
        pages: list[str] = [conv_result.document.export_to_markdown(page_no=page_no) for page_no in conv_result.document.pages]
        doc: Document
        result: list[Document] = []
        page: str
        page_no: int 
        for page_no, page in enumerate(pages, start=1):
            splits: list[str] = split_markdown_to_chunks(text=page, chunk_length=self.chunk_lengths)
            split: str
            for split in splits:
                table: bool
                if split.startswith("##TABLE##"):
                    table = True
                    split = split.replace("##TABLE##", "")
                else:
                    table = False
                if add_metadata:
                    doc = Document(page_content=f'page number: {page_no} \n file name: {file_name} \n company name: {self.company_name} \n {split}',
                                    metadata={"page_number":page_no, "file_name": file_name, "company_name":self.company_name, "table": table})
                else:
                    doc = Document(page_content=split,
                                    metadata={"page_number":page_no, "file_name": file_name, "company_name":self.company_name, "table": table})
                
                result.append(doc)
        
        return result

    def export_by_pages_split(self, text_format: Literal["md", "html", "txt"], 
                              conv_result: ConversionResult, file_name: str, add_metadata: bool) -> list[Document]:
        """Function to export the text from the pdf document to markdown format

        Args:
            text_format (Literal[&quot;md&quot;, &quot;html&quot;, &quot;txt&quot;]): text format which we used when parsing
            conv_result (ConversionResult): result of docling pdf parsing
            file_name (str): name of the file to be saved in metadata
            add_metadata (bool): whether to add metadata to the text chunks

        Raises:
            ValueError: if no valid text format is provided

        Returns:
            list[Document]: list of text chunks in Document wrapper
        """
        pages: list[str]
        splitter: RecursiveCharacterTextSplitter
        if text_format == "md":
            pages = [conv_result.document.export_to_markdown(page_no=page_no) for page_no in conv_result.document.pages]
            splitter = self.markdown_splitter
        elif text_format == "html":
            pages = [conv_result.document.export_to_html(page_no=page_no) for page_no in conv_result.document.pages]
            splitter = self.html_splitter
        elif text_format == "txt":
            pages = [conv_result.document.export_to_text(page_no=page_no) for page_no in conv_result.document.pages]
            splitter = self.text_splitter
        else: 
            raise ValueError("Invalid text format for export in conversion process")
        
        doc: Document
        result: list[Document] = []
        page: str
        page_no: int 
        for page_no, page in enumerate(pages, start=1):
            if len(page) > self.chunk_lengths:
                splits: list[str] = splitter.split_text(text=page)
                split: str
                for split in splits:
                    if add_metadata:
                        doc = Document(page_content=f'page number: {page_no} \n file name: {file_name} \n company name: {self.company_name} \n {split}',
                                        metadata={"page_number":page_no, "file_name": file_name, "company_name":self.company_name})
                    else:
                        doc = Document(page_content=split,
                                        metadata={"page_number":page_no, "file_name": file_name, "company_name":self.company_name})
                    
                    result.append(doc)
            else:
                if add_metadata:
                    doc = Document(page_content=f'page number: {page_no} \n file name: {file_name} \n company name: {self.company_name} \n {page}',
                                        metadata={"page_number":page_no, "file_name": file_name, "company_name":self.company_name})
                else:
                    doc = Document(page_content=page,
                                        metadata={"page_number":page_no, "file_name": file_name, "company_name":self.company_name})
                result.append(doc)

        return result

    def create_vecdb(self) -> list[Document]:
        """function that goes through the entire process of creating documents for all of the vector databases

        Returns:
            list[Document]: list of documents in Document wrapper to be embedded
        """
        # , text_format: Literal["md", "html", "txt"]
        pipeline_options = PdfPipelineOptions(
            ocr_engine=OcrEngine.TESSERACT,  # Choose OCR engine: TESSERACT, EASY_OCR, etc.
            do_table_structure=True,         # Enable table structure recognition
            table_structure_options={
                'mode': TableFormerMode.ACCURATE  # Use accurate mode for better table parsing
            },
            enable_remote_services=False     # Disable remote services for local processing
        )
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.AUTO
        )
        converter: DocumentConverter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options)
            }
        ) 
        print(self.pdf_documents_directory)
        file_name: str
        for file_name in os.listdir(self.pdf_documents_directory):
            if file_name.lower().endswith(".pdf"):
                print(f"Currently handling file: {file_name} from directory: {self.company_name}")
                conv_result: ConversionResult = converter.convert(os.path.join(self.pdf_documents_directory, file_name))
                self.store_docs(conv_result=conv_result, database="basic_vectors", file_name=file_name)
                self.store_docs(conv_result=conv_result, database="basic_m_enriched_vectors", file_name=file_name)
                # self.store_docs(conv_result=conv_result, database="semantically_split_vectors", file_name=file_name)
                    
    
   

if __name__ == "__main__":
    os.environ['CURL_CA_BUNDLE'] = r'C:\Users\TEO.PAZERA\OneDrive - Zurich Insurance\Desktop\bakalarka_cody\Bakalarka_rag\huggingface.co.crt'
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    for folder in os.listdir("pdf_files"):
        if folder in ["BP", "Holcim", "Marriott", "RWE"]:
            parser = Parser(pdf_documents_directory=f"pdf_files\{folder}", chunk_lengths=1000)
            parser.create_vecdb()