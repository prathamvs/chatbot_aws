
import boto3
import os
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import openai
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
from PyPDF2 import PdfReader
from langchain.schema import Document
import os
import requests
import tempfile

# Initialize the S3 client
s3 = boto3.client('s3')

### Parsing PDF
# Function to format text with page numbers and proper headings
def format_text_with_page_numbers(lines, page_number):
    
    """
        This function formats a list of text lines by adding page numbers and proper headings.
    """
    
    formatted_text = ""
    section_heading = ""
    

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue

        if stripped_line.isupper() or stripped_line.endswith(":"):
            section_heading = stripped_line
            formatted_text += f"**{section_heading}**:- Page {page_number}\n"
        else:
            formatted_text += f"{stripped_line}:- Page {page_number}\n"
    
    return formatted_text


# Function to extract and format text from PDFs
def create_formatted_text_from_pdfs(pdf_path, output_folder):
    
    '''
        This function extracts text from a PDF file, formats it with page numbers and headings, 
        and saves the formatted text to a specified output folder.
    '''
    
    base_name = os.path.basename(pdf_path).replace('.pdf', '')
    output_file_path = os.path.join(output_folder, f"{base_name}.txt")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    documents = []
    reader = PdfReader(pdf_path)
            # Attempt to extract the title from PDF metadata
    title = reader.metadata.get('/Title', None)
    name = base_name

    # If no title, use the filename as the title
    if not title:
        title = base_name


    
    with open(output_file_path, "w", encoding="utf-8") as text_file:
        

            # Store the extracted content
        extracted_text = f"Title of the PDF: **{title}**\n Name of te PDF:**{name}**"
        text_file.write(extracted_text)

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                lines = text.splitlines()
                formatted_text = format_text_with_page_numbers(lines, page_number + 1)

                chunked_documents = formatted_text.split('\n\n')

                for chunk in chunked_documents:
                    # Collect documents (chunks) for embedding
                    document_content = f"Title of the PDF: **{title}**\nName of the PDF: **{name}**\n\n{chunk}"
                    documents.append(Document(page_content=document_content,metadata={"page_number":page_number})) # Split into chunks by double newlines

                # Write formatted content to the text file
                text_file.write(f"******************\nPage {page_number + 1}\n\n")
                text_file.write(formatted_text + "\n")
                text_file.write(f"******************\nPage {page_number + 1}\n\n")

    return documents




def store_documents_in_temp(documents_list, temp_path,final_path):
    """
    Store the list of documents in the /tmp directory as a JSON file.

    Args:
        documents_list (list): List of lists of Document objects.
        temp_path (str): Path in the /tmp directory to save the file.
    """
    
    documents_json = [
        [
            {
                "metadata": doc.metadata,
                "page_content": doc.page_content
            }
            for doc in documents
        ]
        for documents in documents_list
    ]
    
    print(temp_path)
    # Serialize to JSON and save in /tmp directory

    with open(temp_path, 'w') as temp_file:
        json.dump(documents_json, temp_file)

    final_dir = os.path.dirname(final_path)
    os.makedirs(final_dir,exist_ok=True)

    print(f"Documents saved to temporary file: {final_dir}")


### Save vector store
# Save vector stores for each PDF separately
def save_vector_stores(pdf_files, vectors_directory,output_folder, embedding_model="text-embedding-ada-002"):
    
    '''
        This function processes a list of PDF files to create and save vector stores using embeddings.
    '''

    embeddings = OpenAIEmbeddings(engine=embedding_model)
    
    if not os.path.exists(vectors_directory):
        os.makedirs(vectors_directory)

    all_documents = []

    for pdf_path in pdf_files:
        # Create documents from each PDF file
        documents = create_formatted_text_from_pdfs(pdf_path,output_folder)
        all_documents.append(documents)

        retries = 3
        for attempt in range(retries):
            try:
                # Create FAISS vector store from documents
                vectorstore = FAISS.from_documents(documents, embeddings)
                # Save vector store with a unique name
                base_name = os.path.basename(pdf_path).replace('.pdf', '')
                vectorstore.save_local(os.path.join(vectors_directory, f"faiss_index_{base_name}"))

                break

            except openai.error.RateLimitError as e:
                if attempt < retries -1:
                    # st.warning(f"Rate limit exceeded: {e}. Waiting for 1 minute before retrying....")
                    time.sleep(60)

                else:
                    raise e

    
    return all_documents


def upload_folder_to_s3(local_folder, bucket_name, s3_folder_name):
    """
    Uploads all files from a local folder (e.g., /tmp) to an S3 bucket.

    Args:
        local_folder (str): Path to the local folder.
        bucket_name (str): Name of the S3 bucket.
        s3_folder_name (str): S3 folder path to upload files under.
    """
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            
            # Define S3 file path, maintaining folder structure
            relative_path = os.path.relpath(local_file_path, local_folder)
            s3_file_path = os.path.join(s3_folder_name, relative_path).replace("\\", "/")
            
            print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_file_path}")
            s3.upload_file(local_file_path, bucket_name, s3_file_path)
    print("Upload to S3 completed.")

def download_pdf_from_s3_url(pdf_url, download_path):
    """
    Downloads a PDF from an S3 URL to a specified path.
    
    Args:
        pdf_url (str): S3 HTTPS URL of the PDF.
        download_path (str): Local path to save the downloaded PDF.
    """
    response = requests.get(pdf_url)
    response.raise_for_status() # Raise an error for bad requests
    
    with open(download_path, 'wb') as file:
        file.write(response.content)

def lambda_handler(pdf_files, bucket_name, username="default-folder"):
    """
    Processes PDFs from S3 URLs, storing vectors and documents in /tmp,
    and then uploads them to S3.

    Args:
        pdf_files (list): List of S3 HTTPS URLs for PDF files to process.
        bucket_name (str): S3 bucket name for saving results.
        username (str): S3 folder name to store files under.
    """
    openai.api_type = "azure"
    openai.api_base = "https://apim-guardian-prv-fc.aihub.se.com"
    openai.api_version = "2024-06-01"
    # openai.api_key = os.environ.get('OPENAI_API_KEY')
    openai.api_key = "b74bf34f88b449f5b25764e363d4dd49"
    # openai.Model = "gpt-35-turbo"
    os.environ["REQUESTS_CA_BUNDLE"]  = r'C:\Users\SESA766670\Desktop\Deployment\aws_functions\REACT\pki-root-cert.crt'
    os.environ["OPENAI_API_KEY"] = openai.api_key
    
    with tempfile.TemporaryDirectory() as temp_dir:
        dir = os.path.join(temp_dir, 'tmp')
        os.makedirs(dir, exist_ok=True) # Ensure temp_dir is created
        
    # Define paths in the /tmp directory
    temp_vectors_dir = os.path.join(dir, username, "vectors")
    docs_dir = os.path.join(dir, username, "docs")
    temp_docs_file = os.path.join(dir, username, "all_documents.json")
    
    # Ensure /tmp directories exist
    os.makedirs(temp_vectors_dir, exist_ok=True)
    final_docs_file = os.path.join(docs_dir,"all_documents.json")
    os.makedirs(docs_dir,exist_ok=True)
    
    # List to collect all documents for storage
    all_documents = []

    for pdf_url in pdf_files:
        # Define the local path for the downloaded PDF in /tmp
        pdf_name = os.path.basename(pdf_url)
        local_pdf_path = os.path.join(dir, pdf_name)
        
        # Download PDF from the S3 URL to /tmp
        download_pdf_from_s3_url(pdf_url, local_pdf_path)
        
        # Process PDF to create vectors and documents
        documents = save_vector_stores([local_pdf_path], temp_vectors_dir, temp_vectors_dir)
        all_documents.extend(documents)
    
    # Store the consolidated documents to the temporary JSON file
    store_documents_in_temp(all_documents, temp_docs_file,final_docs_file)

    upload_folder_to_s3(os.path.join(dir, username), bucket_name, username)
