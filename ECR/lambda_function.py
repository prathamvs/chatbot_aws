
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import openai
import boto3
import json
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import tempfile
from openai.error import RateLimitError # Import the specific error
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.chains import ConversationChain
from langchain_experimental.agents import create_csv_agent
import pandas as pd
from botocore.exceptions import ClientError
import time
import base64
from io import BytesIO
from PIL import Image

dynamodb = boto3.client('dynamodb')

MAX_CONVERSATIONS = 6

custom_prompt_template = """
       Use the following pieces of information to answer the user's question.
        
        1. If the question is asked based on what is the title of the PDF/what is the PDF about/name of the PDF/what is the PDF mainly about then please only give the description of what PDF contains and what information does the PDF actually provides don't provide page number
        

        2. for answering every question:
        2.1 Analyze Both Text and Tables, When presented with a question, the chatbot will examine both the textual context and any relevant tables.
        Page 68 
        *****
        69
        Only qualified electrical maintenance personnel should install, operate, 
        service or maintain this equipment. 

            
        ***********
        Page 68 
        So here the page number is 68 Please follow this
        2.4 If there is no answer present in the catlogue then don't give answers just say don't have information about it 
        For example 
        how can we change N31 to N33:- There is no answer mentioned so by replacing KTA5000ZC31 with KTA5000ZC33 is incorrect answer so since the question's answer is not present then don't give wrong answer and the page number is provided before and after asteris likewise

        2.5 The answer might be in different section so consider all the sections analyze the sections and then provide answer along with respective page number such as phase conductor answer is on Page 150 not 180 so handle the page numbers carefully which can be above or below the context/table
        2.6 Please don't give any incomplete answer complete the sentences but dont't provide incomplete sentences please use some logic of yours here
        2.7 If Page number is not found then don't give page number 

        3. for answering questions related to current ratings, please carefully examine the table and provide all relevant information, including any multiple values or ratings that may be listed.
        For example 
        3.1. If tables have information likewise | 1000 | 1732 | 866 | then the value for 2500 is 4000 & 5000 & please give right info. as for 2500A the values are in front of it not above or below please follow the instruction carefully
        for example:-
        | Busbar | 1 Source | 2 Sources |
        |--------|----------|-----------|
        | 800    | 1386     | 693       |
        | 1000   | 1732     | 866       |
        
        
        Note:-  This is just example context will start after this instruction
        
        3.2. The structure of table might not be seen for example it may look like  
            e.g.
            Utilization
            'Medical Party
            P1 installation 0.45 0.23
            P2 installation 0.72 0.82
            ' but which is actually a table and has answers likewise

            '
            |P1 Installation|Medical|0.45|
            |P1 Installation|Party|0.23|
            |P2 Installation|Medical|0.72|
            |P2 Installation|Party|0.82|

            which means P1 installation for medical is 0.45 and for party is 0.23
        
        4. for analyzing correct page numbers:
        4.1 Check the page numbers correctly because some page numbers are mentioned after paragraph
            for example:-
            Page 68 
            *****
            Only qualified electrical maintenance personnel should install, operate, 
            service or maintain this equipment. 

              
            ***********
            Page 68 
            Where in this context the page number is Page 68 which is provided after the context
        
        5. Type of Question
        5.1 For some of the question please use some keyword searching for example For which type of conductor the section is provided in catalogue? because it's answer is 4 + PE 
        5.2 Some questions might be incomplete so please autocomplete those questions by using logic and then give answer
        
        6. So, Atlast while giving answer do mention page number and Name of the PDf now there will be multiple context so you can use seperate by finding "Answer found in this PDF" text for example

        **Answer found in this PDF**:
        Title: **Canalis KT - Busbar Trunking System - Installation Manual - 11/2018**
        Name:**PDF2**
        Supports and Run Components:- Page 22
        22 QGH3492101-01 11/2018General Rules for Installing Supports:- Page 22
        Safety Instructions:- Page 22
    
        **Answer found in this PDF**:
        Title: **CANALIS KTA 800-5000 DEBU021EN 2023-V3**
        Name:**CANALIS KTA 800-5000 DEBU021EN 2023-V3**
        -1/2:- Page 34
        **L1  L2  L3N  PEDD205853 DB430444 DD205854**:- Page 34
        DD202434DD202435-m DD202436-mDD205855:- Page 34
        32:- Page 34
        **3L + N + PER3L + PE**:- Page 34
        **3L + N + PE**:- Page 34

        So, above you can see after every line of **Answer found in this PDF**: the Name and Title of the uploaded multiple PDFs does get change so please update the Name & Title as per the question asked
        
        But while answering only Name of the PDF is to be mentioned not Title of the PDF

        Helpful answer:
"""


s3 = boto3.client('s3')


# Function to classify files in an S3 bucket
def classify_files_in_s3(bucket_name, folder_prefix=''):
    """Classify files in an S3 bucket based on their file type."""
    csv_files, excel_files, pdf_files,images_files = [], [], [],[]
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)

    if 'Contents' in response:
        for obj in response['Contents']:
            file_key = obj['Key']
            if file_key.endswith('.csv'):
                csv_files.append(file_key)
            elif file_key.endswith(('.xlsx', '.xls')):
                excel_files.append(file_key)
            elif file_key.endswith('.pdf'):
                pdf_files.append(file_key)
            
            elif file_key.endswith(('.jpg','png','jpeg')):
                images_files.append(file_key)

    return {
        'CSV Files': csv_files,
        'Excel Files': excel_files,
        'PDF Files': pdf_files,
        'Image Files':images_files
    }

## Process csv excel file
def process_csv_excel(csv_dir,uploaded_file):
    """Process CSV/Excel files."""
    if uploaded_file.endswith('.csv'):
        # File is already a CSV
        return uploaded_file
    else:
        # Convert Excel to CSV
        excel_data = pd.read_excel(uploaded_file)
        csv_path = f"{csv_dir}/converted_file.csv"
        excel_data.to_csv(csv_path, index=False)
        return csv_path
    
### Conversational history
def get_conversation_history(session_id):
    """Retrieve the conversation history for a specific session from DynamoDB."""
    try:
        response = dynamodb.get_item(
            TableName='chatHistory',
            Key={'sessionId': {'S': session_id}}
        )
        if 'Item' in response:
            # Decode the conversation history (it's stored as JSON)
            history = json.loads(response['Item']['messages']['S'])
            return [HumanMessage(content=msg['content']) if msg['role'] == 'user' else SystemMessage(content=msg['content']) for msg in history]
        else:
            # If there's no previous history, return an empty list
            return []
    except ClientError as e:
        print(f"Failed to retrieve conversation history: {e}")
        return []

def save_conversation_history(session_id, messages):
    """Save the updated conversation history to DynamoDB."""
    try:
        # Convert messages to a simple format that can be saved (as JSON)
        message_data = [{'role': 'user' if isinstance(msg, HumanMessage) else 'assistant', 'content': msg.content} for msg in messages]

        dynamodb.put_item(
            TableName='chatHistory',
            Item={
                'sessionId': {'S': session_id},
                'messages': {'S': json.dumps(message_data)}
            }
        )
    except ClientError as e:
        print(f"Failed to save conversation history: {e}")
        

def retrieve_documents_from_s3(bucket_name, file_key):
    """ Fetch the JSON data from S3 and convert it to Document objects. """
    # try:
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    serialized_data = response['Body'].read().decode('utf-8')
    documents_json = json.loads(serialized_data)

    documents_list = []
    for documents in documents_json:
        file_documents = []
        for doc in documents:
            file_documents.append(Document(metadata=doc['metadata'], page_content=doc['page_content']))
        documents_list.append(file_documents)
    
    return documents_list

def download_folder_from_s3(bucket_name, s3_folder, temp_dir):
    """ Download a folder from an S3 bucket to a temporary directory. """
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)

    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            if not key.endswith('/'): # Skip if it's a folder
                # Create a temporary file path using tempfile
                rel_path = os.path.relpath(key, s3_folder)
                temp_file_path = os.path.join(temp_dir, rel_path)
                
                # Ensure the directory for the file exists using tempfile
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                
                # Download the file from S3 to the temporary path
                s3.download_file(bucket_name, key, temp_file_path)
                print(f"Downloaded {key} to {temp_file_path}")
    else:
        raise Exception(f"No files found in S3 folder: {s3_folder}")



def load_vector_stores(vectors_directory, embedding_model="text-embedding-ada-002"):
    """ Load FAISS vector stores from the directory """
    vectorstores = []
    embeddings = OpenAIEmbeddings(engine=embedding_model)

    for file_name in os.listdir(vectors_directory):
        if file_name.startswith("faiss_index_"):
            index_path = os.path.join(vectors_directory, file_name)
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            vectorstores.append(vectorstore)

    return vectorstores
    
    
def retrieve_relevant_content(bucket_name, username, query):
    
    """ Retrieves relevant content using ensemble retrievers """
        
    with tempfile.TemporaryDirectory() as temp_dir:
        vectors_dir = os.path.join(temp_dir, 'vectors')
        os.makedirs(vectors_dir, exist_ok=True) # Ensure temp_dir is created
    

    # Download the vector folder from S3 to temp_dir
    download_folder_from_s3(bucket_name, f"{username}/vectors", vectors_dir)
    
    # Retrieve all documents from S3
    all_documents = retrieve_documents_from_s3(bucket_name, f'{username}/all_documents.json')
    
    # Load the FAISS vector stores
    vectorstores = load_vector_stores(vectors_dir)
    retrievers = []
    
    # Combine retrievers for each namespace
    for i, vectorstore in enumerate(vectorstores):
        retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5})
        keyword_retriever = BM25Retriever.from_documents(all_documents[i])
        keyword_retriever.k = 5

        ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb, keyword_retriever], weights=[0.4, 0.6])
        retrievers.append(ensemble_retriever)

    # Fetch relevant content based on the query
    
    time.sleep(20)
    all_relevant_content = ""
    for retriever in retrievers:
        print(retriever)
        docs_rel = retriever.get_relevant_documents(query)

        if docs_rel:
            all_relevant_content += "**Answer found in this PDF**: \n"
            all_relevant_content += "\n".join([doc.page_content for doc in docs_rel])
        else:
            all_relevant_content += "No relevant information found.\n"

    return all_relevant_content
    # return res


def conversational_chat(all_relevant_content, query, user_id):
    
    '''
        This function facilitates a conversational chat using relevant content, a query, 
        and specified input and output folders.
    '''
    
    buffer_memory_store = {}

    session_id = user_id

    # Fetch previous conversation history from DynamoDB
    previous_messages = get_conversation_history(session_id)

    # Add the current input to the conversation history
    previous_messages.append(HumanMessage(content=query))

    llm = ChatOpenAI(engine="gpt-4o") # Define LLM model

    # Check if the user has memory stored
    if user_id not in buffer_memory_store:
        buffer_memory_store[user_id] = ConversationBufferWindowMemory(k=3, return_messages=True)

    # Define conversation
    system_msg_template = SystemMessagePromptTemplate.from_template(template=custom_prompt_template)
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
    
    conversation = ConversationChain(memory=buffer_memory_store[user_id], prompt=prompt_template, llm=llm, verbose=True)

    
    # Chat
    response = conversation.predict(input=f"Context:\n {all_relevant_content} \n\n Query:\n{query} with context & only name and Page number of the PDF if answer found in multiple PDF then only mention context & Name of both PDF don't mention Title of the PDF else asked, if title of the PDF is asked then only mention title and nothing else Previous message: {previous_messages}")

    # Store the assistant's reply in the conversation history
    previous_messages.append(SystemMessage(content=response))

    if len(previous_messages) > MAX_CONVERSATIONS:
        previous_messages = previous_messages[-MAX_CONVERSATIONS:]

    # Save the updated conversation back to DynamoDB
    save_conversation_history(session_id, previous_messages)
    
    return response


def csv_excel_agent(bucket_name, username, query):
    '''
    This function processes CSV/Excel files from an S3 bucket, creates an agent to handle queries, and returns the response.
    '''
    
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_dir = os.path.join(temp_dir, 'csvfile')
        os.makedirs(csv_dir, exist_ok=True) # Ensure temp_dir is created
        
        download_folder_from_s3(bucket_name, username, csv_dir)
    
        llm = ChatOpenAI(engine="gpt-4o")
        
        # Create the CSV agent
        print(csv_dir)
        for file_name in os.listdir(csv_dir):
            print(os.listdir(csv_dir))
            file1 = process_csv_excel(csv_dir, f"{csv_dir}/{file_name}")
            file = os.path.join(csv_dir, file1)
            agent = create_csv_agent(llm, file, verbose=True, allow_dangerous_code=True)

        # Attempt to invoke the agent with retry logic
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = agent.invoke(query)
                return response['output']
            except RateLimitError as e:
                print(f"Rate limit exceeded: {e}. Waiting for 1 minute before retrying...")
                time.sleep(60) # Wait for 1 minute before retrying
            except Exception as e:
                print(f"An error occurred: {e}")
                break # Exit on other exceptions
        else:
            print("Failed to get response after maximum retries.")
            return None
        
        
        
### Answering question based on Images

def encode_image(image_path, max_image=512):
    
    '''
        This function encodes an image to a base64 string, resizing it if necessary to fit within a maximum dimension.
    '''
    
    with Image.open(image_path) as img:
        width, height = img.size
        max_dim = max(width, height)
        if max_dim > max_image:
            scale_factor = max_image / max_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height))

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    

def chat_response(bucket_name,username,user_prompt):
    
    '''
        This function generates a chat response by sending an image and user prompt to an AI model.
    '''
    
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, 'imgfile')
        os.makedirs(image_path, exist_ok=True) # Ensure temp_dir is created
    
    download_folder_from_s3(bucket_name, username, image_path)
    
    system_prompt = "You are an expert at analyzing images."
    
    for image in os.listdir(image_path):
        encoded_image = encode_image(f"{image_path}/{image}")

    try:
        apiresponse = openai.ChatCompletion.create(
            engine="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                        },
                    ],
                },
            ],
            max_tokens=500,
        )

        # Print the response
        response_content = apiresponse.choices[0].message.content
        
        
        # print("Response:", response_content)

    except Exception as e:
        print("An error occurred:", e)
        
    return response_content

    

def lambda_handler(event, context):
    
    '''
        The lambda_handler function is designed to connect with OpenAI API endpoints and 
        handle different types of files (PDF, CSV/Excel, and images) to return a response to the user.
    '''

    openai.api_type = "azure"
    openai.api_base = "https://apim-guardian-prv-fc.aihub.se.com"
    openai.api_version = "2024-06-01"
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    os.environ["OPENAI_API_TYPE"] = openai.api_type
    os.environ["OPENAI_API_BASE"] = openai.api_base
    os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')
    os.environ["OPENAI_API_VERSION"] = openai.api_version
    os.environ["REQUESTS_CA_BUNDLE"]  = '/var/task/pki-root-cert.crt'
    

        
    data = json.loads(event['body'])
    
    bucket_name = data.get('bucket_name')
    username = data.get('username')
    query = data.get('query')
    session_id = username
    
    # Fetch relevant content
    classified_files = classify_files_in_s3(bucket_name, username)

    if classified_files['PDF Files']:
        all_relevant_content = retrieve_relevant_content(bucket_name, username, query)
        res = conversational_chat(all_relevant_content, query, username)

    elif classified_files['CSV Files'] or classified_files['Excel Files']:
        res = csv_excel_agent(bucket_name,username,query)
        
    elif classified_files['Image Files']:
        res = chat_response(bucket_name,username,query)

    
    # Return the result
    return {
        'statusCode': 200,
        'body': json.dumps(res)
    }
