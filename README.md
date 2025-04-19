# chatbot_aws
## Chatbot using AWS
### Description

This chatbot answers user queries based on the files they upload. The supported file types are:

1. PDFs
2. CSV/Excel files
3.Images

The chatbot processes the uploaded files, extracts the necessary information, and uses advanced embeddings and AI models to provide accurate answers.

### Features
Flask Interface: Users can upload files through an intuitive web interface.

File Visualization: Uploaded files are displayed on the frontend for reference.

Document Querying: Users can ask questions about the uploaded files, and the chatbot provides meaningful answers based on the context.

### Prerequisites

1. Ensure you have Python 3.12 installed.
2. Install the required libraries by running the following command:
### pip install -r requirements.txt

### Usage
### To run the chatbot:
1. Navigate to the project directory.
2. Use the following command to start the Streamlit app:
### flask run

Once started, you can upload files and begin querying them.

### Files and Their Purpose

### 1. ECR
#### a. lambda_function.py
This file includes functions to identify the type of uploaded file (PDF, CSV/Excel, or Image) and to provide responses to user queries through the chatbot interface.

#### b. Docker file
Contains all certificates and files necessary for the backend, which are created and pushed to AWS ECR.

### 2. Flask
#### a. static folder
Contains images used on the website and all files required for styling the webpage.

#### b. templates
This folder contains all HTML pages used for the web application.

#### c. requirements.txt
Lists the libraries to be installed for running the chatbot.

#### d. upload_file.py
Facilitates uploading files to the AWS S3 bucket.

#### e. app.py
Serves as the Flask application, managing navigation between HTML pages and handling backend operations such as file uploads and storing user information in DynamoDB.

