from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import requests
from flask_cors import CORS
from boto3.dynamodb.conditions import Key, Attr
import boto3
from datetime import timedelta
from upload_file import lambda_handler
import re

app = Flask(__name__)
app.secret_key = '78'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1) # Session lasts for 1 day
CORS(app)

# Replace with your actual endpoint
CHATBOT_API_URL = "https://ss0yltf9sf.execute-api.eu-west-1.amazonaws.com/dev/ppp-bot"



s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
TABLE_NAME = 'chatHistory'
BUCKET_NAME = 'pp-chatbot'
# Replace with your API Gateway URL
API_GATEWAY_URL = 'https://rgmkal1aug.execute-api.eu-west-1.amazonaws.com/dev/pp-upload'



# Helper function for password validation
def is_valid_password(password):
    # Ensure password has at least one uppercase letter, one digit, and one special character
    return (len(password) >= 8 and
            re.search(r"[A-Z]", password) and
            re.search(r"[0-9]", password) and
            re.search(r"[!@#$%^&*(),.?\":{}|<>]", password))


@app.route('/')
def  main_page():
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('home'))
    
    else: 
        return redirect(url_for('signup'))

@app.route('/signup', methods=['POST', 'GET'])
def signup():
    
    '''
        This function handles user signup by validating input, checking for existing users, and registering new users.
    '''
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password1 = request.form['password1']
        password2 = request.form['password2']
        
        print("*********username*********",username)
        table = dynamodb.Table('pp-users')
        
        # Check if username or email already exists
        response = table.scan(
            FilterExpression=Attr('username').eq(username) | Attr('email').eq(email)
        )
        
        if response['Items']:
            # Username or email is already taken
            error_msg = "Username or Email already exists. Please choose a different one."
            return render_template('index.html', error_msg=error_msg)

        # Validate password
        if not is_valid_password(password1):
            error_msg = "Password must be at least 8 characters long, include one uppercase letter, one number, and one special character."
            return render_template('index.html', error_msg=error_msg)
        
        if password1 != password2:
            error_msg = "Password does not match."
            return render_template('index.html', error_msg=error_msg)
        

        # Register user if validations pass
        table.put_item(
            Item={
                'username': username,
                'email': email,
                'password': password1
            }
        )
        success_msg = "Registration Complete. Please Login to your account!"
        return render_template('index.html', success_msg=success_msg)
    
    return render_template('index.html')


@app.route('/check', methods=['POST', 'GET'])
def check():
    '''
        This function checks user credentials during login and manages session state.
    '''
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        table = dynamodb.Table('pp-users')
        response = table.query(
            KeyConditionExpression=Key('username').eq(username)
        )
        items = response['Items']
        
        if items and password == items[0]['password']:
            session.permanent = True
            session['logged_in'] = True
            session['name'] = items[0]['username']
            print(session.permanent)
             # Make the session permanent, i.e., use the lifetime set above
            return redirect(url_for('home'))
    
        print(session)
        return render_template("index.html", msg="Invalid credentials")
    
    return render_template("index.html")


@app.route('/home',methods=['GET'])
def home():
    
    '''
        This function renders the home page if the user is logged in, otherwise redirects to the signup page.
    '''
    
    if 'logged_in' in session:
        return render_template('home.html', name=session['name'])
    
    return redirect(url_for('signup'))

@app.route('/logout', methods=['POST'])
def logout():
    
    '''
        This function handles user logout, clearing the session and deleting user-specific files from S3 if no required files are left.
    '''

    folder_name=session['name']
    
    folder_prefix = f'{folder_name}/'
    all_files = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_prefix)
    
    if 'Contents' in all_files:
        has_required_files = False
        for obj in all_files['Contents']:
            # Check if any remaining files have a .pdf, .csv, or .excel extension
            if obj['Key'].endswith(('.pdf', '.csv', '.xlsx','xls')):
                has_required_files = True
                break

        # If no required files are left, delete all items in foldername/
        if not has_required_files:
            delete_all_keys = [{'Key': obj['Key']} for obj in all_files['Contents']]
            s3_client.delete_objects(
                Bucket=BUCKET_NAME,
                Delete={
                    'Objects': delete_all_keys
                }
            )
                
    table = dynamodb.Table(TABLE_NAME)
    table.delete_item(
            Key={
                'sessionId':folder_name
            }
        )
    session.clear()
    return redirect(url_for('signup'))

@app.route('/upload_multiple', methods=['POST'])
def upload_multiple():
    
    '''
        This function handles the upload of multiple files, sending them to an API Gateway and uploading them to S3 using presigned URLs.
    '''
    
    folder=session['name']
    # Get files from request
    files = request.files.getlist('files')
    
    # Get folder name from session or use a default value
    folder_name = session.get("folder_name", folder)  # Replace with your logic to get the folder name

    # Prepare the file information for Lambda
    file_data = [{'file_name': file.filename, 'file_type': file.content_type} for file in files]

    # print(file_data['file_name'])
    # Send files and folder name to Lambda
    response = requests.post(
        API_GATEWAY_URL,
        json={'files': file_data, 'folder_name': folder_name}  # Include folder name
    )

    if response.status_code != 200:
        return jsonify({'error': 'Failed to get presigned URLs'}), response.status_code

    presigned_urls = response.json().get('urls', {})

    # Upload each file to its presigned URL
    upload_results = []
    total_files = len(files)
    data = []
    for index, file in enumerate(files):
        file_name = file.filename
        data.append(f"https://pp-chatbot.s3.eu-west-1.amazonaws.com/{folder_name}/{file_name}")
        presigned_url = presigned_urls.get(file_name)
        if presigned_url:
            upload_response = requests.put(presigned_url, data=file.read(), headers={'Content-Type': file.content_type})
            if upload_response.status_code == 200:
                upload_results.append({'file_name': file_name, 'folder_name': folder_name, 'status': 'uploaded', 'file_type': file.content_type})
                print(f"File {index + 1}/{total_files}: '{file_name}' uploaded successfully.")
            else:
                upload_results.append({'file_name': file_name, 'folder_name': folder_name, 'status': 'failed'})
                print(f"File {index + 1}/{total_files}: '{file_name}' failed to upload. Status code: {upload_response.status_code}")
        else:
            upload_results.append({'file_name': file_name, 'folder_name': folder_name, 'status': 'no_presigned_url'})
            print(f"File {index + 1}/{total_files}: No presigned URL for '{file_name}'.")

        # Print progress after each file upload

    
        progress = (index + 1) / total_files * 100
        print(f"Upload Progress: {progress:.2f}%")

    
    if file_name.lower().endswith('.pdf'):
        lambda_handler(data,BUCKET_NAME,folder_name)

    return jsonify({'results': upload_results, 'presigned_urls': presigned_urls})




@app.route('/delete_file', methods=['POST','GET'])
def delete_file():
    
    '''
        This function handles the deletion of a specified file and its associated data from S3 and DynamoDB.
    '''
    
    data = request.json
    file_name = data.get('file_name')
    folder_name = data.get('folder_name')
    
    file = file_name.split('.')[0]
    print("file_name:---",file_name)
    # Construct the S3 key for deletion
    s3_key = f'{folder_name}/{file_name}'
    text_key = f'{folder_name}/vectors/{file}.txt'
    vectors_key = f'{folder_name}/vectors/faiss_index_{file}'
    folder_prefix = f'{folder_name}/' # e.g., foldername/
    
    
    try:
        # Delete the file vector and text file from S3
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        s3_client.delete_object(Bucket=BUCKET_NAME  , Key=text_key)
        
        s3 = boto3.resource('s3')
        bucket = s3.Bucket('pp-chatbot')
        bucket.objects.filter(Prefix=vectors_key).delete()
        
        # Check if foldername/ contains any remaining .pdf, .csv, or .excel files
        remaining_files = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_prefix)

        if 'Contents' in remaining_files:
            has_required_files = False
            for obj in remaining_files['Contents']:
                # Check if any remaining files have a .pdf, .csv, or .excel extension
                if obj['Key'].endswith(('.pdf', '.csv', '.xlsx','xls')):
                    has_required_files = True
                    break

            # If no required files are left, delete all items in foldername/
            if not has_required_files:
                delete_all_keys = [{'Key': obj['Key']} for obj in remaining_files['Contents']]
                s3_client.delete_objects(
                    Bucket=BUCKET_NAME,
                    Delete={
                        'Objects': delete_all_keys
                    }
                )
                
        table = dynamodb.Table(TABLE_NAME)
        table.delete_item(
            Key={
                'sessionId':folder_name
            }
        )
        return jsonify({'status': 'success', 'message': f'File {file_name} deleted successfully.'})
    except Exception as e:
        print(f"Error deleting file: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to delete {file_name}. Error: {str(e)}'}), 500


@app.route('/get', methods=['POST'])
def chatbot():
    
    '''
         This function handles user queries by sending them to a chatbot API and returning the response.
    '''
    
    user=session['name']
    response_data = None
    user_query = request.form.get('query')

    if user_query:
        payload = {
            "bucket_name": "pp-chatbot",
            "username": user,
            "query": user_query
        }

        # Send the request to the API
        try:
            response = requests.post(CHATBOT_API_URL, json=payload)
            response_data = response.json()
        except Exception as e:
            response_data = f"Error: {str(e)}"

    print(response_data)
    return jsonify({'response': response_data})

if __name__ == '__main__':
    app.run(debug=True)
