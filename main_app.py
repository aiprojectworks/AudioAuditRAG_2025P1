#https://medium.com/@leopoldwieser1/how-to-build-a-speech-transcription-app-with-streamlit-and-whisper-and-then-dockerize-it-88418fd4a90

import csv
from functools import partial
from groq import Groq
from io import BytesIO
import json
from operator import is_not
import glob
import pandas as pd
import re
import tkinter as tk
from tkinter import filedialog
import requests
import streamlit as st
# import sys
#WhisperX import
# import whisperx
# import gc 
import torch
import os
import bcrypt
from dotenv import load_dotenv
from datetime import datetime
#LLM import
# from langchain.embeddings import LlamaCppEmbeddings
# from langchain_community.llms import LlamaCpp
# from langchain_core.prompts import PromptTemplate
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
# from openpyxl import Workbook
# from openpyxl.styles import PatternFill
# from streamlit.components.v1 import html
# import openai
from openai import OpenAI
from mutagen.mp3 import MP3, HeaderNotFoundError
from mutagen.id3 import ID3, ID3NoHeaderError
from pydub import AudioSegment
import zipfile
import io
from database import Session, User, seed_users
# from streamlit_cookies_manager import EncryptedCookieManager
# from streamlit_cookies_controller import CookieController
# from streamlit.web.server.websocket_headers import _get_websocket_headers 
# from urllib.parse import unquote
# import extra_streamlit_components as stx
# from streamlit.web.server.websocket_headers import _get_websocket_headers
import threading
import time
from sqlalchemy.exc import IntegrityError
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.runtime import get_instance
from typing import Tuple
# import streamlit_js_eval
# from streamlit_js_eval import streamlit_js_eval


#testestest
# from pydub.playback import play

# import assemblyai as aai
# import httpx
# import threading
# from deepgram import (
# DeepgramClient,
# PrerecordedOptions,
# FileSource,
# DeepgramClientOptions,
# LiveTranscriptionEvents,
# LiveOptions,
# )

class KillableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

# Load .env variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="IPPFA Trancribe & Audit",
                            page_icon=":books:")
groq_client = Groq(api_key=GROQ_API_KEY)
# deepgram = DeepgramClient(st.secrets["DEEPGRAM_API_KEY"])
client = OpenAI(api_key=OPENAI_API_KEY)

def is_admin(username: str) -> bool:
    """Check if user has admin role"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        return user is not None and user.role == "admin"
    except Exception as e:
        print(f"Database error checking admin status: {e}")
        return False
    finally:
        session.close()

def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate the password against specific requirements.
    Requirements:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not any(char.isupper() for char in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(char.islower() for char in password):
        return False, "Password must contain at least one lowercase letter."
    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one digit."
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character."
    return True, ""

def hash_password(password):
    # Salt the password before hashing it
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

def add_user(username: str, password: str, role: str = "user") -> Tuple[bool, str]:
    """Add a new user to the database"""
    try:
        # Validate password
        session = Session()
        hashed_password = hash_password(password)
        new_user = User(username=username, password=hashed_password, role=role)
        is_valid, message = validate_password(password)
        if not is_valid:
            return False, message
        session.add(new_user)
        session.commit()
        return True, "User added successfully"
    except IntegrityError:
        session.rollback()
        return False, "Username already exists"
    except Exception as e:
        session.rollback()
        return False, f"Error adding user: {e}"
    finally:
        session.close()

def delete_user(username: str) -> Tuple[bool, str]:
    """Delete a user from the database"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        if user:
            if user.role == "admin":
                # Count number of admin users
                admin_count = session.query(User).filter_by(role="admin").count()
                if admin_count <= 1:
                    return False, "Cannot delete the last admin user"
                if st.session_state.get("username") == username:
                    return False, "Cannot delete the currently logged in user"
            cleanup_on_logout(username, refresh=False)
            session.delete(user)
            session.commit()
            return True, "User deleted successfully"
        return False, "User not found"
    except Exception as e:
        session.rollback()
        return False, f"Error deleting user: {e}"
    finally:
        session.close()

def get_all_users() -> list:
    """Get all users from the database"""
    try:
        session = Session()
        users = session.query(User).all()
        return [{"username": user.username, "role": user.role} for user in users]
    except Exception as e:
        print(f"Error fetching users: {e}")
        return []
    finally:
        session.close()

def change_password(username: str, new_password: str) -> Tuple[bool, str]:
    """Change a user's password"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        if user:
            user.password = hash_password(new_password)
            session.commit()
            return True, "Password changed successfully"
        return False, "User not found"
    except Exception as e:
        session.rollback()
        return False, f"Error changing password: {e}"
    finally:
        session.close()

def change_role(username: str, new_role: str) -> Tuple[bool, str]:
    """Change a user's role"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        if user:
            # Prevent removing the last admin
            if user.role == "admin" and new_role != "admin":
                admin_count = session.query(User).filter_by(role="admin").count()
                if admin_count <= 1:
                    return False, "Cannot remove the last admin user"
            user.role = new_role
            session.commit()
            return True, "Role changed successfully"
        return False, "User not found"
    except Exception as e:
        session.rollback()
        return False, f"Error changing role: {e}"
    finally:
        session.close()

def admin_interface():
    """Render the admin interface in Streamlit"""
    st.title("Admin Panel")
    
    # Add New User Section
    st.subheader("Add New User")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_username = st.text_input("Username", key="new_username")
    with col2:
        new_password = st.text_input("Password", type="password", key="new_password")
    with col3:
        new_role = st.selectbox("Role", ["user", "admin"], key="new_role")
    
    if st.button("Add User"):
        if new_username and new_password:
            hashed_password = hash_password(new_password)
            success, message = add_user(new_username, hashed_password, new_role)
            if success:
                st.success(message)
                create_log_entry(f"Admin Action: Added new user - {new_username}")
            else:
                st.error(message)
                create_log_entry(f"Admin Action Failed: Add user - {new_username} - {message}")
        else:
            st.warning("Please fill in all fields")

    # Manage Existing Users Section
    st.subheader("Manage Users")
    users = get_all_users()
    
    if users:
        for user in users:
            with st.expander(f"User: {user['username']} ({user['role']})"):
                with st.container():
                    # Row 1: Change Password
                    st.subheader("Change Password")
                    new_pass = st.text_input("New Password", type="password", key=f"pass_{user['username']}")
                    if st.button("Change Password", key=f"btn_pass_{user['username']}"):
                        if new_pass:
                            success, message = change_password(user['username'], new_pass)
                            if success:
                                st.success(message)
                                create_log_entry(f"Admin Action: Changed password for user - {user['username']}")
                            else:
                                st.error(message)
                    
                    # Row 2: Change Role
                    st.subheader("Change Role")
                    new_role = st.selectbox("New Role", ["user", "admin"], 
                                            index=0 if user['role']=="user" else 1,
                                            key=f"role_{user['username']}")
                    if st.button("Change Role", key=f"btn_role_{user['username']}"):
                        success, message = change_role(user['username'], new_role)
                        if success:
                            st.success(message)
                            create_log_entry(f"Admin Action: Changed role for user - {user['username']} to {new_role}")
                        else:
                            st.error(message)

                    # Row 3: Delete User
                    st.subheader("Delete User")
                    if st.button("Delete User", key=f"btn_del_{user['username']}"):
                        success, message = delete_user(user['username'])
                        if success:
                            st.success(message)
                            create_log_entry(f"Admin Action: Deleted user - {user['username']}")
                            st.rerun()
                        else:
                            st.error(message)
    else:
        st.info("No users found")


def cleanup_on_logout(username = st.session_state.get("username"), refresh = True):
    """Handle cleanup when user logs out"""
    # username = st.session_state.get("username")
    if username:
        # Clear files
        directory = username
        if os.path.exists(directory):
            delete_mp3_files(directory)
            directory = "./" + directory
            os.rmdir(directory)

    if refresh:
        st.session_state.clear()   

def user_exists(username):
    """Check if user exists in database"""
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        return user is not None
    except Exception as e:
        print(f"Database error checking user existence: {e}")
        return False
    finally:
        session.close()


def heartbeat(username):
    # print(f"Heartbeat for user: {username}")
    if not user_exists(username):
        print(f"User {username} not found. Stopping heartbeat.")
        # Use the instance to call stop_beating
        heartbeat_manager.active_threads[username].stop()
        cleanup_on_logout(username, refresh=False)  
        st.rerun()
        st.session_state["user_deleted"] = True

    

class HeartbeatManager:
    def __init__(self):
        self.active_threads = {}

    def start_beating(self, username):
        """Start heartbeat thread"""
        def heartbeat_loop():
            while not self.active_threads[username].stopped():
                # Get current session context
                ctx = get_script_run_ctx()
                runtime = get_instance()

                if ctx and runtime.is_active_session(session_id=ctx.session_id):
                    # Session is still active
                    heartbeat(username)
                    time.sleep(2)  # Wait for 2 seconds
                else:
                    # Session ended - clean up
                    print(f"Session ended for user: {username}")
                    cleanup_on_logout(username, refresh=False)
                    return

        # Create new killable thread
        thread = KillableThread(target=heartbeat_loop)
        
        # Add Streamlit context to thread
        add_script_run_ctx(thread)
        
        # Store thread reference
        self.active_threads[username] = thread
        
        # Start thread
        thread.start()

    def stop_beating(self, username):
        """Stop heartbeat thread for specific user"""
        if username in self.active_threads:
            self.active_threads[username].stop()
            # Remove join() if called from within the thread
            if threading.current_thread() != self.active_threads[username]:
                self.active_threads[username].join()  # Only join if called from a different thread
            del self.active_threads[username]

    def stop_all(self):
        for username in list(self.active_threads.keys()):
            self.stop_beating(username)

#!important
heartbeat_manager = HeartbeatManager()


def start_beating(username):
    """Start heartbeat thread"""
    thread = threading.Timer(interval=2, function=start_beating, args=(username,))
    
    # Add Streamlit context to thread
    add_script_run_ctx(thread)
    
    # Get current session context
    ctx = get_script_run_ctx()
    runtime = get_instance()
    
    
    if ctx and runtime.is_active_session(session_id=ctx.session_id):
        # Session is still active
        thread.start()
        #!no logs
        heartbeat(username)
    else:
        # Session ended - clean up
        print(f"Session ended for user: {username}")
        cleanup_on_logout(username, refresh=False)
        return

# Authenticate function
def authenticate(username, password):
    try:
        session = Session()
        user = session.query(User).filter_by(username=username).first()
        session.close()
        if user and verify_password(user.password, password):  # Check password
            return user
    except Exception as e:
        st.error(f"Database error: {e}")
        return None
    
def verify_password(stored_hash: str, input_password: str) -> bool:
    """Verify if the input password matches the stored hashed password."""
    # Convert the stored hash back to bytes if it's a string
    stored_hash_bytes = stored_hash.encode('utf-8') if isinstance(stored_hash, str) else stored_hash
    return bcrypt.checkpw(input_password.encode('utf-8'), stored_hash_bytes)

# Login Page
def login_page():
    """Display login form and authenticate users."""
    st.title("Login Portal")
    

    # Group inputs and button in a form for "Enter" support
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        # Visible Login button (inside the form for "Enter" key support)
        login_button = st.form_submit_button("Login")

    # Handle login logic when either the "Login" button is clicked or "Enter" is pressed
    if login_button:
        user = authenticate(username, password)  # Call authentication function
        if user:
            st.session_state["logged_in"] = True
            st.session_state["username"] = user.username
            st.rerun()
        else:
            st.error("Invalid username or password!")

def save_audio_file(audio_bytes, name):
    try:
        if not audio_bytes:
            print(f"📂 Received {len(audio_bytes)} bytes for {name}")
            print(f"Error: No data in {name}")
            return None

        if name.lower().endswith(".wav") or name.lower().endswith(".mp3"):
            username = st.session_state.get("username", "default_user")
            user_folder = os.path.join(".", username)
            os.makedirs(user_folder, exist_ok=True)

            # Sanitize filename and construct full path
            name = os.path.basename(name)
            base_name, ext = os.path.splitext(name)  
            short_name = base_name[:20]  
            new_file_name = os.path.join(user_folder, f"{short_name}{ext}")

            # Ensure unique file name
            counter = 1
            while os.path.exists(new_file_name):
                print(f"✅ File already exists: {new_file_name}")
                return os.path.abspath(new_file_name)

            # Save file
            with open(new_file_name, "wb") as f:
                f.write(audio_bytes)

            # Ensure file is actually written
            time.sleep(1)  
            full_path = os.path.abspath(new_file_name)

            if os.path.exists(full_path):  
                print(f"✅ File successfully saved at: {full_path}")
                return full_path  
            else:
                print(f"❌ File not found after writing: {full_path}")
                return None

    except Exception as e:
        print(f"❌ Failed to save file: {e}")
        return None
        
    except Exception as e:
        print(f"Failed to save file: {e}")
        return None  # Explicitly return None on failure

def delete_mp3_files(directory):
    # Construct the search pattern for MP3 files
    mp3_files = glob.glob(os.path.join(directory, "*.mp3"))
    
    for mp3_file in mp3_files:
        try:
            os.remove(mp3_file)
            # print(f"Deleted: {mp3_file}")
        except FileNotFoundError:
            print(f"{mp3_file} does not exist.")
        except Exception as e:
            print(f"Error deleting file {mp3_file}: {e}")

def convert_audio_to_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)
    wav_file = audio_file.name.split(".")[0] + ".wav"
    audio.export(wav_file, format="wav")
    return wav_file

def make_fetch_request(url, headers, method='GET', data=None):
    if method == 'POST':
        response = requests.post(url, headers=headers, json=data)
    else:
        response = requests.get(url, headers=headers)
    return response.json()

def speech_to_text_groq(audio_file):
    #print into dialog format
    dialog =""

    #Function to run Groq with user prompt
    #different model from Groq
    # Groq_model="llama3-8b-8192"
    # Groq_model="llama3-70b-8192"
    # Groq_model="mixtral-8x7b-32768"
    # Groq_model="gemma2-9b-it"

    # Transcribe the audio
    audio_model="whisper-large-v3-turbo"

    with open(audio_file, "rb") as file:
        # Create a transcription of the audio file
        transcription = groq_client.audio.transcriptions.create(
        file=(audio_file, file.read()), # Required audio file
        model= audio_model, # Required model to use for transcription
        prompt="Elena Pryor, Samir, Sahil, Mihir, IPP, IPPFA",
        temperature=0,
        response_format="verbose_json"
          
        )

    # Print the transcription text
    print(transcription.text)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": """Insert speaker labels for a telemarketer and a customer. Return in a JSON format together with the original language code. Always translate the transcript fully to English."""},
        {"role": "user", "content": f"The audio transcript is: {transcription.text}"}
        ],
        temperature=0,
        max_tokens=16384
    )

    output = response.choices[0].message.content
    print(output)
    dialog = output.replace("json", "").replace("```", "")
    formatted_transcript = ""
    dialog = json.loads(dialog)
    language_code = dialog["language_code"]
    print(language_code)
    for entry in dialog['transcript']:
        formatted_transcript += f"{entry['speaker']}: {entry['text']}  \n\n"
    print(formatted_transcript)

    # Joining the formatted transcript into a single string
    dialog = formatted_transcript

    
    return dialog, language_code



def speech_to_text(audio_file):
    dialog =""

    # Transcribe the audio
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_file, "rb"),
        prompt="Elena Pryor, Samir, Sahil, Mihir, IPP, IPPFA",
        temperature=0

    )
    dialog = transcription.text
    # OPTIONAL: Uncomment the line below to print the transcription
    # print("Transcript: ", dialog + "  \n\n")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": """Insert speaker labels for a telemarketer and a customer. Return in a JSON format together with the original language code. Always translate the transcript fully to English."""},
        {"role": "user", "content": f"The audio transcript is: {dialog}"}
        ],
        temperature=0
    )

    output = response.choices[0].message.content
    # print(output)
    dialog = output.replace("json", "").replace("```", "")
    formatted_transcript = ""
    dialog = json.loads(dialog)
    language_code = dialog["language_code"]
    print(language_code)
    for entry in dialog['transcript']:
        formatted_transcript += f"{entry['speaker']}: {entry['text']}  \n\n"
    print(formatted_transcript)

    # Joining the formatted transcript into a single string
    dialog = formatted_transcript

    # Displaying the output
    # print(dialog)
    

    # gladia_key = "fb8ac339-e239-4253-926a-c963e2a1162b"
    # gladia_url = "https://api.gladia.io/audio/text/audio-transcription/"

    # headers = {
    #     "x-gladia-key": gladia_key
    # }

    # filename, file_ext = os.path.splitext(audio_file)

    # with open(audio_file, 'rb') as audio:
    #     # Prepare data for API request
    #     files = {
    #         'context_prompt': "Label the speeches in the dialog either as Telemarketer or Customer.",
    #         'audio':(filename, audio, f'audio/{file_ext[1:]}'),  # Specify audio file type
    #         'toggle_diarization': (None, 'True'),  # Toggle diarization option
    #         'diarization_max_speakers': (None, '2'),  # Set the maximum number of speakers for diarization
    #         'diarization_enhanced': True,
    #         'output_format': (None, 'txt'),  # Specify output format as text
    #     }

    #     print('Sending request to Gladia API')

    #     # Make a POST request to Gladia API
    #     response = requests.post(gladia_url, headers=headers, files=files)

    #     if response.status_code == 200:
    #         # If the request is successful, parse the JSON response
    #         response_data = response.json()

    #         # Extract the transcription from the response
    #         dialog = response_data['prediction']

    #     else:
    #         # If the request fails, print an error message and return the JSON response
    #         print(f'Request failed with status code {response.status_code}')
    #         return response.json()
    

    # # Define the API URL
    # url = "https://api.gladia.io/v2/upload"


    # file_name, file_extension = os.path.splitext(audio_file)

    # with open(audio_file, 'rb') as file:
    #     file_content = file.read()

    # headers = {
    # 'x-gladia-key': 'fb8ac339-e239-4253-926a-c963e2a1162b',
    # 'accept': 'application/json'
    # }

    # files = [("audio", (file_name, file_content, "audio/" + file_extension[1:]))]

    # response = requests.post(url, headers=headers, files=files)
    # # Check if the request was successful
    # if response.status_code == 200:
    #     # If the response is JSON, parse it
    #     try:
    #         result = response.json()
    #         print(result)  # Print the parsed JSON response
    #     except ValueError:
    #         print("Error: Response content is not valid JSON")
    # else:
    #     print(f"Error: Request failed with status code {response.status_code}")
    #     print(response.text)  # Print the raw response text for debugging

    # url = "https://api.gladia.io/v2/transcription"

    # payload = {
    #     "context_prompt": "An audio conversation between a telemarketer trying to selling their company's products and services and a customer.",
    #     "detect_language": True,
    #     "diarization": True,
    #     "diarization_enhanced": True,
    #     "diarization_config": {
    #         "number_of_speakers": 2,
    #         "min_speakers": 2,
    #         "max_speakers": 2
    #     },
    #     "translation": True,
    #     "translation_config":
    #     {
    #     "target_languages": ['en'],
    #     "model": "enhanced",
    #     "match_original_utterances": True
    #     },
    #     "custom_vocabulary": ["Elena Pryor, Samir, Sahil, Mihir, IPP, IPPFA"],
    #     "audio_url": result["audio_url"],
    #     "sentences": True,
    # }
    # headers = {
    #     "x-gladia-key": "fb8ac339-e239-4253-926a-c963e2a1162b",
    #     "Content-Type": "application/json"
    # }

    # response = requests.request("POST", url, json=payload, headers=headers)
    # if response.status_code == 201:
    #     dialog = ""

    #     response = response.json()

    #     print(response)

    #     response_id = response["id"]

    #     print(response_id)

    #     url = f"https://api.gladia.io/v2/transcription/{response_id}"

    #     print(url)

    #     headers = {"x-gladia-key": "fb8ac339-e239-4253-926a-c963e2a1162b"}

    #     response = requests.request("GET", url, headers=headers)

    #     print(response.status_code)

    #     if response.status_code == 200:
    #         # If the request is successful, parse the JSON response
    #         response_data = response.json()
    #         # print(response_data)
    #         while response_data["completed_at"] == None:

    #             response = requests.request("GET", url, headers=headers)
    #             response_data = response.json()


    #         labeled_dialog = []
    #         combined_dialog = ""
    #         previous_speaker = None
    #         combined_text = ""

    #         print(response_data)
    #         for utterance in response_data['result']['diarization_enhanced']['results']:
    #             speaker = utterance['speaker']
    #             text = utterance['text']

    #             # If the current speaker is the same as the previous one, combine the text
    #             if speaker == previous_speaker:
    #                 combined_text += f" {text}"
    #             else:
    #                 # If we have combined text from the previous speaker, add it to the dialog
    #                 if previous_speaker is not None:
    #                     labeled_dialog.append({'label': previous_speaker, 'text': combined_text})
                    
    #                 # Update the speaker and start a new combined text
    #                 previous_speaker = speaker
    #                 combined_text = text

    #         # Add the last combined entry to the dialog
    #         if previous_speaker is not None:
    #             labeled_dialog.append({'label': previous_speaker, 'text': combined_text})

    #         # Print the combined dialog
    #         for entry in labeled_dialog:
    #             combined_dialog += f"{entry['label']}: {entry['text']}  \n\n"
    #         dialog = combined_dialog
    # else:
    #     # If the request fails, print an error message and return the JSON response
    #     print(f'Request failed with status code {response.status_code}')
    #     return response.json()


    # try:
    #     with open(audio_file, "rb") as file:
    #         buffer_data = file.read()

    #     payload: FileSource = {
    #         "buffer": buffer_data,
    #     }

    #     options = PrerecordedOptions(
    #         model="whisper-medium",
    #         language="en",
    #         smart_format=True,
    #         punctuate=True,
    #         paragraphs=True,
    #         diarize=True,
    #     )

    #     response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

    #     response = response.to_json(indent=4)
    #     response_dict = json.loads(response)
    #     # print(response_dict['results']['channels'][0]['alternatives'][0]['paragraphs']['transcript'])
    #     dialog = response_dict['results']['channels'][0]['alternatives'][0]['paragraphs']['transcript']
    
    # except Exception as e:
    #     print(f"Exception: {e}")




    # try:
    #     aai.settings.api_key = "c6ecc4cf36fa4d9f9eb3dbdedcc1f772" 

    #     config = aai.TranscriptionConfig(
    #         speech_model=aai.SpeechModel.best,
    #         sentiment_analysis=True,
    #         entity_detection=True,
    #         speaker_labels=True,
    #         language_detection=True,
    #     )

    #     dialog = ""

    #     for utterance in aai.Transcriber().transcribe(audio_file, config).utterances:
    #         # print(f"Speaker {utterance.speaker}: {utterance.text}")
    #         dialog += f"Speaker {utterance.speaker}: {utterance.text}  \n\n"

    #     print(dialog)

    # except Exception as e:
    #     print(f"Exception: {e}")


    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #     {"role": "system", "content": """Identify which speaker is the telemarketer and the customer from the speaker labels. Output in JSON format."""},
    #     {"role": "user", "content": f"The audio transcript is: {dialog}"}
    #     ],
    #     temperature=0
    # )

    # output = response.choices[0].message.content
    # print(output)   
    # output = output.replace("json", "").replace("```", "")
    # output = json.loads(output)

    # # Replace speaker names in the dialog with their roles
    # for role, speaker in output.items():
    #     if speaker:  # Ensure speaker is not null
    #         dialog = re.sub(r'\b' + re.escape(speaker) + r'\b', role.capitalize(), dialog, flags=re.I)

    # print(dialog)

    return dialog, language_code


def groq_LLM_audit(dialog):
    stage_1_prompt = """
    You are an auditor for IPP or IPPFA. 
    You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
    The audit evaluates whether the telemarketer adhered to specific criteria during the dialogue.

    ### Instruction:
        - Review the provided conversation transcript and assess the telemarketer's compliance based on the criteria outlined below. 
        - For each criterion, provide a detailed assessment, including quoting reasons from the conversation and a result status. 
        - Ensure all evaluations are based strictly on the content of the conversation. 
        - Only mark a criterion as "Pass" if you are very confident (i.e., nearly certain) based on clear and specific evidence from the conversation. 
        - Do not include words written in the brackets () as part of the criteria during the response.
        - The meeting is always a location in Singapore, rename the location to a similar word if its not a location in Singapore.

        Audit Criteria:
            1. Did the telemarketer introduced themselves by stating their name? (Usually followed by 'calling from')
            2. Did the telemarketer state that they are calling from one of these ['IPP', 'IPPFA', 'IPP Financial Advisors'] without mentioning on behalf of any other insurers?(accept anyone one of the 3 name given)
            3. Did the customer asked how did the telemarketer obtained their contact details? If they asked, did telemarketer mentioned who gave the customer's details to him? (Not Applicable if customer didn't)
            4. Did the telemarketer specify the types of financial services offered?
            5. Did the telemarketer offered to set up a meeting or zoom session with the consultant for the customer? (Try to specify the date and location if possible)
            6. Did the telemarketer stated that products have high returns, guaranteed returns, or capital guarantee? (Fail if they did, Pass if they didn't)
            7. Was the telemarketer polite and professional in their conduct?

        ** End of Criteria**

    ### Response:
        Generate JSON objects for each criteria in a list that must include the following keys:
        - "Criteria": State the criterion being evaluated.
        - "Reason": Provide specific reasons based on the conversation.
        - "Result": Indicate whether the criterion was met with "Pass", "Fail", or "Not Applicable".

        For Example:
            [
                {
                    "Criteria": "Did the telemarketer asked about the age of the customer",
                    "Reason": "The telemarketer asked how old the customer was.",
                    "Result": "Pass"
                }
            ]
    """ 

    stage_2_prompt = """
    You are an auditor for IPP or IPPFA. 
    You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
    The audit evaluates whether the telemarketer adhered to specific criteria during the dialogue.

    ### Instruction:
        - Review the provided conversation transcript and assess the telemarketer's compliance based on the criteria outlined below. 
        - For each criterion, provide a detailed assessment, including quoting reasons from the conversation and a result status. 
        - Ensure all evaluations are based strictly on the content of the conversation. 
        - Only mark a criterion as "Pass" if you are very confident (i.e., nearly certain) based on clear and specific evidence from the conversation. 
        - Do not include words written in the brackets () as part of the criteria during the response.

        Audit Criteria:
            1. Did the telemarketer ask if the customer is keen to explore how they can benefit from IPPFA's services?
            2. Did the customer show uncertain response to the offer of the product and services? If Yes, Check did the telemarketer propose meeting or zoom session with company's consultant?
            3. Did the telemarketer pressure the customer for the following activities (product introduction, setting an appointment)? (Fail if they did, Pass if they didn't)

        ** End of Criteria**

    ### Response:
        Generate JSON objects for each criteria in a list that must include the following keys:
        - "Criteria": State the criterion being evaluated.
        - "Reason": Provide specific reasons based on the conversation.
        - "Result": Indicate whether the criterion was met with "Pass", "Fail", or "Not Applicable".

        For Example:
            [
                {
                    "Criteria": "Did the telemarketer asked about the age of the customer",
                    "Reason": "The telemarketer asked how old the customer was.",
                    "Result": "Pass"
                }
            ]

    ### Input:
        %s
    """ % (dialog)

    chat_completion  = groq_client.chat.completions.create(
    model="llama3-groq-70b-8192-tool-use-preview",
    messages=[
        {
            "role": "system",
            "content": f"{stage_1_prompt}",
        },
        {
            "role": "user",
            "content": f"{dialog}",
        }
    ],
    temperature=0,
    max_tokens=4096,
    stream=False,
    stop=None,
    )
    stage_1_result = chat_completion.choices[0].message.content
    print(stage_1_result)
    

    stage_1_result = stage_1_result.replace("Audit Results:","")
    stage_1_result = stage_1_result.replace("### Input:","")
    stage_1_result = stage_1_result.replace("### Output:","")
    stage_1_result = stage_1_result.replace("### Response:","")
    stage_1_result = stage_1_result.replace("json","").replace("```","")
    stage_1_result = stage_1_result.strip()

    stage_1_result = json.loads(stage_1_result)

    print(stage_1_result)

    output_dict = {"Stage 1": stage_1_result}

    # for k,v in output_dict.items():
    #    person_names.append(get_person_entities(v[0]["Reason"]))

    #    if len(person_names) != 0:
            # print(person_names)
    #        v[0]["Result"] = "Pass"

    # print(output_dict)

    overall_result = "Pass"

    for i in range(len(stage_1_result)):
        if stage_1_result[i]["Result"] == "Fail":
            overall_result = "Fail"
            break  

    output_dict["Overall Result"] = overall_result

    if output_dict["Overall Result"] == "Pass":
        del output_dict["Overall Result"]

        chat_completion  = groq_client.chat.completions.create(
        model="llama3-groq-70b-8192-tool-use-preview",
        messages=[
            {
                "role": "system",
                "content": f"{stage_2_prompt}",
            },
            {
                "role": "user",
                "content": f"{dialog}",
            }
        ],
        temperature=0,
        max_tokens=4096,
        stream=False,
        stop=None,
        )
        stage_2_result = chat_completion.choices[0].message.content
        
        stage_2_result = stage_2_result.replace("Audit Results:","")
        stage_2_result = stage_2_result.replace("### Input:","")
        stage_2_result = stage_2_result.replace("### Output:","")
        stage_2_result = stage_2_result.replace("### Response:","")
        stage_2_result = stage_2_result.replace("json","").replace("```","")
        stage_2_result = stage_2_result.strip()

        # print(stage_2_result)

        stage_2_result = json.loads(stage_2_result)
        
        output_dict["Stage 2"] = stage_2_result

        overall_result = "Pass"

        for i in range(len(stage_2_result)):
            if stage_2_result[i]["Result"] == "Fail":
                overall_result = "Fail"
                break  
                
        output_dict["Overall Result"] = overall_result

    # print(output_dict)
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return output_dict
        

    



def LLM_audit(dialog):
    stage_1_prompt = """
    You are an auditor for IPP or IPPFA. 
    You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
    The audit evaluates whether the telemarketer adhered to specific criteria during the dialogue.

    ### Instruction:
        - Review the provided conversation transcript and assess the telemarketer's compliance based on the criteria outlined below. 
        - For each criterion, provide a detailed assessment, including quoting reasons from the conversation and a result status. 
        - Ensure all evaluations are based strictly on the content of the conversation. 
        - Only mark a criterion as "Pass" if you are very confident (i.e., nearly certain) based on clear and specific evidence from the conversation. 
        - Do not include words written in the brackets () as part of the criteria during the response.
        - The meeting is always a location in Singapore, rename the location to a similar word if its not a location in Singapore.

        Audit Criteria:
            1. Did the telemarketer introduced themselves by stating their name? (Usually followed by 'calling from')
            2. Did the telemarketer state that they are calling from one of these ['IPP', 'IPPFA', 'IPP Financial Advisors'] without mentioning on behalf of any other insurers?(accept anyone one of the 3 name given)
            3. Did the customer asked how did the telemarketer obtained their contact details? If they asked, did telemarketer mentioned who gave the customer's details to him? (Not Applicable if customer didn't)
            4. Did the telemarketer specify the types of financial services offered?
            5. Did the telemarketer offered to set up a meeting or zoom session with the consultant for the customer? (Specify the date and location if possible)
            6. Did the telemarketer stated that products have high returns, guaranteed returns, or capital guarantee? (Fail if they did, Pass if they didn't)
            7. Was the telemarketer polite and professional in their conduct?

        ** End of Criteria**

    ### Response:
        Generate JSON objects for each criteria in a list that must include the following keys:
        - "Criteria": State the criterion being evaluated.
        - "Reason": Provide specific reasons based on the conversation.
        - "Result": Indicate whether the criterion was met with "Pass", "Fail", or "Not Applicable".

        For Example:
            [
                {
                    "Criteria": "Did the telemarketer asked about the age of the customer",
                    "Reason": "The telemarketer asked how old the customer was.",
                    "Result": "Pass"
                }
            ]

    ### Input:
        %s
    """ % (dialog)

    stage_2_prompt = """
    You are an auditor for IPP or IPPFA. 
    You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
    The audit evaluates whether the telemarketer adhered to specific criteria during the dialogue.

    ### Instruction:
        - Review the provided conversation transcript and assess the telemarketer's compliance based on the criteria outlined below. 
        - For each criterion, provide a detailed assessment, including quoting reasons from the conversation and a result status. 
        - Ensure all evaluations are based strictly on the content of the conversation. 
        - Only mark a criterion as "Pass" if you are very confident (i.e., nearly certain) based on clear and specific evidence from the conversation. 
        - Do not include words written in the brackets () as part of the criteria during the response.

        Audit Criteria:
            1. Did the telemarketer ask if the customer is keen to explore how they can benefit from IPPFA's services?
            2. Did the customer show uncertain response to the offer of the product and services? If Yes, Check did the telemarketer propose meeting or zoom session with company's consultant?
            3. Did the telemarketer pressure the customer for the following activities (product introduction, setting an appointment)? (Fail if they did, Pass if they didn't)

        ** End of Criteria**

    ### Response:
        Generate JSON objects for each criteria in a list that must include the following keys:
        - "Criteria": State the criterion being evaluated.
        - "Reason": Provide specific reasons based on the conversation.
        - "Result": Indicate whether the criterion was met with "Pass", "Fail", or "Not Applicable".

        For Example:
            [
                {
                    "Criteria": "Did the telemarketer asked about the age of the customer",
                    "Reason": "The telemarketer asked how old the customer was.",
                    "Result": "Pass"
                }
            ]

    ### Input:
        %s
    """ % (dialog)


    # Set up the model and prompt
    # model_engine = "text-davinci-003"
    model_engine ="gpt-4o-mini"

    messages=[{'role':'user', 'content':f"{stage_1_prompt}"}]


    completion = client.chat.completions.create(
    model=model_engine,
    messages=messages,
    temperature=0,)

    # print(completion)

    # extracting useful part of response
    stage_1_result = completion.choices[0].message.content
    stage_1_result = stage_1_result.replace("Audit Results:","")
    stage_1_result = stage_1_result.replace("### Input:","")
    stage_1_result = stage_1_result.replace("### Output:","")
    stage_1_result = stage_1_result.replace("### Response:","")
    stage_1_result = stage_1_result.replace("json","").replace("```","")
    stage_1_result = stage_1_result.strip()

    print(stage_1_result)

    def format_json_with_line_break(json_string):
        # Step 1: Add missing commas after "Criteria" and "Reason" key-value pairs
        corrected_json = re.sub(r'("Criteria":\s*".+?")(\s*")', r'\1,\2', json_string)
        corrected_json = re.sub(r'("Reason":\s*".+?")(\s*")', r'\1,\2', corrected_json)

        # Ensure there is a newline after the comma for "Criteria"
        corrected_json = re.sub(r'("Criteria":\s*".+?"),(\s*")', r'\1,\n\2', corrected_json)
        
        # Ensure there is a newline after the comma for "Reason"
        corrected_json = re.sub(r'("Reason":\s*".+?"),(\s*")', r'\1,\n\2', corrected_json)

        return corrected_json

    def get_person_entities(text):
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Apply part-of-speech tagging
        pos_tags = pos_tag(tokens)
        
        # Perform named entity recognition (NER)
        named_entities = ne_chunk(pos_tags)
        
        # Extract PERSON entities
        person_entities = []
        for chunk in named_entities:
            if isinstance(chunk, Tree) and chunk.label() == 'PERSON' or isinstance(chunk, Tree) and chunk.label() == 'FACILITY' or isinstance(chunk, Tree) and chunk.label() == 'GPE':
                person_name = ' '.join([token for token, pos in chunk.leaves()])
                person_entities.append(person_name)
        return person_entities

    # person_names = []


    stage_1_result = format_json_with_line_break(stage_1_result)
    stage_1_result = json.loads(stage_1_result)

    output_dict = {"Stage 1": stage_1_result}

    # for k,v in output_dict.items():
    #    person_names.append(get_person_entities(v[0]["Reason"]))

    #    if len(person_names) != 0:
            # print(person_names)
    #        v[0]["Result"] = "Pass"

    # print(output_dict)

    overall_result = "Pass"

    for i in range(len(stage_1_result)):
        if stage_1_result[i]["Result"] == "Fail":
            overall_result = "Fail"
            break  

    output_dict["Overall Result"] = overall_result

    if output_dict["Overall Result"] == "Pass":
        del output_dict["Overall Result"]


        messages=[{'role':'user', 'content':f"{stage_2_prompt}"}]

        model_engine ="gpt-4o-mini"

        completion = client.chat.completions.create(
        model=model_engine,
        messages=messages,
        temperature=0,)

        # print(completion)

        # extracting useful part of response
        stage_2_result = completion.choices[0].message.content
        
        stage_2_result = stage_2_result.replace("Audit Results:","")
        stage_2_result = stage_2_result.replace("### Input:","")
        stage_2_result = stage_2_result.replace("### Output:","")
        stage_2_result = stage_2_result.replace("### Response:","")
        stage_2_result = stage_2_result.replace("json","").replace("```","")
        stage_2_result = stage_2_result.strip()

        print(stage_2_result)

        stage_2_result = format_json_with_line_break(stage_2_result)
        stage_2_result = json.loads(stage_2_result)
        
        output_dict["Stage 2"] = stage_2_result

        overall_result = "Pass"

        for i in range(len(stage_2_result)):
            if stage_2_result[i]["Result"] == "Fail":
                overall_result = "Fail"
                break  
                
        output_dict["Overall Result"] = overall_result

    print(output_dict)
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return output_dict



def select_folder():
   root = tk.Tk()
   root.wm_attributes('-topmost', 1)
   root.withdraw()
   folder_path = filedialog.askdirectory(parent=root)
    
   root.destroy()
   return folder_path

def create_log_entry(event_description, log_file='logfile.txt', csv_file='logfile.csv'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Log to text file
    with open(log_file, mode='a') as file:
        file.write(f"{timestamp} - {event_description}\n")
    
    # Log to CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file is new
        if file.tell() == 0:
            writer.writerow(["timestamp", "event_description"])
        writer.writerow([timestamp, event_description])

@st.fragment
def handle_download_json(count, data, file_name, mime, log_message):
    st.download_button(
        label=f"Download {file_name.split('.')[-1].upper()}", 
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
        key=f"download_json_{count}"
    )

@st.fragment
def handle_download_csv(count, data, file_name, mime, log_message):
    st.download_button(
        label=f"Download {file_name.split('.')[-1].upper()}", 
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
        key=f"download_csv_{count}"
    )

@st.fragment
def handle_download_log_file(data, file_name, mime, log_message):
    st.download_button(
        label="Download Logs", 
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
    )

@st.fragment
def handle_download_text(count, data, file_name, mime, log_message):
    st.download_button(
        label="Download Transcript", 
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
        key=f"download_text_{count}"
    )

@st.fragment
def zip_download(count, data, file_name, mime, log_message):
    st.download_button(
        label="Download All Files as ZIP",
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
        key=f"download_zip_{count}"
    )

@st.fragment
def combined_audit_result_download(data, file_name, mime, log_message):
    st.download_button(
        label="Download Combined Audit Results As ZIP",
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
    )


@st.fragment
def handle_combined_audit_result_download(data_text, data_csv, file_name_prefix):
    # Create an in-memory buffer for the ZIP file
    buffer = io.BytesIO()

    # Convert CSV data to a pandas DataFrame
    df = pd.read_csv(io.StringIO(data_csv))

    # Create an in-memory buffer for the Excel file (XLSX)
    xlsx_buffer = io.BytesIO()

    # Write DataFrame to XLSX format and add hyperlinks to the filenames
    with pd.ExcelWriter(xlsx_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')

        # Access the xlsxwriter workbook and worksheet objects
        # workbook = writer.book
        worksheet = writer.sheets['Results']

        # Add hyperlinks to text files based on the filenames in the DataFrame
        for index, row in df.iterrows():
            filename = row['Filename']  # Assuming 'filename' column exists
            # print(filename)
            if pd.notna(filename):  # Check if filename is not NaN (valid string)
                # Replace file extensions and create the hyperlink
                hyperlink = f"./{filename.replace('.mp3', '.txt').replace('.wav', '.txt')}"
                # Add the hyperlink to the 'filename' column in the Excel file (adjust the column index)
                worksheet.write_url(f"F{index + 2}", hyperlink, string=filename)

    # Move the pointer to the beginning of the xlsx_buffer to prepare it for reading
    xlsx_buffer.seek(0)

    with zipfile.ZipFile(buffer, "w") as zip_file:
        # Add text files
        for k, v in data_text.items():
            zip_file.writestr(k.replace(".mp3", ".txt").replace(".wav", ".txt"), v)

        # Add the CSV file as plain text
        # zip_file.writestr(f"{file_name_prefix}_file.csv", data_csv)

        # Add the XLSX file to the ZIP archive
        zip_file.writestr(f"{file_name_prefix}_file.xlsx", xlsx_buffer.read())

    # Move buffer pointer to the beginning of the ZIP buffer
    buffer.seek(0)

    # Return the buffer containing the ZIP archive
    return buffer

@st.fragment
def handle_combined_download(data_text, data_json, data_csv, file_name_prefix):
    # Create an in-memory buffer for the ZIP file
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w") as zip_file:
        # Add text file
        zip_file.writestr(f"{file_name_prefix}_file.txt", data_text)

        # Add JSON file
        zip_file.writestr(f"{file_name_prefix}_file.json", data_json)

        # Add CSV file
        zip_file.writestr(f"{file_name_prefix}_file.csv", data_csv)

    # Move buffer pointer to the beginning
    buffer.seek(0)

    # Return the buffer containing the ZIP archive
    return buffer


def read_log_file(log_file='logfile.txt'):
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            log_content = file.readlines()
        # Reverse the log entries
        log_content = log_content[::-1]
        if log_content:
            log_content[0] = f"<span style='color: yellow;'>{log_content[0].strip()}</span>\n"

        return ''.join(log_content)
    else:
        return "Log file does not exist."
    
    
def log_selection():
    method = st.session_state.upload_method
    if method == "Upload Files":
        create_log_entry("Method Chosen: File Upload")
    elif method == "Upload Folder":
        create_log_entry("Method Chosen: Folder Upload")

def is_valid_mp3(file_path):
    """Ensure the file exists and is a valid MP3"""
    file_path = os.path.abspath(file_path)  

    # Retry mechanism for file recognition
    retry_attempts = 3
    for _ in range(retry_attempts):
        if os.path.exists(file_path):
            break
        time.sleep(1)  # Wait before retrying
    else:
        print(f"❌ File still not found after retries: {file_path}")
        return False

    try:
        audio = MP3(file_path)
        if audio.info.length <= 0:  
            print("❌ Invalid MP3: length is zero.")
            return False
        
        print(f"✅ Valid MP3 with duration: {audio.info.length} seconds")
        return True
    except HeaderNotFoundError:
        print(f"❌ {file_path} has no valid MP3 headers")
        return False
    except Exception as e:
        print(f"❌ Invalid MP3 file: {e}")
        return False
    

def main():
    try:
        if st.session_state.get("user_deleted", False):
            st.error("Your account has been deleted. Please contact the administrator.")
            st.session_state.clear()
            time.sleep(1)  # Give a moment for the message to display
            st.rerun()

        # cookies = controller.get("username")
        # if cookies:
        #     st.session_state["logged_in"] = True
        #     st.session_state["username"] = cookies
        if "logged_in" not in st.session_state:
            st.session_state["logged_in"] = False
            st.session_state["username"] = None

        if not st.session_state["username"]:
            # Show the login page if not logged in
            login_page()
        else:
            if 'heartbeat_started' not in st.session_state:
                st.session_state.heartbeat_started = True
                heartbeat_manager.start_beating(st.session_state["username"])

            # After successful login, show the main dashboard
            st.sidebar.success(f"Logged in as: {st.session_state['username']}")

            if st.sidebar.button("Logout"):
                cleanup_on_logout()
                st.rerun()

            if is_admin(st.session_state["username"]):
                button_text = "Back to Transcription Service" if st.session_state.get('show_admin', False) else "Open Admin Panel"
                if st.sidebar.button(button_text, key="admin_toggle_button"):
                    st.session_state.show_admin = not st.session_state.get('show_admin', False)
                    st.rerun()



            if st.session_state.get('show_admin', False):
                cleanup_on_logout(username=st.session_state["username"], refresh=False)
                admin_interface()
                # if st.button("Back to Transcription Service"):
                #     st.session_state.show_admin = False
                #     st.rerun()
            else:
                with st.sidebar:
                    st.title("AI Model Selection")
                    transcribe_option = st.radio(
                        "Choose your transcription AI model:",
                        ("OpenAI (Recommended)", "Groq")
                    )
                    audit_option = st.radio(
                        "Choose your Audit AI model:",
                        ("OpenAI (Recommended)", "Groq")
                    )
                    st.write(f"Transcription Model:\n\n{transcribe_option.replace('(Recommended)','')}\n\nAudit Model:\n\n{audit_option.replace('(Recommended)','')}")
                    st.markdown('<p style="color:red;">Groq AI Models are not recommended for file sizes of more than 1MB. Model will start to hallucinate.</p>', unsafe_allow_html=True)

                st.title("AI Transcription & Audit Service")
                st.info("Upload an audio file or select a folder to convert audio to text.", icon=":material/info:")
                
                method = st.radio("Select Upload Method:", options=["Upload Files / Folder"], horizontal=True, key='upload_method', on_change=log_selection)


            

                audio_files = []
                status = ""

                # Initialize session state to track files and change detection
                if 'uploaded_files' not in st.session_state:
                    st.session_state.uploaded_files = {}
                if 'file_change_detected' not in st.session_state:
                    st.session_state.file_change_detected = False
                if 'audio_files' not in st.session_state:
                    st.session_state.audio_files = []

                save_to_path = st.session_state.get("save_to_path", None)

            # Choose upload method

            # with st.expander("Other Options"):
            #     save_audited_transcript = st.checkbox("Save Audited Results to Folder (CSV)")
            #     if save_audited_transcript:
            #         save_to_button = st.button("Save To Folder")
            #         if save_to_button:
            #             save_to_path = select_folder()  # Assume this is a function that handles folder selection
            #             if save_to_path:
            #                 st.session_state.save_to_path = save_to_path
            #                 create_log_entry(f"Action: Save To Folder - {save_to_path}")

            #     save_to_display = st.empty()

            #     if save_audited_transcript == False:
            #         st.session_state.save_to_path = None
            #         save_to_path = None
            #         save_to_display.empty()
            #     else:
            #         save_to_display.write(f"Save To Folder: {save_to_path}")

                if method == "Upload Files / Folder":
                    # File uploader
                    uploaded_files = st.file_uploader(
                        label="Choose audio files", 
                        label_visibility="collapsed", 
                        type=["wav", "mp3"], 
                        accept_multiple_files=True
                    )
                    st.write("Uploaded Files:", uploaded_files)
                    
                    # Ensure session states exist
                    if 'uploaded_files' not in st.session_state:
                        st.session_state.uploaded_files = {}

                    if 'audio_files' not in st.session_state:
                        st.session_state.audio_files = []

                    # Track currently uploaded files
                    current_files = {file.name: file for file in uploaded_files} if uploaded_files else {}

                    # Detect removed files (files that were previously uploaded but are now missing)
                    removed_files = [
                        file_name for file_name in st.session_state.uploaded_files 
                        if file_name not in current_files
                    ]

                    # Remove files from session state and delete them from the system
                    for file_name in removed_files:
                        # Remove from UI state
                        st.session_state.audio_files = [f for f in st.session_state.audio_files if not f.endswith(file_name)]
                        
                        # Get the actual saved file path
                        username = st.session_state["username"]
                        user_folder = os.path.abspath(os.path.join(".", username))  # Get absolute path for user folder
                        matching_files = glob.glob(os.path.abspath(os.path.join(user_folder, f"*{file_name}")))  # Match file

                        if matching_files:
                            for full_path in matching_files:
                                if os.path.exists(full_path):
                                    try:
                                        os.remove(full_path)  # Delete file from the system
                                        create_log_entry(f" File deleted: {full_path}")
                                    except Exception as e:
                                        st.error(f"❌ Error deleting file {file_name}: {e}")
                                        create_log_entry(f" Error deleting file {file_name}: {e}")
                        else:
                            create_log_entry(f"⚠️ File to delete NOT DETECTED: {file_name}")
                        
                        # Remove file from storage
                        # full_path = os.path.join(st.session_state["username"], f"audio_{file_name}")
                        # if os.path.exists(full_path):
                        #     try:
                        #         os.remove(full_path)  # Delete file from the system
                        #         create_log_entry(f"File deleted: {full_path}")
                        #     except Exception as e:
                        #         st.error(f"Error deleting file {file_name}: {e}")
                        # else:
                        #     create_log_entry("File to delete IS NOT DETECTED")

                        # Remove from session state
                        del st.session_state.uploaded_files[file_name]


                    # # Create a set to track unique filenames
                    # unique_filenames = set()

                    # # Check for duplicates and collect unique filenames
                    # for file in uploaded_files:
                    #     if file.name in unique_filenames:
                    #         del st.session_state['uploaded_files']
                    #         st.warning("File has already been added!")
                    #     else:
                    #         unique_filenames.add(file.name)


                    if uploaded_files is not None:
                        #!removing this because its broken
                        # # Track current files
                        # current_files = {file.name: file for file in uploaded_files}

                        # # Determine files that have been added
                        # added_files = [file_name for file_name in current_files if file_name not in st.session_state.uploaded_files]
                        # files_to_remove = [
                        #     file_name for file_name in st.session_state.uploaded_files 
                        #     if file_name not in added_files
                        # ]
                        # Track current files from the file uploader
                        current_files = {file.name: file for file in uploaded_files}

                        # Determine files that have been added (newly uploaded files)
                        added_files = [file_name for file_name in current_files if file_name not in st.session_state.uploaded_files]

                        # Determine files that have been removed (no longer in the uploader)
                        files_to_remove = [
                            file_name for file_name in st.session_state.uploaded_files 
                            if file_name not in current_files
                        ]

                        #* deletes the files that are not in the current_files
                        for file_name in files_to_remove:
                            # create_log_entry(f"Action: File Removed - {file_name}")

                            # Update `st.session_state.audio_files` to exclude the removed file
                            st.session_state.audio_files = [f for f in st.session_state.audio_files if not f.endswith(file_name)]
                            st.session_state.file_change_detected = True

                            username = st.session_state["username"]

                            # Delete the corresponding file from the directory
                            audio_file_name = "audio_" + file_name
                            full_path = os.path.join(username, audio_file_name)  # Root directory or adjust to your save directory
                            if os.path.exists(full_path):
                                try:
                                    # print(full_path)
                                    os.remove(full_path)  # Delete the file 
                                    
                                    create_log_entry(f"Action: File Deleted - {full_path}")
                                    del st.session_state.uploaded_files[file_name]

                                except Exception as e:
                                    st.error(f"Error deleting file {file_name}: {e}")
                                    create_log_entry(f"Error deleting file {file_name}: {e}")                    

                        if uploaded_files:
                            for file in uploaded_files:
                                try:
                                    audio_content = file.read()
                                    saved_path = save_audio_file(audio_content, file.name)

                                    if saved_path and os.path.exists(saved_path):  
                                        saved_path = os.path.abspath(saved_path)  # Ensure absolute path
                                        print(f"✅ File saved at: {saved_path}")
                                        
                                        if is_valid_mp3(saved_path):
                                            if saved_path not in st.session_state.audio_files:  # Prevent duplication
                                                st.session_state.audio_files.append(saved_path)
                                                print(f"✅ Added to session state: {saved_path}")
                                        else:
                                            st.error(f"❌ {saved_path} is an Invalid MP3 or WAV File")
                                    else:
                                        st.error("❌ Failed to save uploaded file.")
                                except Exception as e:
                                    st.error(f"❌ Error loading audio file: {e}")
                                    print(f"❌ Error loading audio file: {e}")

                        for file_name in added_files:
                            create_log_entry(f"Action: File Uploaded - {file_name}")
                            file = current_files[file_name]
                            st.session_state.uploaded_files[file_name] = current_files[file_name]

                            try:
                                file.seek(0)
                                audio_content = file.read()
                                saved_path = save_audio_file(audio_content, file_name)
                                if saved_path and os.path.exists(saved_path):  # Ensure the file is actually saved
                                    if is_valid_mp3(saved_path):
                                        if saved_path not in st.session_state.audio_files:  # Prevent duplication
                                            st.session_state.audio_files.append(saved_path)
                                    else:
                                        st.error(f"{saved_path} is an Invalid MP3 or WAV File")
                                else:
                                    if saved_path:
                                        create_log_entry(f"YAY File was returned: {saved_path}")
                                        if os.path.exists(saved_path):
                                            create_log_entry(f"YAY File successfully exists: {saved_path}")
                                        else:
                                            create_log_entry(f"BRUH File does not exist even though it was returned: {saved_path}")
                                    else:
                                        create_log_entry(f"BRUH save_audio_file() returned None")
                                    st.error("Failed to save the uploaded file.")
                            except Exception as e:
                                st.error(f"Error loading audio file: {e}")
                                create_log_entry(f"Error loading audio file: {e}")  

                        # Display Uploaded Audio Files
                        if st.session_state.uploaded_files:
                            st.subheader("Uploaded Audio Files Player")
                            updated_files = list(st.session_state.uploaded_files.keys())  # Updated file list

                            # Only display the remaining valid files
                            for file_name in updated_files:
                                if file_name in st.session_state.uploaded_files:
                                    with st.expander(f"Audio: {file_name}"):
                                        st.audio(st.session_state.uploaded_files[file_name], format="audio/mp3", start_time=0)

                        # Determine files that have been removed
                        # removed_files = [file_name for file_name in st.session_state.uploaded_files if file_name not in current_files]
                        # for file_name in removed_files:
                        #     create_log_entry(f"Action: File Removed - {file_name}")
                        #     st.session_state.audio_files = [f for f in st.session_state.audio_files if not f.endswith(file_name)]
                        #     st.session_state.file_change_detected = True

                        # Update session state with the current file list if a change was detected
                        if st.session_state.file_change_detected:
                            st.session_state.uploaded_files = current_files
                            st.session_state.file_change_detected = False

                    
                    audio_files = list(st.session_state.audio_files)
                    # st.write(audio_files)
                    # print(st.session_state.audio_files)
                    # print(type(audio_files))

                elif method == "Upload Folder":
                    # create_log_entry("Method Chosen: Folder Upload")
                    # Initialize the session state for folder_path
                    selected_folder_path = st.session_state.get("folder_path", None)

                    # Create two columns for buttons
                    col1, col2 = st.columns(spec=[2, 8])

                    with col1:
                        # Button to trigger folder selection
                        folder_select_button = st.button("Upload Folder")
                        if folder_select_button:
                            selected_folder_path = select_folder()  # Assume this is a function that handles folder selection
                            if selected_folder_path:
                                st.session_state.folder_path = selected_folder_path
                                create_log_entry(f"Action: Folder Uploaded - {selected_folder_path}")

                    with col2:
                        # Option to remove the selected folder
                        if selected_folder_path:
                            remove_folder_button = st.button("Remove Uploaded Folder")
                            if remove_folder_button:
                                username = st.session_state["username"]
                                st.session_state.folder_path = None
                                selected_folder_path = None
                                directory = username
                                delete_mp3_files(directory)
                                create_log_entry("Action: Uploaded Folder Removed")
                                success_message = "Uploaded folder has been removed."
                      
                        

                    # Display the success message if it exists
                    if 'success_message' in locals():
                        st.success(success_message)

                    # Display the selected folder path
                    if selected_folder_path:
                        st.write("Uploaded folder path:", selected_folder_path)

                        # Get all files in the selected folder
                        files_in_folder = os.listdir(selected_folder_path)
                        st.write("Files in the folder:")

                        # Process each file
                        for file_name in files_in_folder:
                            try:
                                file_path = os.path.join(selected_folder_path, file_name)
                                with open(file_path, 'rb') as file:
                                    audio_content = file.read()
                                    just_file_name = os.path.basename(file_name)
                                    save_path = os.path.join(just_file_name)
                                    saved_file_path = save_audio_file(audio_content, save_path)
                                    if is_valid_mp3(saved_file_path):
                                        audio_files.append(saved_file_path)
                                    else:
                                        st.error(f"{saved_file_path[2:]} is an Invalid MP3 or WAV File")
                                        create_log_entry(f"Error: {saved_file_path[2:]} is an Invalid MP3 or WAV File")


                            except Exception as e:
                                st.warning(f"Error processing file '{file_name}': {e}")
                        
                        #Filter files that are not in MP3 or WAV extensions
                        audio_files = list(filter(partial(is_not, None), audio_files))

                        st.write(audio_files)
                        # print(audio_files)

                st.write("Audio files in session state:", st.session_state.audio_files)

                # Submit button
                submit = st.button("Submit", use_container_width=True)

                if submit and audio_files == []:
                    create_log_entry("Service Request: Fail (No Files Uploaded)")
                    st.error("No Files Uploaded, Please Try Again!")


                elif submit:
                    combined_results = []
                    all_text = {}
                    # if not save_audited_transcript or (save_audited_transcript and save_to_path != None):
                    current = 1
                    end = len(audio_files)
                    for audio_file in audio_files:
                        print(audio_file)
                        if not os.path.isfile(audio_file):
                            st.error(f"{audio_file[2:]} Not Found, Please Try Again!")
                            continue
                        else:
                            try:
                                with st.spinner("Transcribing & Auditing In Progress..."):
                                    if transcribe_option == "OpenAI (Recommended)":   
                                        text, language_code = speech_to_text(audio_file)
                                        if audit_option == "OpenAI (Recommended)":
                                            result = LLM_audit(text)
                                            if result["Overall Result"] == "Fail":
                                                status = "<span style='color: red;'> (FAIL)</span>"
                                            else:
                                                status = "<span style='color: green;'> (PASS)</span>"
                                        elif audit_option == "Groq":
                                            result = groq_LLM_audit(text)
                                            if result["Overall Result"] == "Fail":
                                                status = "<span style='color: red;'> (FAIL)</span>"
                                            else:
                                                status = "<span style='color: green;'> (PASS)</span>"
                                    elif transcribe_option == "Groq":
                                        text, language_code = speech_to_text_groq(audio_file)
                                        if audit_option == "OpenAI (Recommended)":
                                            result = LLM_audit(text)
                                            if result["Overall Result"] == "Fail":
                                                status = "<span style='color: red;'> (FAIL)</span>"
                                            else:
                                                status = "<span style='color: green;'> (PASS)</span>"
                                        elif audit_option == "Groq":
                                            result = groq_LLM_audit(text)
                                            if result["Overall Result"] == "Fail":
                                                status = "<span style='color: red;'> (FAIL)</span>"
                                            else:
                                                status = "<span style='color: green;'> (PASS)</span>"
                                    
                                    # Ensure we only add non-duplicate results
                                    if audio_file not in all_text:
                                        all_text[audio_file[2:]] = text
                            except Exception as e:
                                error_message = f"Error processing file: {audio_file} - {e}"
                                create_log_entry(error_message)
                                st.error(error_message)
                                continue
                            col1, col2 = st.columns([0.9,0.1])
                            with col1:
                                with st.expander(audio_file[2:] + f" ({language_code})"):
                                # with st.expander(audio_file[2:]):
                                    st.write()
                                    tab1, tab2, tab3 = st.tabs(["Converted Text", "Audit Result", "Download Content"])
                            with col2:
                                st.write(f"({current} / {end})")
                                st.markdown(status, unsafe_allow_html=True)
                            with tab1:
                                st.write(text)
                                all_text[audio_file[2:]] = text
                                print(all_text)
                                handle_download_text(count=audio_files.index(audio_file) ,data=text, file_name=f'{audio_file[2:].replace(".mp3", ".txt").replace(".wav", ".txt")}', mime='text/plain', log_message="Action: Text File Downloaded")
                            with tab2:
                                st.write(result)
                                # Convert result to JSON string
                                json_data = json.dumps(result, indent=4)
                                filename = audio_file[2:]
                                if isinstance(result, dict) and "Stage 1" in result:
                                    cleaned_result_stage1 = result["Stage 1"]
                                    cleaned_result_stage2 = result.get("Stage 2", [])  # Default to an empty list if Stage 2 is not present
                                    overall_result = result.get("Overall Result", "Pass")
                                else:
                                    cleaned_result_stage1 = cleaned_result_stage2 = result
                                    overall_result = "Pass"

                                # Process Stage 1 results
                                if isinstance(cleaned_result_stage1, list) and all(isinstance(item, dict) for item in cleaned_result_stage1):
                                    df_stage1 = pd.json_normalize(cleaned_result_stage1)
                                    df_stage1['Stage'] = 'Stage 1'
                                else:
                                    df_stage1 = pd.DataFrame(columns=['Stage'])  # Create an empty DataFrame for Stage 1 if no valid results

                                # Process Stage 2 results
                                if isinstance(cleaned_result_stage2, list) and all(isinstance(item, dict) for item in cleaned_result_stage2):
                                    df_stage2 = pd.json_normalize(cleaned_result_stage2)
                                    df_stage2['Stage'] = 'Stage 2'
                                else:
                                    df_stage2 = pd.DataFrame(columns=['Stage'])  # Create an empty DataFrame for Stage 2 if no valid results

                                # Concatenate Stage 1 and Stage 2 results
                                df = pd.concat([df_stage1, df_stage2], ignore_index=True)

                                # Add the Overall Result as a new column (same value for all rows)
                                df['Overall Result'] = overall_result

                                # Add the filename as a new column (same value for all rows)
                                df['Filename'] = filename

                                # Save DataFrame to CSV
                                output = BytesIO()
                                df.to_csv(output, index=False)

                                # Get CSV data
                                csv_data = output.getvalue().decode('utf-8')

                                try:
                                    col1, col2 = st.columns([2, 6])
                                    with col1:
                                        handle_download_json(count=audio_files.index(audio_file) ,data=json_data, file_name=f'{audio_file[2:]}.json', mime='application/json', log_message="Action: JSON File Downloaded")

                                    with col2:
                                        handle_download_csv(count=audio_files.index(audio_file), data=csv_data, file_name=f'{audio_file[2:]}.csv', mime='text/csv', log_message="Action: CSV File Downloaded")
                                
                                except Exception as e:
                                    create_log_entry(f"{e}")
                                    st.error(f"Error processing data: {e}")
                            with tab3:
                                zip_buffer = handle_combined_download(
                                    data_text=text,
                                    data_json=json_data,
                                    data_csv=csv_data,
                                    file_name_prefix=audio_file[2:]
                                )
                                zip_download(count=audio_files.index(audio_file) ,data=zip_buffer, file_name=f'{audio_file[2:]}.zip', mime="application/zip", log_message="Action: Audited Results Zip File Downloaded")
                                
                        current += 1
                        create_log_entry(f"Successfully Audited: {audio_file[2:]}")
                        df.loc[len(df)] = pd.Series(dtype='float64')
                        combined_results.append(df)
                    #     if save_audited_transcript:
                    #         if save_to_path:
                    #             try:
                    #                 # Ensure the directory exists
                    #                 os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
                    #                 file_name_without_extension, _ = os.path.splitext(audio_file[2:])
                    #                 full_path = os.path.join(save_to_path, file_name_without_extension + ".csv")
                    #                 # df = pd.json_normalize(csv_data)
                    #                 # Save the DataFrame as a CSV to the specified path
                    #                 df.to_csv(full_path, index=False)
                    #                 print(f"Saved audited results (CSV) to {save_to_path}")
                    #             except Exception as e:
                    #                 print(f"Failed to save file: {e}")
                    #         else:
                    #             print("Save path not specified.")
                    # if save_audited_transcript:
                    #     if save_to_path:
                    #         combined_df = pd.concat(combined_results, ignore_index=True)
                    #         os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
                    #         full_path = os.path.join(save_to_path, "combined_results.csv")
                    #         combined_df.to_csv(full_path, index=False)

                    if combined_results != []:
                        # Concatenate all DataFrames
                        combined_df = pd.concat(combined_results, ignore_index=True)

                        # Create an in-memory CSV using BytesIO
                        output = BytesIO()
                        combined_df.to_csv(output, index=False)
                        output.seek(0)  # Reset buffer position to the start

                        # Get the CSV data as a string
                        combined_csv_data = output.getvalue().decode('utf-8')
                        with st.spinner("Preparing Consolidated Results..."):
                            zip_buffer = handle_combined_audit_result_download(
                                            data_text=all_text,
                                            data_csv=combined_csv_data,
                                            file_name_prefix="combined_audit_results"
                                        )
                            
                        combined_audit_result_download(data=zip_buffer, file_name='CombinedAuditResults.zip', mime="application/zip", log_message="Action: Audited Results Zip File Downloaded")  
                        username = st.session_state["username"]
                        directory = username
                        delete_mp3_files(directory)
                    # else:
                    #     st.error("Please specify a destination folder to save audited transcript!")


                st.subheader("Event Log")
                log_container = st.container()
                with log_container:
                    # Read and display the log content
                    log_content = read_log_file()
                    log_content = log_content.replace('\n', '<br>')

                    # Display the log with custom styling
                    html_content = (
                        "<div style='height:200px; overflow-y:scroll; background-color:#2b2b2b; color:#f8f8f2; "
                        "padding:10px; border-radius:5px; border:1px solid #444;'>"
                        "<pre style='font-family: monospace; font-size: 13px; line-height: 1.5em;'>{}</pre>"
                        "</div>"
                    ).format(log_content)
                    
                    st.markdown(html_content, unsafe_allow_html=True)
                    
                csv_file = 'logfile.csv'
                st.markdown("<br>", unsafe_allow_html=True)
                if os.path.exists(csv_file):
                    with open(csv_file, 'rb') as file:
                        file_contents = file.read()
                        handle_download_log_file(data=file_contents, file_name='log.csv', mime='text/csv', log_message="Action: Event Log Downloaded")
    except Exception as e:
        # st.error(f"An error occurred: {e}")
        create_log_entry(f"Error: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    seed_users()
    print(torch.cuda.is_available())  # Should return True if CUDA is set up

    main()
