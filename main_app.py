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
from datetime import datetime, timedelta
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

# VectorRAG imports:
from IPython.display import display
import ipywidgets as widgets
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai

# GRAPH RAG IMPORTS
# GraphRAG imports 
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
# from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import uuid
from typing import List, Dict, Tuple, Any
import numpy as np
import json
import ast
import openai  # or any LLM wrapper
import random  # for dummy confidence
import spacy
import os
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from llama_index.core import ServiceContext
import matplotlib.pyplot as plt
import heapq
from typing import List, Dict, Any
from difflib import SequenceMatcher
import logging
import json
from openai import OpenAI
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

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
groq_client = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
# deepgram = DeepgramClient(st.secrets["DEEPGRAM_API_KEY"])
client = openai.OpenAI(api_key=OPENAI_API_KEY)

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

    st.subheader("Event Log")

    search_query = st.text_input("Search logs", "")  # Search bar for keyword filtering

    # Date filtering UI
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=7))  # Default is 7 days ago
    with col2:
        end_date = st.date_input("End Date", datetime.now().date())  # Default is today's date

    log_container = st.container()
    with log_container:
        # Read and display the log content based on date filter and search query
        log_content = read_log_file(start_date=start_date, end_date=end_date, search_query=search_query)
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
                create_log_entry(f"ADMIN: {st.session_state.username}, ADDED NEW USER: {new_username}")
            else:
                st.error(message)
                create_log_entry(f"ADMIN: {st.session_state.username}, ACTION FAILED: Unable to add user '{new_username}', Reason: {message}")
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
                                create_log_entry(f"ADMIN: {st.session_state.username}, CHANGED PASSWORD FOR USER: {user['username']}")
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
                            old_role = st.session_state.get("role")
                            create_log_entry(f"ADMIN: {st.session_state.username}, CHANGED ROLE FOR USER: {user['username']} from {old_role} to {new_role}")
                        else:
                            st.error(message)

                    # Row 3: Delete User
                    st.subheader("Delete User")
                    if st.button("Delete User", key=f"btn_del_{user['username']}"):
                        success, message = delete_user(user['username'])
                        if success:
                            st.success(message)
                            create_log_entry(f"ADMIN: {st.session_state.username}, DELETED USER: {user['username']}")
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
            st.session_state["role"] = user.role
            st.rerun()
        else:
            st.error("Invalid username or password!")

def save_audio_file(audio_bytes, name):
    try:
        if not audio_bytes:
            print(f"Received {len(audio_bytes)} bytes for {name}")
            print(f"Error: No data in {name}")
            return None

        if name.lower().endswith(".wav") or name.lower().endswith(".mp3"):
            username = st.session_state.get("username", "default_user")
            user_folder = os.path.join(".", username)
            os.makedirs(user_folder, exist_ok=True)

            # Sanitize filename and construct full path
            name = os.path.basename(name)
            base_name, ext = os.path.splitext(name)  
            short_name = base_name[:20]  # Limit filename length
            file_path = os.path.join(user_folder, f"{short_name}{ext}")

            # 🔄 If file exists, delete it and remove from session state
            if os.path.exists(file_path):
                print(f"Replacing existing file: {file_path}")
                os.remove(file_path)

                # Ensure UI updates by removing old reference
                if name in st.session_state.uploaded_files:
                    del st.session_state.uploaded_files[name]

            # Save the new file
            with open(file_path, "wb") as f:
                f.write(audio_bytes)

            # Ensure file is actually written
            time.sleep(1)  
            full_path = os.path.abspath(file_path)

            if os.path.exists(full_path):  
                print(f"File successfully saved at: {full_path}")

                # ✅ Update session state
                st.session_state.uploaded_files[os.path.basename(file_path)] = full_path

                return full_path  
            else:
                print(f"File not found after writing: {full_path}")
                return None

    except Exception as e:
        print(f"Failed to save file: {e}")
        return None

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
        

# Vector RAG splitting and embedding step
def semantic_chunk_transcript(
    raw_transcript: str,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.90,
    overlap_turns: int = 1,
    min_turns_per_chunk: int = 3,
    max_turns_per_chunk: int = 6,
) -> list[str]:
    print("Loading Chunks")
    
    # Takes in raw transcript, performs semantic chunking using LlamaIndex, embeds using HuggingFace's sentence-transformer, and ensures speaker labels and turns are preserved in mini-dialogue chunks.

    # Parameters:
    #     raw_transcript (str): Full diarized transcript text
    #     embed_model_name (str): HF embedding model for semantic chunking
    #     similarity_threshold (float): Cosine similarity threshold for chunk splits
    #     overlap_turns (int): Number of speaker turns to overlap between chunks
    #     min_turns_per_chunk (int): Minimum number of speaker turns per chunk
    #     max_turns_per_chunk (int): Maximum number of speaker turns per chunk

    # Returns:
    #     List[str]: List of processed dialogue chunks (speaker-aware)
    
    
    def extract_speaker(line: str) -> str:
        """Extracts any speaker label at the start of a line, ending with a colon."""
        match = re.match(r"^([^:]+):", line)
        return match.group(1).strip() if match else ""

    # Step 1: Create a Document object
    document = Document(text=raw_transcript)

    # Step 2: Create embedding model
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # Step 3: Initialize Semantic Chunker
    chunker = SemanticSplitterNodeParser(
        embed_model=embed_model,
        similarity_threshold=similarity_threshold,
    )

    # Step 4: Chunk the document semantically
    nodes = chunker.get_nodes_from_documents([document])
    lines = []
    for node in nodes:
        for line in node.text.strip().split("\n"):
            clean_line = line.strip()
            if clean_line:
                speaker = extract_speaker(clean_line)
                if speaker:
                    last_detected_speaker = speaker
                else:
                    # Assign previous speaker if not detected
                    speaker = last_detected_speaker
                    clean_line = f"{speaker}: {clean_line}"
                lines.append((speaker, clean_line))
                
    # Debug print to verify speaker detection
    for speaker, line in lines:
        print(f"[{speaker}] {line}")

    # Step 5: Group speaker turns into mini-dialog chunks
    chunks = []
    i = 0
    prev_last_speaker = ""
    
    while i < len(lines):
        chunk_lines = []

        # Collect up to max_turns_per_chunk speaker turns
        for j in range(i, min(i + max_turns_per_chunk, len(lines))):
            chunk_lines.append(lines[j])

        # Fix speaker label if chunk starts mid-speaker
        if prev_last_speaker:
            first_speaker, first_line = chunk_lines[0]
            if first_speaker == "" and prev_last_speaker:
                # Prepend the speaker label manually
                chunk_lines[0] = (
                    prev_last_speaker,
                    f"{prev_last_speaker}: {first_line}"
                )

        # Save the last speaker for the next round
        last_speaker = chunk_lines[-1][0]
        prev_last_speaker = last_speaker

        # Add chunk to result
        chunk_text = "\n".join([line for _, line in chunk_lines])
        chunks.append(chunk_text)

        # Move forward but leave overlap
        i += max(min_turns_per_chunk, len(chunk_lines) - overlap_turns)
        
    for speaker, line in lines:
        print(f"[{speaker}] {line}")    

    return chunks

#Vector Storing and Embedding
def store_chunks_as_vector_index(
    chunks: list[str],
    persist_dir: str = "./storage_mini",
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    openai_model: str = "gpt-4o-mini"
):
    
    # Stores dialogue chunks as a vector index using HuggingFace embeddings and OpenAI LLM for RAG.

    # Parameters:
    #     chunks (list[str]): Dialogue chunks returned from semantic_chunk_transcript().
    #     persist_dir (str): Path to store the vector index.
    #     embed_model_name (str): HuggingFace model name for embedding.
    #     openai_model (str): OpenAI model name (e.g., gpt-3.5-turbo).
    

    print("Creating Vector Index...")

    # Convert chunks into LlamaIndex Document objects
    documents = [Document(text=chunk) for chunk in chunks]

    # Load embedding and LLM
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    llm = LlamaOpenAI(model=openai_model)

    # Create node parser (simpler than SemanticSplitter here)
    node_parser = SimpleNodeParser()

    # Build the vector index
    vector_index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        llm=llm,
        node_parser=node_parser,
        show_progress=True,
    )

    # Persist it to disk
    os.makedirs(persist_dir, exist_ok=True)
    vector_index.storage_context.persist(persist_dir=persist_dir) # UNCOMMENT WHEN WANT TO MAKE PERSISTENT

    print(f"Vector store persisted at: {persist_dir}")
    print(f"🧠 Total Chunks Stored: {len(documents)}")
    print("\n--- Stored Chunks Preview ---")
    for i, doc in enumerate(documents):
        preview = doc.text.strip().replace("\n", " ")[:100]
        print(f"Chunk {i+1}: {preview}...")
    
    
def retrieve_relevant_chunks(
    stage_1_criteria: list[str],
    persist_dir: str = "./storage_mini",
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 3,
) -> dict[str, list[tuple[int, float, str]]]:
    """
    Retrieves the top-k most relevant transcript chunks for each Stage 1 criterion.

    Parameters:
        stage_1_criteria (list[str]): List of criteria as questions or prompts.
        persist_dir (str): Directory where the vector index is stored.
        embed_model_name (str): HuggingFace model name.
        top_k (int): Number of top matching chunks to retrieve per criterion.

    Returns:
        dict[str, str]: Mapping from each criterion to a raw text string of the top-k relevant chunks.
    """
    print("Loading vector index for retrieval...")

    # Load the embedding model
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # Load stored index
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    vector_index = load_index_from_storage(storage_context)

    # Load stored documents
    documents = list(vector_index.docstore.docs.values())
    chunk_texts = [doc.text for doc in documents]

    # Embed all stored chunks once
    chunk_embeddings = embed_model.get_text_embedding_batch(chunk_texts)

    results = {}

    for criterion in stage_1_criteria:
        print(f"\n🔍 Retrieving for: {criterion}")
        
        # Embed the criterion
        criterion_embedding = embed_model.get_text_embedding(criterion)

        # Compute cosine similarity
        similarities = cosine_similarity(
            [criterion_embedding], chunk_embeddings
        )[0]  # shape: (num_chunks,)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Get top-k chunk texts
        top_chunks = [chunk_texts[i] for i in top_indices]
        top_chunks_info = [
            (i, float(similarities[i]), chunk_texts[i]) for i in top_indices
        ]
        combined_text = "\n\n".join(top_chunks)

        # Store result
        results[criterion] = top_chunks_info

    return results

def stage_1_criteria_audit(relevant_chunks_dict: dict[str, list[tuple[int, float, str]]], model_engine="gpt-4o-mini"):
    import json
    import re
    from openai import OpenAI

    client = OpenAI()

    # Stage 1 Audit Criteria with bracket context
    criteria_list = [
        "Did the telemarketer introduced themselves by stating their name? (Usually followed by 'calling from')",
        "Did the telemarketer state that they are calling from one of these ['IPP', 'IPPFA', 'IPP Financial Advisors'] without mentioning on behalf of any other insurers?(accept anyone one of the 3 name given)",
        "Did the customer asked how did the telemarketer obtained their contact details? If they asked, did telemarketer mentioned who gave the customer's details to him? (Not Applicable if customer didn't)",
        "Did the telemarketer specify the types of financial services offered?",
        "Did the telemarketer offered to set up a meeting or zoom session with the consultant for the customer? (Try to specify the date and location if possible)",
        "Did the telemarketer stated that products have high returns, guaranteed returns, or capital guarantee? (Fail if they did, Pass if they didn't)",
        "Was the telemarketer polite and professional in their conduct?"
    ]

    output_results = []
    overall_result = "Pass"
    fail_count = 0  # New counter for failed criteria

    for criterion in criteria_list:
        # Get the top-k chunks retrieved for this specific criterion
        top_chunks_info = relevant_chunks_dict.get(criterion, [])
        combined_text = "\n\n".join([chunk[2] for chunk in top_chunks_info])

        print(f"\n--- Auditing Criterion ---\n{criterion}\n")

        prompt = f"""
        You are an auditor for IPP or IPPFA. 
        You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
        The audit evaluates whether the telemarketer adhered to a specific criterion from a predefined list.

        ### Instruction:
        - Review the provided conversation transcript.
        - Assess the telemarketer's compliance **only for the following single criterion**:
            "{criterion}"
        - Quote specific reasons from the conversation to justify the result.
        - Only mark a criterion as "Pass" if you are very confident (i.e., nearly certain) based on clear and specific evidence.
        - If not applicable, you may return "Not Applicable".
        
        ### Input Transcript:
        {combined_text}
        
        ### Response Format (JSON):
        [
            {{
                "Criteria": "{criterion}",
                "Reason": "<Your explanation>",
                "Result": "Pass" or "Fail" or "Not Applicable"
            }}
        ]
        """

        response = client.chat.completions.create(
            model=model_engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        result_text = response.choices[0].message.content
        cleaned = (
            result_text.replace("```json", "")
            .replace("```", "")
            .replace("### Response:", "")
            .strip()
        )

        try:
            result_json = json.loads(cleaned)
        except Exception as e:
            print(f"[!] JSON parsing error for criterion:\n{criterion}")
            print("Raw output:\n", result_text)
            raise e

        output_results.append(result_json[0])

        # Update fail count
        if result_json[0]["Result"] == "Fail":
            fail_count += 1

    # Final result decision based on fail count
    if fail_count >= 2:
        overall_result = "Fail"

    final_output = {
        "Stage 1": output_results,
        "Overall Result": overall_result
    }

    return final_output


def stage_2_criteria_audit(
    stage_1_result: dict,
    relevant_chunks_dict: dict[str, list[tuple[int, float, str]]],
    model_engine="gpt-4o-mini"
):
    import json
    import re
    from openai import OpenAI

    client = OpenAI()

    # Only proceed if Stage 1 passed
    if stage_1_result.get("Overall Result") != "Pass":
        print("❌ Stage 1 did not pass. Skipping Stage 2.")
        return None

    # Stage 2 Audit Criteria
    criteria_list = [
        "Did the telemarketer ask if the customer is keen to explore how they can benefit from IPPFA's services?",
        "Did the customer show uncertain response to the offer of the product and services? If Yes, Check did the telemarketer propose meeting or zoom session with company's consultant?",
        "Did the telemarketer pressure the customer for the following activities (product introduction, setting an appointment)? (Fail if they did, Pass if they didn't)"
    ]

    output_results = []
    overall_result = "Pass"
    fail_count = 0  # Allow up to 1 failure

    for criterion in criteria_list:
        # Get top-k retrieved chunks for the criterion
        top_chunks_info = relevant_chunks_dict.get(criterion, [])
        combined_text = "\n\n".join([chunk[2] for chunk in top_chunks_info])

        print(f"\n--- Auditing Stage 2 Criterion ---\n{criterion}\n")

        prompt = f"""
        You are an auditor for IPP or IPPFA. 
        You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
        The audit evaluates whether the telemarketer adhered to a specific criterion from a predefined list.

        ### Instruction:
        - Review the provided conversation transcript.
        - Assess the telemarketer's compliance **only for the following single criterion**:
            "{criterion}"
        - Quote specific reasons from the conversation to justify the result.
        - Only mark a criterion as "Pass" if you are very confident (i.e., nearly certain) based on clear and specific evidence.
        - If not applicable, you may return "Not Applicable".

        ### Input Transcript:
        {combined_text}

        ### Response Format (JSON):
        [
            {{
                "Criteria": "{criterion}",
                "Reason": "<Your explanation>",
                "Result": "Pass" or "Fail" or "Not Applicable"
            }}
        ]
        """

        response = client.chat.completions.create(
            model=model_engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        result_text = response.choices[0].message.content
        cleaned = (
            result_text.replace("```json", "")
            .replace("```", "")
            .replace("### Response:", "")
            .strip()
        )

        try:
            result_json = json.loads(cleaned)
        except Exception as e:
            print(f"[!] JSON parsing error for criterion:\n{criterion}")
            print("Raw output:\n", result_text)
            raise e

        output_results.append(result_json[0])

        # Stage 2 fails immediately on any "Fail"
        if result_json[0]["Result"] == "Fail":
            fail_count += 1
    
    # Allow up to 1 fail
    if fail_count > 1:
        overall_result = "Fail"

    final_output = {
        "Stage 2": output_results,
        "Overall Result": overall_result
    }

    return final_output



def combine_stage_results(stage_1_result: dict, stage_2_result: dict | None = None) -> dict:
    """
    Combines the results from stage_1_criteria_audit and stage_2_criteria_audit
    to match the structure of the original LLM_audit() output.
    """
    combined_result = {
        "Stage 1": stage_1_result["Stage 1"]
    }

    # If Stage 2 was executed, include it
    if stage_2_result is not None and "Stage 2" in stage_2_result:
        combined_result["Stage 2"] = stage_2_result["Stage 2"]
        combined_result["Overall Result"] = stage_2_result["Overall Result"]
    else:
        combined_result["Overall Result"] = stage_1_result["Overall Result"]

    return combined_result

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

# ==================================GRAPH RAG=================================================
# ==================================GRAPH RAG=================================================
# ==================================GRAPH RAG=================================================

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunker = SemanticSplitterNodeParser(embed_model=embed_model, chunk_size=5)  # adjust chunk size as needed
sentence_splitter = SentenceSplitter()

# Step 1: Sentence-level split
sentence_splitter = SentenceSplitter()
# semantic chunk + sentence splitting 
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

def hybrid_semantic_chunk(text):
    # Step 1: Sentence split using nltk
    sentence_texts = sent_tokenize(text)

    print(f"\n📘 Sentence Split ({len(sentence_texts)} segments):")
    for s in sentence_texts:
        print(f"- {s}")

    # Step 2: Apply semantic chunking to long sentences
    final_chunks = []
    for sentence in sentence_texts:
        if len(sentence.split()) > 20:
            deeper_chunks = chunker.get_nodes_from_documents([TextNode(text=sentence)])
            final_chunks.extend([n.text for n in deeper_chunks])
        else:
            final_chunks.append(sentence)

    if len(final_chunks) > 1:
        print("\n🔹 Original Text:")
        print(text)
        print("🔹 Split Into:")
        for ch in final_chunks:
            print(f"- {ch}")

    return final_chunks

# Semantic chunking ONLY, splitter
def semantic_chunk(text):
    doc = TextNode(text=text)
    nodes = chunker.get_nodes_from_documents([doc])
    chunks = [node.text for node in nodes]

    # DEBUG
    if len(chunks) > 1:
        print("\n🔹 Original Text:")
        print(text)
        print("🔹 Split Into:")
        for ch in chunks:
            print(f"- {ch}")
    
    return chunks

# Main transcript processor with chunking applied
def preprocess_transcript(transcript: str):
    lines = transcript.strip().split("\n")
    structured = []

    speaker_line_pattern = re.compile(r"^(Telemarketer|Customer):\s*(.*)")

    for line in lines:
        match = speaker_line_pattern.match(line)
        if match:
            speaker, text = match.groups()
            chunks = hybrid_semantic_chunk(text.strip()) # hybrid_semantic_chunk(text.strip()) 

            for chunk in chunks:
                structured.append({
                    "speaker": speaker,
                    "text": chunk.strip()
                })

    return structured

def print_preprocessed_transcript(preprocessed):
    print("\n=============== Preprocessed Transcript ===============")
    for i, entry in enumerate(preprocessed, start=1):
        print(f"🔹 Chunk {i}")
        print(f"👤 Speaker: {entry['speaker']}")
        print(f"🗣️ Text: {entry['text']}")
        print("-" * 50)

openai.api_key = OPENAI_API_KEY
llm = OpenAI(api_key=OPENAI_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)
# Optional spaCy fallback
# Extracts root verbs and main objects using dependency tags.
nlp = spacy.load("en_core_web_sm")

# FACT SUMMARY METADATA 
def generate_fact_summary(text: str, llm=False) -> str:
    if llm:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "Extract all the key actions or intentions expressed in the sentence. "
                        "Respond with a concise sentence fragment or list (e.g., 'Introduced self, proposed a meeting', "
                        "'Acknowledged understanding', 'Asked a clarifying question'). "
                        "Avoid general topics — focus only on specific speaker actions or intents."
                    )},
                    {"role": "user", "content": text}
                ],
                max_tokens=50,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ LLM failed for fact summary: {text} — {str(e)}")
            return rule_based_fact_summary(text)
    else:
        return rule_based_fact_summary(text)


def detect_topics(text: str, llm=False) -> List[str]:
    if llm:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Identify high-level topics or themes in the following sentence. Return them as a comma-separated list."},
                    {"role": "user", "content": text}
                ],
                max_tokens=50,
                temperature=0
            )
            topics_str = clean_text(response.choices[0].message.content.strip())
            return [t.strip().lower() for t in topics_str.split(",") if t.strip()]
        except Exception as e:
            print(f"⚠️ LLM failed for topics: {text} — {str(e)}")
            return rule_based_detect_topics(text)
    else:
        return rule_based_detect_topics(text)


def detect_details(text: str, llm=False) -> List[str]:
    if llm:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "List any contextual or delivery-related details in this sentence, such as tone, language clarity, or interaction style. Use short phrases, comma-separated."},
                    {"role": "user", "content": text}
                ],
                max_tokens=60,
                temperature=0
            )
            details_str = clean_text(response.choices[0].message.content.strip())
            return [d.strip() for d in details_str.split(",") if d.strip()]
        except Exception as e:
            print(f"⚠️ LLM failed for details: {text} — {str(e)}")
            return rule_based_detect_details(text)
    else:
        return rule_based_detect_details(text)
    
# RULE-BASED EXTRACTOR FUNCTIONS (Keyword Mappings from Audit Criteria)
def rule_based_fact_summary(text: str) -> str:
    doc = nlp(text)
    actions = set()
    text_lower = text.lower()

    for sent in doc.sents:
        lemmas = [token.lemma_ for token in sent]
        deps = [token.dep_ for token in sent]
        sent_text = sent.text.lower()

        # Introduced self
        if any(t.lemma_ in ["call", "introduce", "name"] and t.dep_ in ["ROOT", "attr"] for t in sent) and ("from" in sent_text or "this is" in sent_text):
            actions.add("Introduced self")

        # Stated company
        if any(org in sent_text for org in ["ipp", "ippfa", "ipp financial advisors"]) and "on behalf" not in sent_text:
            actions.add("Stated company")

        # Asked about contact source
        if re.search(r"(how.*get.*number|who.*gave.*(number|contact))", sent_text) or any("contact" in token.text.lower() for token in sent):
            actions.add("Asked about contact source")

        # Explained services
        if any(l in lemmas for l in ["invest", "plan", "save", "return", "guarantee", "insurance", "cover"]):
            actions.add("Explained services")

        # Proposed meeting
        if any(l in lemmas for l in ["schedule", "arrange", "book", "meet", "zoom", "set"]):
            if "meeting" in sent_text or "zoom" in sent_text:
                actions.add("Proposed meeting")
                
        # Proposed meeting (expanded)
        if re.search(r"(join.*(zoom|call|meeting)|hop on a call|talk to consultant|schedule a session)", sent_text):
            actions.add("Proposed meeting")

        # Mentioned guaranteed returns
        if "high return" in sent_text or "guarantee" in sent_text or "capital guarantee" in sent_text:
            actions.add("Mentioned guaranteed returns")

        # Professional tone
        if any(p in sent_text for p in ["thank you", "no worries", "sure", "alright", "okay", "i see", "that’s ok", "understood"]):
            actions.add("Professional tone")

        # Checked interest
        if "keen to explore" in sent_text or re.search(r"are you (still )?(interested|keen|open)", sent_text):
            actions.add("Checked interest")

        # Customer uncertainty
        if re.search(r"not interested|maybe later|not now|not sure|unsure|don’t think so", sent_text):
            actions.add("Customer uncertainty")

        # Pressured appointment
        if "not interested" in sent_text and re.search(r"(set.*meeting|zoom|schedule)", sent_text):
            actions.add("Pressured appointment")
        
        # Add detection for "Who gave your number" responses
        if re.search(r"(my (manager|consultant|colleague).*(gave|shared).*(your|the).*contact)", sent_text):
            actions.add("Explained contact source")
        
        # Explained contact source
        if re.search(r"(my (manager|consultant|colleague).*(gave|shared).*(your|the).*contact)", sent_text):
            actions.add("Explained contact source")

    return ", ".join(sorted(actions)) or "No key actions found."


def rule_based_detect_topics(text: str) -> List[str]:
    doc = nlp(text)
    topics = set()
    text_lower = text.lower()

    for sent in doc.sents:
        lemmas = [token.lemma_ for token in sent]

        if any(l in lemmas for l in ["introduce", "call", "name"]) and "from" in sent.text.lower():
            topics.add("introduction")
        if any(org in sent.text.lower() for org in ["ipp", "ippfa", "financial advisor"]):
            topics.add("company disclosure")
        if "how did you get" in sent.text.lower() or "who gave" in sent.text.lower():
            topics.add("lead source")
        if any(l in lemmas for l in ["invest", "return", "insurance", "plan", "cover"]):
            topics.add("financial services")
        if "meeting" in sent.text.lower() or "zoom" in sent.text.lower():
            topics.add("meeting setup")
        if "high return" in sent.text.lower() or "guarantee" in sent.text.lower():
            topics.add("product claim")
        if "not interested" in sent.text.lower() or "maybe later" in sent.text.lower() or "unsure" in sent.text.lower():
            topics.add("customer uncertainty")
        if "keen to explore" in sent.text.lower() or "would you be interested" in sent.text.lower():
            topics.add("interest probe")
        if "not interested" in sent.text.lower() and "meeting" in sent.text.lower():
            topics.add("pressure tactics")

    return sorted(topics) or ["general"]


def rule_based_detect_details(text: str) -> List[str]:
    doc = nlp(text)
    details = set()
    text_lower = text.lower()

    if len(text.split()) < 8:
        details.add("Concise")

    if "?" in text:
        details.add("Asked a question")

    polite_phrases = ["thank you", "no worries", "sure", "ok", "okay", "alright"]
    if any(p in text_lower for p in polite_phrases):
        details.add("Friendly tone")
        
    polite_expressions = ["thank you", "thanks", "appreciate it", "sure", "ok", "no worries", "alright", "understood"]
    if any(phrase in text_lower for phrase in polite_expressions):
        details.add("Friendly tone")

    if re.search(r"not interested|maybe later|not now|unsure|don’t think so", text_lower):
        details.add("Customer hesitance")

    # Tone detection via POS tags or dependency (very basic)
    if any(token.lemma_ == "apologize" or token.text.lower() == "sorry" for token in doc):
        details.add("Apologetic")

    return sorted(details) or ["No details found"]

# Full enrichment step ( FACT SUMMARY + DETAIL + TOPIC METDATA )
def enrich_chunks(chunks: List[Dict], llmTF = False) -> List[Dict]:
    enriched = []

    for i, chunk in enumerate(chunks, start=1):
        node_id = f"n{i}"
        text = chunk["text"]
        speaker = chunk["speaker"]

        print(f"\n🔄 Processing Chunk {node_id} ({speaker}):")
        print(f"💬 Text: {text}")

        print("   ➤ Generating 📌 Fact Summary...")
        fact_summary = generate_fact_summary(text, llm=llmTF)

        print("   ➤ Generating 🧩 Details...")
        details = detect_details(text, llm=llmTF)

        print("   ➤ Generating 🏷️  Topics...")
        topics = detect_topics(text, llm=llmTF)
        
        #Compute confidence using cosine similarity
        try:
            text_embedding = embed_model.get_text_embedding(text)
            summary_embedding = embed_model.get_text_embedding(fact_summary)
            similarity = cosine_similarity(
                np.array(text_embedding).reshape(1, -1),
                np.array(summary_embedding).reshape(1, -1)
            )[0][0]
            confidence_score = round(float(similarity), 3)
        except Exception as e:
            print(f"⚠️ Failed to compute confidence: {e}")
            confidence_score = 0.0

        enriched.append({
            "id": node_id,
            "speaker": speaker,
            "original_text": text,
            "fact_summary": fact_summary,
            "details": details,
            "topics": topics,
            "confidence": confidence_score
        })

    return enriched

def clean_text(text: str) -> str:
    # Fix broken words caused by newlines or wrap issues
    # e.g. "familia\n rity" → "familiarity", but keep real word spacing
    text = re.sub(r'(\w+)[\n\r]\s+(\w+)', r'\1\2', text)  # Join words split by newline + indent
    text = re.sub(r'(\w+)\s{2,}(\w+)', r'\1 \2', text)     # Normalize accidental double spaces
    return text.strip()

def print_enriched_nodes(enriched_nodes):
    for node in enriched_nodes:
        print(f"🟦 ID: {node['id']}")
        print(f"👤 Speaker: {node['speaker']}")
        print(f"🗣️ Original Text: {node['original_text']}")
        print(f"📌 Fact Summary: {node['fact_summary']}")
        print(f"🧩 Details: {', '.join(node['details'])}")
        print(f"🏷️ Topics: {', '.join(node['topics'])}")
        print(f"🎯 Confidence: {node['confidence']}")
        print("-" * 60)

# ------------------------------Node & Edge Construction--------------------------------

# CONSTRUCT NODES
def construct_nodes(enriched_nodes, embed_model):
    print("\n=============== Constructing Nodes ===============")
    chunk_texts = [node["original_text"] for node in enriched_nodes]
    embeddings = embed_model.get_text_embedding_batch(chunk_texts)
    nodes = []
    for node in enriched_nodes:
        node_data = {
            "id": node["id"],
            "speaker": node["speaker"],
            "text": node["original_text"],
            "fact_summary": node["fact_summary"],
            "topics": node["topics"],
            "details": node["details"],
            "confidence": node["confidence"]
        }
        nodes.append(node_data)
        print(f"\n🟢 Node: {node_data['id']}")
        print(f"👤 Speaker: {node_data['speaker']}")
        print(f"🗣️ Text: {node_data['text']}")
        print(f"📌 Fact Summary: {node_data['fact_summary']}")
        print(f"🏷️ Topics: {node_data['topics']}")
        print(f"🧩 Details: {node_data['details']}")
        print(f"🎯 Confidence: {node_data['confidence']}")
        print("-" * 60)
    return nodes, embeddings

def construct_continuation_edges(enriched_nodes):
    print("\n=============== Constructing Edges ===============")
    edges = []
    for i in range(len(enriched_nodes) - 1):
        curr = enriched_nodes[i]
        next_node = enriched_nodes[i + 1]
        if curr["speaker"] == next_node["speaker"]:
            edge_cont = { "from": curr["id"], "to": next_node["id"], "type": "continuation" }
            edges.append(edge_cont)
            print(f"⏩ Edge (Continuation): {curr['id']} → {next_node['id']} (Same speaker)")
    return edges

# Question & Answer Edges
def is_question(text):
    return "?" in text.strip()

def construct_qa_edges(nodes):
    print("\n=============== Question & Answer Edges ===============")
    edges = []
    for i, curr in enumerate(nodes):
        curr_id = curr["id"]
        curr_text = curr["text"]
        if is_question(curr_text):
            print(f"❓ Question Detected: {curr_id} | {curr_text}")
            for j in range(i + 1, len(nodes)):
                responder = nodes[j]
                responder_id = responder["id"]
                if responder_id != curr_id:
                    question_edge = {
                        "from": curr_id,
                        "to": responder_id,
                        "type": "question"
                    }
                    answer_edge = {
                        "from": responder_id,
                        "to": curr_id,
                        "type": "answer"
                    }
                    edges.extend([question_edge, answer_edge])
                    print(f"🧭 Question Edge: {question_edge['from']} → {question_edge['to']}")
                    print(f"💬 Answer Edge: {answer_edge['from']} → {answer_edge['to']}")
                    break  # Only first match used
    return edges

def construct_related_topic_edges(nodes, embeddings, min_similarity=0.50):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    print("\n=============== Pairwise Semantic Similarity Edges ===============")
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            sim = cosine_similarity(
                np.array(embeddings[i]).reshape(1, -1),
                np.array(embeddings[j]).reshape(1, -1)
            )[0][0]

            shared = set(nodes[i]["topics"]) & set(nodes[j]["topics"])
            if sim >= min_similarity:
                edge = {
                    "from": nodes[i]["id"],
                    "to": nodes[j]["id"],
                    "type": "related_topic",
                    "similarity": round(float(sim), 4),
                    "shared_topics": list(shared)
                }
                edges.append(edge)
                print(f"🔗 Edge (Related Topic): {edge['from']} → {edge['to']} | Similarity: {edge['similarity']}")
    return edges
            
def construct_graph(enriched_nodes, embed_model, min_similarity=0.50):
    nodes, embeddings = construct_nodes(enriched_nodes, embed_model)
    edges = []
    edges += construct_continuation_edges(enriched_nodes)
    edges += construct_qa_edges(nodes)
    edges += construct_related_topic_edges(nodes, embeddings, min_similarity=min_similarity)

    print("\n=============== Final Edge List ===============")
    for edge in edges:
        print(edge)
    print("\n=============== Final Node List ===============")
    for node in nodes:
        print(node)

    return nodes, edges
    
#---------------GRAPH STORAGE, POPULATE GRAPH WITH DATA--------------
    
def build_graph(nodes, edges):
    """
    Creates a directed graph from node and edge dictionaries.
    """
    G = nx.DiGraph()

    # Add all nodes with metadata
    for node in nodes:
        G.add_node(node["id"], **node)

    # Add edges with attributes
    for edge in edges:
        G.add_edge(edge["from"], edge["to"], **{k: v for k, v in edge.items() if k not in ["from", "to"]})

    return G

def draw_graph(G, layout=None, with_labels=True):
    """
    Optional visualization helper using matplotlib.
    """

    if layout is None:
        layout = nx.shell_layout(G)

    edge_labels = nx.get_edge_attributes(G, 'type')
    node_labels = {node: node for node in G.nodes()} # Shows node IDs like n1, n2, etc. instead of speakers
    
    # Assign colors based on speaker
    colors = []
    for node in G.nodes(data=True):
        speaker = node[1].get("speaker", "")
        if speaker == "Telemarketer":
            colors.append("skyblue")
        elif speaker == "Customer":
            colors.append("lightgreen")
        else:
            colors.append("gray")

    nx.draw(G, layout, with_labels=with_labels, labels=node_labels, node_color=colors, node_size=1500, font_size=10, edge_color='gray')
    nx.draw_networkx_edge_labels(G, layout, edge_labels=edge_labels, font_color='red')

    plt.title("Conversation Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#-------------------Subgraph Retrieval for each audit criteria------------------- 
def retrieve_subgraph_for_criterion_fixed(
    criterion: str,
    nodes: list,
    edges: list,
    embed_model,
    min_score: float = 0.3,
    top_k: int = 3
) -> dict:
    print(f"\n🔍 Retrieving Subgraph for Criterion: {criterion}")
    criterion_embedding = embed_model.get_text_embedding(criterion)
    scored_nodes = []

    for node in nodes:
        scores_sources = []

        try:
            summary_score = cosine_similarity(
                [criterion_embedding],
                [embed_model.get_text_embedding(node.get("fact_summary", ""))]
            )[0][0]
            scores_sources.append((summary_score, "fact_summary"))
        except:
            pass

        try:
            text_score = cosine_similarity(
                [criterion_embedding],
                [embed_model.get_text_embedding(node.get("text", ""))]
            )[0][0]
            scores_sources.append((text_score, "text"))
        except:
            pass

        for topic in node.get("topics", []):
            try:
                topic_score = cosine_similarity(
                    [criterion_embedding],
                    [embed_model.get_text_embedding(topic)]
                )[0][0]
                scores_sources.append((topic_score, f"topic: {topic}"))
            except:
                continue

        for detail in node.get("details", []):
            try:
                detail_score = cosine_similarity(
                    [criterion_embedding],
                    [embed_model.get_text_embedding(detail)]
                )[0][0]
                scores_sources.append((detail_score, f"detail: {detail}"))
            except:
                continue

        if scores_sources:
            max_score, match_source = max(scores_sources, key=lambda x: x[0])
            if max_score >= min_score:
                scored_nodes.append((node, max_score, match_source))

    def get_node_text(nid):
        return next((n["text"] for n in nodes if n["id"] == nid), "[Text not found]")

    # Sort and get top nodes
    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [item[0] for item in scored_nodes[:top_k]]
    top_node_ids = set(n["id"] for n in top_nodes)

    # Collect extended nodes and edges
    all_included_node_ids = set(top_node_ids)
    final_edges = []

    for node in top_nodes:
        node_id = node["id"]

        # CONTINUATION edge
        continuation_edge = next(
            (e for e in edges if e["from"] == node_id and e["type"] == "continuation"), None)
        if continuation_edge:
            cont_node_id = continuation_edge["to"]
            all_included_node_ids.add(cont_node_id)
            final_edges.append(continuation_edge)

        # QUESTION/ANSWER edges
        # Q/A logic: prioritize question → answer or answer → question pairing
        qa_edge = None
        qa_connected_id = None

        # Check if this node is asking a question
        qa_edge = next((e for e in edges if e["type"] == "question" and e["from"] == node_id), None)
        if qa_edge:
            qa_connected_id = qa_edge["to"]
        else:
            # Check if this node is answering a question
            qa_edge = next((e for e in edges if e["type"] == "answer" and e["to"] == node_id), None)
            if qa_edge:
                qa_connected_id = qa_edge["from"]

        if qa_edge and qa_connected_id:
            all_included_node_ids.add(qa_connected_id)
            final_edges.append(qa_edge)
            qa_text = get_node_text(qa_connected_id)
        else:
            # RELATED TOPIC fallback
            related_edges = [
                e for e in edges
                if e["type"] == "related_topic" and e["from"] == node_id
            ]
            if related_edges:
                best_related = max(related_edges, key=lambda e: e.get("similarity", 0))
                all_included_node_ids.add(best_related["to"])
                final_edges.append(best_related)

    # Filter nodes to return only those involved
    final_nodes = [node for node in nodes if node["id"] in all_included_node_ids]

    # Logging
    for node, score, source in scored_nodes[:top_k]:
        print(f"\n🟩 Node {node['id']} (Max Score: {round(score, 3)})")
        print(f"📌 Best match from: {source}")
        print(f"👤 Speaker: {node['speaker']}")
        print(f"🗣️ Text: {node['text']}")
        print(f"📌 Summary: {node['fact_summary']}")
        print(f"🏷️ Topics: {node['topics']}")
        print(f"🧩 Details: {node['details']}")
        print(f"🎯 Confidence: {node['confidence']}")

        # Connected nodes log
        node_id = node["id"]
        print(f"🟦 Connected nodes for {node_id}:")
        
        cont_edge = next((e for e in edges if e["from"] == node_id and e["type"] == "continuation"), None)

        # Continuation
        if cont_edge:
            cont_text = get_node_text(cont_edge["to"])
            print(f"   - Continuation ➡️ {cont_edge['to']}: {cont_text}")

        # Q/A (recalculate within log scope)
        qa_edge_log = next((e for e in edges if e["type"] == "question" and e["from"] == node_id), None)
        if qa_edge_log:
            qa_connected_id = qa_edge_log["to"]
            qa_text = get_node_text(qa_connected_id)
            print(f"   - Q/A ➡️ {qa_connected_id}: {qa_text}")
        else:
            qa_edge_log = next((e for e in edges if e["type"] == "answer" and e["to"] == node_id), None)
            if qa_edge_log:
                qa_connected_id = qa_edge_log["from"]
                qa_text = get_node_text(qa_connected_id)
                print(f"   - Q/A ➡️ {qa_connected_id}: {qa_text}")
            else:
                # fallback
                related_edges = [
                    e for e in edges if e["type"] == "related_topic" and e["from"] == node_id
                ]
                if related_edges:
                    best_related = max(related_edges, key=lambda e: e.get("similarity", 0))
                    rel_text = get_node_text(best_related["to"])
                    print(f"   - Related Topic ➡️ {best_related['to']} (similarity: {round(best_related['similarity'], 3)}): {rel_text}")

    return {
        "criterion": criterion,
        "top_nodes": final_nodes,
        "edges": final_edges,
        "log": [(n, float(score), source) for (n, score, source) in scored_nodes[:top_k]]
    }
  


#-------------------COMBINING OUTPUTS OF VECTORRAG AND GRAPHRAG------------------- 

# Function to combine and print results for each criterion
def combine_rag_outputs(vector_rag: Dict[str, List[Tuple[int, float, str]]],
                        graph_rag: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("------------------COMBINING OUTPUTS OF VECTORRAG AND GRAPHRAG----------------")
    combined_output = []
    
    for entry in graph_rag:
        criterion = normalize_criterion(entry["criterion"])
        graph_nodes = entry["top_nodes"]
        vector_chunks = vector_rag.get(criterion, [])
        
        combined_output.append({
            "criterion": criterion,
            "vector_chunks": [
                {
                    "chunk_id": int(chunk[0]),
                    "score": float(chunk[1]),
                    "text": chunk[2]
                } for chunk in vector_chunks
            ],
            "graph_nodes": [
                {
                    "node_id": node["id"],
                    "speaker": node["speaker"],
                    "text": node["text"],
                    "confidence": node["confidence"]
                } for node in graph_nodes
            ]
        })
    
    return combined_output
        


def print_combined_results(results):
    print("=" * 25 + " COMBINED RESULTS " + "=" * 25)
    
    for idx, result in enumerate(results):
        print(f"\n[{idx+1}] Criterion:")
        print(f"  {result['criterion']}\n")
        
        # Print Vector Chunks
        vector_chunks = result.get("vector_chunks", [])
        if vector_chunks:
            print("  Vector Chunks:")
            for chunk in vector_chunks:
                print(f"    - Chunk ID: {chunk['chunk_id']}")
                print(f"      Score: {chunk['score']:.3f}")
                print(f"      Text: {chunk['text'].strip()[:200]}{'...' if len(chunk['text']) > 200 else ''}\n")
        else:
            print("  Vector Chunks: None\n")

        # Print Graph Nodes
        graph_nodes = result.get("graph_nodes", [])
        if graph_nodes:
            print("  Graph Nodes:")
            for node in graph_nodes:
                print(f"    - Node ID: {node['node_id']}")
                print(f"      Speaker: {node['speaker']}")
                print(f"      Confidence: {node['confidence']:.3f}")
                print(f"      Text: {node['text'].strip()[:200]}{'...' if len(node['text']) > 200 else ''}\n")
        else:
            print("  Graph Nodes: None\n")
    
    print("=" * 70)


# ------------------------------AUDITING RESULTS----------------------------------

def combined_audit(combined_result: list[dict], model_engine="gpt-4o-mini", stage=1):
    print("==================AUDITING RESULTS:===============\n")
    client = llm
    output_results = []
    overall_result = "Pass"
    fail_count = 0

    for entry in combined_result:
        criterion = entry["criterion"]
        vector_chunks = entry.get("vector_chunks", [])
        graph_nodes = entry.get("graph_nodes", [])

        # Combine all retrieved text into one transcript
        vector_text = "\n\n".join(chunk["text"] for chunk in vector_chunks)
        graph_text = "\n\n".join(node["text"] for node in graph_nodes)
        combined_text = (vector_text + "\n\n" + graph_text).strip()

        print(f"\n--- Auditing Criterion ---\n{criterion}\n")

        prompt = f"""
        You are an auditor for IPP or IPPFA. 
        You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
        The audit evaluates whether the telemarketer adhered to a specific criterion from a predefined list.

        ### Instruction:
        - Review the provided conversation transcript.
        - Assess the telemarketer's compliance **only for the following single criterion**:
            "{criterion}"
        - Quote specific reasons from the conversation to justify the result.
        - Only mark a criterion as "Pass" if you are very confident (i.e., nearly certain) based on clear and specific evidence.
        - If not applicable, you may return "Not Applicable".

        ### Input Transcript:
        {combined_text}

        ### Response Format (JSON):
        [
            {{
                "Criteria": "{criterion}",
                "Reason": "<Your explanation>",
                "Result": "Pass" or "Fail" or "Not Applicable"
            }}
        ]
        """

        response = client.chat.completions.create(
            model=model_engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        result_text = response.choices[0].message.content
        cleaned = (
            result_text.replace("```json", "")
            .replace("```", "")
            .replace("### Response:", "")
            .strip()
        )

        try:
            result_json = json.loads(cleaned)
        except Exception as e:
            print(f"[!] JSON parsing error for criterion:\n{criterion}")
            print("Raw output:\n", result_text)
            raise e

        output_results.append(result_json[0])
        if result_json[0]["Result"] == "Fail":
            fail_count += 1

    # Overall decision
    if (stage == 1 and fail_count > 2) or (stage == 2 and fail_count > 1):
        overall_result = "Fail"

    final_output = {
        f"Stage {stage}": output_results,
        "Overall Result": overall_result
    }

    return final_output

# ==================================END OF GRAPHRAG===========================================
# ==================================END OF GRAPHRAG===========================================
# ==================================END OF GRAPHRAG===========================================

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


def read_log_file(log_file='logfile.txt', start_date=None, end_date=None, search_query=""):
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            log_content = file.readlines()
        
        # Reverse log entries (newest first)
        log_content = log_content[::-1]
        filtered_logs = []

        for log_entry in log_content:
            log_entry = log_entry.strip()

            # Filter by search query first
            if search_query.lower() not in log_entry.lower():
                continue

            log_date = None
            # Extract date from log assuming format: "YYYY-MM-DD HH:MM:SS - Message"
            parts = log_entry.split(" - ")
            if len(parts) > 1:
                try:
                    log_date = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S").date()
                except ValueError:
                    pass  # Ignore lines that don’t match the expected format

            # If dates are not selected (i.e., default dates), show all logs without filtering by date
            if start_date and end_date:
                if log_date and not (start_date <= log_date <= end_date):
                    continue  # Skip logs that are outside the selected date range

            # Apply color coding
            if "ADMIN" in log_entry:  
                log_entry = f"<span style='color: yellow;'>{log_entry}</span>\n"  # 🟡 Admin Log
            
            elif "ERROR" in log_entry or "NOT" in log_entry or "FAIL" in log_entry:  
                log_entry = f"<span style='color: red;'>{log_entry}</span>\n"  # 🔴 Error Log
            
            else:  
                log_entry = f"<span style='color: white;'>{log_entry}</span>\n"  # ⚪ Default Log

            filtered_logs.append(log_entry)

        return ''.join(filtered_logs) if filtered_logs else "NO MATCHING LOG ENTRIES FOUND."
    else:
        return "LOG FILE DOES NOT EXIST."
    
    
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
        print(f"File still not found after retries: {file_path}")
        return False

    try:
        audio = MP3(file_path)
        if audio.info.length <= 0:  
            print("Invalid MP3: length is zero.")
            return False
        
        print(f"Valid MP3 with duration: {audio.info.length} seconds")
        return True
    except HeaderNotFoundError:
        print(f"{file_path} has no valid MP3 headers")
        return False
    except Exception as e:
        print(f"Invalid MP3 file: {e}")
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
                    # Check for duplicate filenames and show warn if have multiple duplicates
                    if uploaded_files:
                        filenames = [file.name for file in uploaded_files]
                        duplicates = [name for name in set(filenames) if filenames.count(name) > 1]
                        if duplicates:
                            st.warning(f"⚠️ Duplicate file(s) detected: {', '.join(duplicates)}. Only the most recently uploaded version of each will be used.")
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
                                        create_log_entry(f"USER: {st.session_state.username}, DELETED FILE: {full_path}")
                                    except Exception as e:
                                        st.error(f"Error deleting file {file_name}: {e}")
                                        create_log_entry(f"ERROR DELETING FILE: {file_name} - {e}")
                        else:
                            create_log_entry(f"FILE TO DELETE IS NOT DETECTED: {file_name}")
                        
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
                                    
                                    create_log_entry(f"USER: {st.session_state.username}, DELETED FILE: {full_path}")
                                    del st.session_state.uploaded_files[file_name]

                                except Exception as e:
                                    st.error(f"Error deleting file {file_name}: {e}")
                                    create_log_entry(f"ERROR DELETING FILE {file_name}: {e}")                    

                        if uploaded_files:
                            for file in uploaded_files:
                                try:
                                    audio_content = file.read()
                                    saved_path = save_audio_file(audio_content, file.name)

                                    if saved_path and os.path.exists(saved_path):  
                                        saved_path = os.path.abspath(saved_path)  # Ensure absolute path
                                        print(f"File saved at: {saved_path}")
                                        
                                        if is_valid_mp3(saved_path):
                                            if saved_path not in st.session_state.audio_files:  # Prevent duplication
                                                st.session_state.audio_files.append(saved_path)
                                                print(f"Added to session state: {saved_path}")
                                        else:
                                            st.error(f"{saved_path} is an Invalid MP3 or WAV File")
                                    else:
                                        st.error("Failed to save uploaded file.")
                                except Exception as e:
                                    st.error(f"Error loading audio file: {e}")
                                    print(f"Error loading audio file: {e}")

                        for file_name in added_files:
                            create_log_entry(f"USER: {st.session_state.username}, UPLOADED FILE: {file_name}")
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
                                        create_log_entry(f"FILE WAS RETURNED: {saved_path}")
                                        if os.path.exists(saved_path):
                                            create_log_entry(f"FILE SUCCESSFULLY EXISTS: {saved_path}")
                                        else:
                                            create_log_entry(f"FILE DOES NOT EXIST EVEN THOUGH IT WAS RETURNED: {saved_path}")
                                    else:
                                        create_log_entry(f"save_audio_file() returned None")
                                    st.error("Failed to save the uploaded file.")
                            except Exception as e:
                                st.error(f"Error loading audio file: {e}")
                                create_log_entry(f"ERROR LOADING AUDIO FILE: {e}")  

                        # Ensure uploaded_files session state exists
                        if "uploaded_files" not in st.session_state:
                            st.session_state.uploaded_files = {}

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

                        # 🔍 Ensure session state tracks only the latest files
                        username = st.session_state.get("username", "default_user")
                        user_folder = os.path.join(".", username)
                        current_files = set(os.listdir(user_folder))  # Get the latest files in the folder

                        # Remove files that no longer exist in the folder
                        removed_files = [f for f in list(st.session_state.uploaded_files.keys()) if f not in current_files]
                        if removed_files:  # Only proceed if there are files to remove
                            for file_name in removed_files:
                                print(f"Removing file from session state: {file_name}")
                                del st.session_state.uploaded_files[file_name]
                            
                            st.session_state.file_change_detected = True

                        # 🔄 Update session state with the latest file list
                        if st.session_state.get("file_change_detected", False):
                            st.session_state.uploaded_files = {f: os.path.join(user_folder, f) for f in current_files}
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
                                create_log_entry(f"USER: {st.session_state.username}, UPLOADED FOLDER: {selected_folder_path}")

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
                                create_log_entry(f"USER: {st.session_state.username}, REMOVED UPLOADED FOLDER")
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
                                        create_log_entry(f"ERROR: {saved_file_path[2:]} IS AN INVALID MP3 OR WAV FILE")


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
                    create_log_entry("SERVICE REQUEST: FAIL (NO FILES UPLOADED)")
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
                                        full_transcript = text
                                        if audit_option == "OpenAI (Recommended)":
                                            
                                            # result = LLM_audit(text) old version
                                            
                                            # ----------------------VectorRAG--------------------------
                                            # Splitting
                                            chunks = semantic_chunk_transcript(text)
                                            for i, chunk in enumerate(chunks):
                                                print(f"\n--- Chunk {i+1} ---\n{chunk}")
                                            
                                            # Embedding and Vector Storing
                                            store_chunks_as_vector_index(chunks)
                                            
                                            # Retrieval 
                                            # For retrieval, not prompting
                                            stage_1_criteria = [
                                                "Did the telemarketer introduced themselves by stating their name? (Usually followed by 'calling from')",
                                                "Did the telemarketer state that they are calling from one of these ['IPP', 'IPPFA', 'IPP Financial Advisors'] without mentioning on behalf of any other insurers?(accept anyone one of the 3 name given)",
                                                "Did the customer asked how did the telemarketer obtained their contact details? If they asked, did telemarketer mentioned who gave the customer's details to him? (Not Applicable if customer didn't)",
                                                "Did the telemarketer specify the types of financial services offered?",
                                                "Did the telemarketer offered to set up a meeting or zoom session with the consultant for the customer? (Try to specify the date and location if possible)",
                                                "Did the telemarketer stated that products have high returns, guaranteed returns, or capital guarantee? (Fail if they did, Pass if they didn't)",
                                                "Was the telemarketer polite and professional in their conduct?"
                                            ]
                                            retrieved_chunks_1 = retrieve_relevant_chunks(stage_1_criteria)

                                            # Print results
                                            for i, (criterion, chunks_info) in enumerate(retrieved_chunks_1.items(), 1):
                                                print(f"\n--- Stage 1 Criterion {i} ---")
                                                print(f"{criterion}\n")
                                                for chunk_index, score, text in chunks_info:
                                                    print(f"[Chunk {chunk_index} | Score: {score:.4f}]\n{text}\n")
                                            
                                            stage_1_result = stage_1_criteria_audit(retrieved_chunks_1)
                                            print(json.dumps(stage_1_result, indent=2))
                                            
                                            stage_2_result = False # initialzie before just in case stage 1 fails and stage 2 doesn't execute at all
                                            # After Stage 1 has passed
                                            if stage_1_result["Overall Result"] == "Pass":
                                                stage_2_criteria = [
                                                    "Did the telemarketer ask if the customer is keen to explore how they can benefit from IPPFA's services?",
                                                    "Did the customer show uncertain response to the offer of the product and services? If Yes, Check did the telemarketer propose meeting or zoom session with company's consultant?",
                                                    "Did the telemarketer pressure the customer for the following activities (product introduction, setting an appointment)? (Fail if they did, Pass if they didn't)"
                                                ]
                                                retrieved_chunks_2 = retrieve_relevant_chunks(stage_2_criteria)
                                                # Print results
                                                for i, (criterion, chunks_info) in enumerate(retrieved_chunks_2.items(), 1):
                                                    print(f"\n--- Stage 2 Criterion {i} ---")
                                                    print(f"{criterion}\n")
                                                    for chunk_index, score, text in chunks_info:
                                                        print(f"[Chunk {chunk_index} | Score: {score:.4f}]\n{text}\n")
                                                        
                                                stage_2_result = stage_2_criteria_audit(stage_1_result, retrieved_chunks_2)
                                                print(json.dumps(stage_2_result, indent=2))
                                                
                                            result = combine_stage_results(stage_1_result, stage_2_result)
                                            
                                            #-------------------------------------------------------------------------
                                            
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
                                error_message = f"ERROR PROCESSING FILE: {audio_file} - {e}"
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
                                st.write(full_transcript)
                                all_text[audio_file[2:]] = full_transcript
                                # print(all_text)
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
                        create_log_entry(f"SUCCESSFULLY AUDITED: {audio_file[2:]}")
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


    except Exception as e:
        # st.error(f"An error occurred: {e}")
        create_log_entry(f"ERROR: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    seed_users()
    print(torch.cuda.is_available())  # Should return True if CUDA is set up

    main()
