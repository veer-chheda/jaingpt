import streamlit as st
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from embeddings import get_embedding_function
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
from google.cloud import translate_v2 as translate
import re
from google.cloud import texttospeech
from dotenv import load_dotenv
import os

PROJECT_ID = os.getenv('PROJECT_ID')
CHROMA_PATH = "./chroma"
PROMPT_TEMPLATE = """
You are a helpful teacher that teaches Jainism religion by answering questions.
Answer the following question on Jainism based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

GUJARATI = """You are a text formatter. Please format the following text into a beautiful format:
{text}
"""

GEMINI_API_URL = f"https://asia-south1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/asia-south1/predict" # Ensure you replace this with your actual API key

# Initialize Google Cloud Translate client
translate_client = translate.Client()
text_to_speech_client = texttospeech.TextToSpeechClient()

# Streamlit UI
st.title("RAG Query Application")
query_text = st.text_input("Enter your query:")

# Flag to indicate whether translation has been performed
translation_done = False

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    response_text = generate_response(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response

def generate_response(prompt: str) -> str:
    # Initialize Vertex AI
    vertexai.init(project="jaingpt-425513", location="asia-south1")

    # Initialize GenerativeModel
    model = GenerativeModel(model_name="gemini-1.0-pro-002")
    chat = model.start_chat(response_validation=False)

    # Get response from GenerativeModel
    response_text = get_chat_response(chat, prompt)

    return response_text

def get_chat_response(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        if chunk.text:
            text_response.append(chunk.text)
    return "".join(text_response)

def translate_text(text: str, target_language: str = "gu"):
    translation = translate_client.translate(text, target_language=target_language)
    return translation["translatedText"]

def text_to_speech(text, language_code='en-US'):
    if not text:
        return None
    
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Select the voice parameters
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code
    )

    # Select the audio file type
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech conversion
    response = text_to_speech_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response.audio_content

if st.button("Answer in English") and query_text:

    # Get response from GenerativeModel
    response_text = generate_response(query_text)

    # Display the response
    st.write(f"Response: {response_text}")

audio_text=''

if st.button("Answer in Gujarati") and query_text:
    if query_text:
        result = generate_response(query_text)
        translated_response = translate_text(result, target_language="gu")
        st.write("Translated Response:")
        text_parts = re.split(r'(?<!\*)\*(?!\*)', translated_response)
        formatted_text = "\n".join(text_parts)
        text_parts = re.split(r'(?<!\d)\d+\.(?!\d)', translated_response)
        formatted_text = "\n".join(text_parts)
        text_parts = re.split(r'\*\*', formatted_text)
        formatted_text = "\n**".join(text_parts)
        # Join the parts with a newline character
        formatted_text = "\n".join(text_parts)
        text_parts = re.split(r'\*',formatted_text)
        formatted_text = "\n".join(text_parts)
        st.write(formatted_text)
        formatted_text = re.sub(r'#', '', formatted_text)
        audio_content = text_to_speech(formatted_text, language_code='gu-IN')
        st.audio(audio_content, format='audio/mp3')
    audio_text = translated_response

if st.button('Speak...'):
            st.write(audio_text)
