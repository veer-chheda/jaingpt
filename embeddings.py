from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import aiplatform
from dotenv import load_dotenv
import os


def get_embedding_function():
    PROJECT_ID = os.getenv('PROJECT_ID')
    # Initialize the AI Platform with your project and location
    aiplatform.init(project=f'{PROJECT_ID}', location='asia-south1')
    
    embeddings = VertexAIEmbeddings(model_name='textembedding-gecko')
    return embeddings
