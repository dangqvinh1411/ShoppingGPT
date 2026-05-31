# config.py
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
DATA_PRODUCT_PATH = os.getcwd() + "/data/products.db"
DATA_TEXT_PATH = os.getcwd() + "/data/policy.txt"
STORE_DIRECTORY = os.getcwd() + "/data/datastore"

# Embeddings
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/embedding-001")