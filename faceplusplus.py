import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load .env safely
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

def analyze_face(image_path):
    url = "https://api-us.faceplusplus.com/facepp/v3/detect"

    files = {"image_file": open(image_path, "rb")}
    data = {
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "return_attributes": "skinstatus"
    }

    response = requests.post(url, files=files, data=data)
    return response.json()

