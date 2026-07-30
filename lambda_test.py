import base64
import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL")

IMAGE_PATH = "./DATA/test/PNEUMONIA/person100_bacteria_475.jpeg"

def lambda_api_test():
    
    with open(IMAGE_PATH, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    payload = {
        "image_base64": encoded_string
    }
    
    print(f"Sending request")
    
    try:
        response = requests.post(API_URL, json=payload)
        
        print("\nRESULT")
        print(f"HTTP Code : {response.status_code}")
        
        if response.status_code == 200:
            resultat_json = response.json()
            print("Diagnostic :", resultat_json.get('diagnostic'))
            print("Probability :", round(resultat_json.get('pneumonie_probability', 0) * 100, 2), "%")
        else:
            print("API error:", response.text)
            
    except Exception as e:
        print("Connection error :", e)

if __name__ == "__main__":
    lambda_api_test()