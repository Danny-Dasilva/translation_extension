"""Test script for manga translation API"""
import requests
import base64
import json
from pathlib import Path

def test_health_check():
    """Test health check endpoint"""
    response = requests.get("http://localhost:8000/health")
    print(f"Health Check Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_translate_with_sample():
    """Test translation with a sample base64 image"""
    # This is a minimal test - in production you'd use actual manga images
    url = "http://localhost:8000/translate"
    
    # Sample request (you'll need to replace with real base64 manga image)
    payload = {
        "base64Images": [
            "data:image/jpeg;base64,/9j/4AAQSkZJRg..."  # Replace with real image
        ],
        "targetLanguage": "English"
    }
    
    print("Testing /translate endpoint...")
    print("Note: This requires a valid GEMINI_API_KEY in .env file")
    print()
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Found {len(result['images'])} image(s)")
            for idx, image_boxes in enumerate(result['images']):
                print(f"  Image {idx + 1}: {len(image_boxes)} text boxes detected")
                for box in image_boxes:
                    print(f"    - OCR: {box['ocrText']} -> Translation: {box['translatedText']}")
        else:
            print(f"Error: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure it's running:")
        print("  cd backend && uv run uvicorn app.main:app --reload")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Manga Translation API Test Suite")
    print("=" * 60)
    print()
    
    test_health_check()
    # test_translate_with_sample()  # Uncomment when you have a real image
    
    print("\nTo test the full translation pipeline:")
    print("1. Create backend/.env with your GEMINI_API_KEY")
    print("2. Start server: cd backend && uv run uvicorn app.main:app --reload")
    print("3. Add a real base64 manga image to this script")
    print("4. Uncomment test_translate_with_sample() above")
