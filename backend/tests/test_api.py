import requests

def test_api_root():
    response = requests.get("http://backend:8765")
    assert response.status_code == 200
