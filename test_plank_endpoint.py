import cv2
import numpy as np
import io
from fastapi.testclient import TestClient
from fastapi_pose_server.app import app

client = TestClient(app)

def test_plank_endpoint():
    # Create a dummy image (black image)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()
    
    # Send request
    response = client.post(
        "/plank",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"flip": "0", "session": "test_session"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    data = response.json()
    assert "duration_perfect" in data
    assert "duration_imperfect" in data
    assert "is_correct" in data
    assert "feedback" in data
    
    # Since it's a black image, we expect "No pose detected" or similar
    # But the schema validation is the important part here.

if __name__ == "__main__":
    test_plank_endpoint()
