import cv2
import numpy as np
from fastapi.testclient import TestClient
from fastapi_pose_server.app import app

client = TestClient(app)

def test_high_knees_endpoint():
    # Create a dummy image (black image)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()
    
    # Send request
    response = client.post(
        "/high_knees",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"flip": "0", "session": "test_session"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    data = response.json()
    assert "reps" in data
    assert "reps_clean" in data
    assert "reps_wrong" in data
    assert "phase" in data
    assert "feedback" in data
    
    print("High Knees endpoint test passed!")

if __name__ == "__main__":
    test_high_knees_endpoint()
