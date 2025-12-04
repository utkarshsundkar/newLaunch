from fastapi.testclient import TestClient
from fastapi_pose_server.app import app
import os

client = TestClient(app)

def test_plank_video():
    video_path = "/Users/utkarshsundkar/Desktop/error/MyApp/videos/plank.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}")
        return

    print(f"Testing with video: {video_path}")
    
    with open(video_path, 'rb') as video_file:
        files = {'file': ('plank.mp4', video_file, 'video/mp4')}
        data = {'flip': '0'}
        
        # Using the analyze_plank endpoint
        response = client.post("/analyze_plank", files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Analysis Results:")
            print(f"  Total Duration: {data['duration_total']:.2f}s")
            print(f"  Perfect Duration: {data['duration_perfect']:.2f}s")
            print(f"  Imperfect Duration: {data['duration_imperfect']:.2f}s")
            print(f"  Processed FPS: {data['processed_fps']:.2f}")
            if data.get('diagnostics'):
                print(f"  Diagnostics (first 3): {data['diagnostics'][:3]}")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    test_plank_video()
