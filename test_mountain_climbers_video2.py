from fastapi.testclient import TestClient
from fastapi_pose_server.app import app
import os

client = TestClient(app)

def test_mountain_climbers_video2():
    video_path = "/Users/utkarshsundkar/Desktop/error/MyApp/videos/MountainClimbers2.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}")
        # Try alternative names
        alt_path = "/Users/utkarshsundkar/Desktop/error/MyApp/videos/mountain_climbers_2.mp4"
        if os.path.exists(alt_path):
            video_path = alt_path
        else:
            print(f"Also tried: {alt_path}")
            return

    print(f"Testing with video: {video_path}")
    
    with open(video_path, 'rb') as video_file:
        files = {'file': (os.path.basename(video_path), video_file, 'video/mp4')}
        data = {'flip': '0'}
        
        # Using the analyze_mountain_climbers endpoint
        response = client.post("/analyze_mountain_climbers", files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Analysis Results:")
            print(f"  Total Reps: {data['reps']}")
            print(f"  Clean Reps: {data['reps_clean']}")
            print(f"  Wrong Reps: {data['reps_wrong']}")
            print(f"  Processed FPS: {data['processed_fps']:.2f}")
            if data.get('diagnostics'):
                print(f"\n  Diagnostics (showing all {len(data['diagnostics'])} reps):")
                for diag in data['diagnostics']:
                    status = "✓ CLEAN" if diag.get('is_clean') else "✗ WRONG"
                    print(f"    Rep {diag['rep']} at {diag['time']:.2f}s [{status}]: {diag['feedback']}")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    test_mountain_climbers_video2()
