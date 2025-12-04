import requests

# Test the pushup analysis endpoint with your pushups1 video
url = "http://192.168.0.105:8001/pushups"

# Path to your test video
video_path = "/Users/utkarshsundkar/Desktop/error/MyApp/videos/pushups1.mp4"

# Open and send the video file
with open(video_path, 'rb') as video_file:
    files = {'file': ('pushups1.mp4', video_file, 'video/mp4')}
    data = {'flip': '1'}  # Flip the video if needed
    
    print("Sending video for pushup analysis...")
    response = requests.post(url, files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Pushup Reps Detected: {result['reps']}")
        print(f"Processing FPS: {result['fps']:.2f}")
        print(f"Phase: {result['phase']}")
        print(f"Body Angle: {result['body_angle']:.2f}")
        print(f"Arms Angle: {result['arms_angle']:.2f}")
    else:
        print(f"Error: {response.text}")