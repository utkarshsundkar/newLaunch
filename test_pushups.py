import requests

# Test the pushup analysis endpoint with your video
url = "http://192.168.0.105:8001/pushups"

# Path to your test video
video_path = "/Users/utkarshsundkar/Desktop/error/MyApp/videos/pushups.mp4"

# Open and send the video file
with open(video_path, 'rb') as video_file:
    files = {'file': ('pushups.mp4', video_file, 'video/mp4')}
    data = {'flip': '1'}  # Flip the video if needed
    
    response = requests.post(url, files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")