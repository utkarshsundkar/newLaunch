import requests

# Test the diamond pushup analysis endpoint with your Diamond_Pushups.mp4 video
url = "http://192.168.0.105:8001/diamond_pushups"

# Path to your Diamond_Pushups.mp4 video
video_path = "/Users/utkarshsundkar/Desktop/error/MyApp/videos/Diamond_Pushups.mp4"

# Open and send the video file
with open(video_path, 'rb') as video_file:
    files = {'file': ('Diamond_Pushups.mp4', video_file, 'video/mp4')}
    data = {'flip': '1'}  # Flip the video if needed
    
    print("Sending Diamond_Pushups.mp4 video for analysis...")
    response = requests.post(url, files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Diamond Pushup Reps Detected: {result['reps']}")
        print(f"Processing FPS: {result['fps']:.2f}")
        print(f"Phase: {result['phase']}")
        print(f"Body Angle: {result['body_angle']:.2f}")
        print(f"Arms Angle: {result['arms_angle']:.2f}")
        print(f"Hand Distance: {result['hand_distance']:.2f}")
        print(f"Is Down: {result['is_down']}")
        print(f"Is Up: {result['is_up']}")
    else:
        print(f"Error: {response.text}")