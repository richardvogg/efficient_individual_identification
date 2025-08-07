import cv2
import matplotlib
matplotlib.use('Agg')

# Path to your video file
video_path = '/usr/users/vogg/sfb1528s3/B06/lemur_video_interaction_dataset/Videos/A_e1_c1_4328_4466_scrounging.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

# Display the frame
cv2.imshow('Frame', frame)
cv2.waitKey(0)  # Wait for a key press to close the window

# Clean up
cap.release()
cv2.destroyAllWindows()