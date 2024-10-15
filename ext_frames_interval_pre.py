import cv2  # Import OpenCV library for video processing
import os   # Import operating system module for file operations

def preprocess_frame(frame):
    # Resize the frame to enhance resolution
    resized_frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    
    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(resized_frame, (5, 5), 0)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    lab_planes = cv2.split(lab)
    
    # Convert the tuple to a list for modification
    lab_planes = list(lab_planes)
    
    # Apply histogram equalization to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    
    # Merge LAB channels
    lab = cv2.merge(lab_planes)
    
    # Convert back to BGR color space
    equalized_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return equalized_frame

def extract_frames(video_path, output_folder, interval=1.25):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):  # Check if the output folder exists
        os.makedirs(output_folder)          # If not, create the output folder

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)  # Open the video file for reading
    fps = video_capture.get(cv2.CAP_PROP_FPS)     # Get frames per second of the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count
    
    frame_number = 0   # Initialize frame number counter
    current_time = 0   # Initialize current time counter

    while True:  # Start a loop to extract frames
        # Set frame position to the current frame number
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame from the video
        success, frame = video_capture.read()
        
        # Check if the frame was successfully read
        if not success:
            break  # If not, exit the loop
        
        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Save the preprocessed frame as an image file
        output_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")  # Construct output path
        cv2.imwrite(output_path, processed_frame)  # Save the frame as an image file
        
        # Increment frame number and calculate current time
        frame_number += int(interval * fps)  # Increment frame number based on interval and fps
        current_time += interval             # Update current time based on interval

        # Break the loop when all frames are extracted
        if frame_number >= total_frames:
            break  # If frame number exceeds total frames, exit the loop

    video_capture.release()   # Release the video capture object
    cv2.destroyAllWindows()   # Close any OpenCV windows

# Example usage
video_path = 'Videos/23_10x  (1).mov'        # Path to the input video file
output_folder = 'output_frames23_10x_1_pre'    # Path to the desired output folder
extract_frames(video_path, output_folder)  # Call the extract_frames function with the specified paths

