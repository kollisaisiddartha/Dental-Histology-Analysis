import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor

def preprocess_frame(frame):
    # Resize the frame to a lower resolution
    #resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize to 50% of original size
    return frame

def stitch_frames(input_folder, output_path, batch_size=10):
    # Get the list of frame filenames
    frame_files = sorted(os.listdir(input_folder))

    # Read the first image to get dimensions
    first_frame_path = os.path.join(input_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, channels = first_frame.shape

    # Create a Stitcher object
    stitcher = cv2.Stitcher_create()

    # Initialize the list to hold the preprocessed frames
    preprocessed_frames = []

    # Start measuring execution time
    start_time = time.time()

    # Process frames in batches
    for i in range(0, len(frame_files), batch_size):
        batch_files = frame_files[i:i + batch_size]

        # Function to preprocess a single frame
        def preprocess_frame_wrapper(frame_file):
            frame_path = os.path.join(input_folder, frame_file)
            frame = cv2.imread(frame_path)
            preprocessed_frame = preprocess_frame(frame)
            return preprocessed_frame

        # Perform preprocessing in parallel using threads
        with ThreadPoolExecutor() as executor:
            batch_preprocessed_frames = list(executor.map(preprocess_frame_wrapper, batch_files))

        preprocessed_frames.extend(batch_preprocessed_frames)

    # Stitch preprocessed frames together
    status, stitched_image = stitcher.stitch(preprocessed_frames)

    # Check if stitching was successful
    if status == cv2.Stitcher_OK:
        # Save the stitched image with original resolution
        cv2.imwrite(output_path, stitched_image)
        print("Stitched image saved successfully!")
    else:
        print("Stitching failed!")

    # End measuring execution time
    end_time = time.time()

    # Calculate and print the total execution time
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")

# Example usage
input_folder = 'output_frames14_10x_pre'  # Change this to the folder containing frames
output_path = 'stitched_image14_10x_pre_hres.jpg'  # Change this to the desired output path
stitch_frames(input_folder, output_path)
