import os
import random

import cv2
import numpy as np


def process_video_to_npy(video_path, output_dir, output_filename, target_fps=3):
    """
    Extracts frames from a video, detects/randomly crops them to 224x224,
    and saves the entire collection as a single .npy file.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the output .npy file.
        output_filename (str): Name of the output .npy file (e.g., 'data.npy').
        target_fps (int): Number of frames to process per second.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("Error: Could not load the Haar Cascade classifier.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(original_fps / target_fps))
    if frame_interval < 1:
        frame_interval = 1

    frame_count = 0
    saved_images = []
    crop_size = 224

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            h_frame, w_frame, _ = frame.shape

            # Skip if frame is too small for a 224x224 crop
            if h_frame < crop_size or w_frame < crop_size:
                print(
                    f"Frame {frame_count} is too small for a {crop_size}x{crop_size} crop. Skipping."
                )
                frame_count += 1
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            cropped_image = None
            if len(faces) > 0:
                # Use the first detected face for cropping
                (x, y, w, h) = faces[0]
                center_x = x + w // 2
                center_y = y + h // 2

                x1 = max(0, center_x - crop_size // 2)
                y1 = max(0, center_y - crop_size // 2)
                x2 = min(w_frame, x1 + crop_size)
                y2 = min(h_frame, y1 + crop_size)

                cropped_image = frame[y1:y2, x1:x2]

                if (
                    cropped_image.shape[0] != crop_size
                    or cropped_image.shape[1] != crop_size
                ):
                    cropped_image = cv2.resize(cropped_image, (crop_size, crop_size))
            else:
                # If no face, perform a random crop
                rand_x = random.randint(0, w_frame - crop_size)
                rand_y = random.randint(0, h_frame - crop_size)
                cropped_image = frame[
                    rand_y : rand_y + crop_size, rand_x : rand_x + crop_size
                ]

            # Add the cropped image to our list
            if cropped_image is not None:
                saved_images.append(cropped_image)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Convert the list of images to a NumPy array
    final_array = np.array(saved_images)

    # Check if the array has 30 frames
    if final_array.shape[0] < 30:
        print(
            f"Warning: Only {final_array.shape[0]} frames were processed. Not enough to fill a (30, 224, 224, 3) array."
        )
        # You may want to pad the array with zeros or handle this differently

    # Save the NumPy array to disk
    output_path = os.path.join(output_dir, output_filename)
    np.save(output_path, final_array)

    print(
        f"\nSuccessfully saved {final_array.shape[0]} images to '{output_path}' with shape {final_array.shape}."
    )


if __name__ == "__main__":
    # --- Usage ---
    video_file = ".\\videos\\1100081044.avi"
    output_directory = "video_data"
    output_file = "video_frames.npy"

    process_video_to_npy(video_file, output_directory, output_file, target_fps=3)
