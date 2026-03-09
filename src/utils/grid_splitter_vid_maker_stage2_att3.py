import os
import cv2
from natsort import natsorted
import argparse


def process_and_create_videos(input_dir, output_dir, args, fps=30):
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all subdirectories in the input directory
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        if not os.path.isdir(subdir_path):
            continue  # Skip if it's not a directory

        # Create corresponding output subdirectory
        output_subdir = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)

        # Get all image files in alphanumeric order
        image_files = natsorted(
            [
                f
                for f in os.listdir(subdir_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        if not image_files:
            print(f"No images found in {subdir_path}. Skipping...")
            continue

        saved_images = []  # To store paths of saved images for video creation

        # Process each image in the subdirectory
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(subdir_path, image_file)
            img = cv2.imread(image_path)

            if img is None or img.shape[:2] != (512, 512):
                print(f"Skipping invalid or non-512x512 image: {image_path}")
                continue

            # Split the 4x4 grid into 128x128 tiles
            tiles = []
            for row in range(4):
                for col in range(4):
                    tile = img[row * 128 : (row + 1) * 128, col * 128 : (col + 1) * 128]
                    tiles.append(
                        (row + 1, col + 1, tile)
                    )  # Store with indices (1-based)

            # Save tiles based on the saving scheme
            if args.condType == "one":
                for r, c, tile in tiles:
                    tile_filename = f"{len(saved_images):04d}_tile_{r}_{c}.png"
                    print(tile_filename)
                    tile_path = os.path.join(output_subdir, tile_filename)
                    cv2.imwrite(tile_path, tile)
                    saved_images.append(tile_path)
            elif args.condType == "three":
                if idx == 0:  # For the first image: Save all tiles
                    for r, c, tile in tiles:
                        tile_filename = f"{len(saved_images):04d}_tile_{r}_{c}.png"
                        print(tile_filename)
                        tile_path = os.path.join(output_subdir, tile_filename)
                        cv2.imwrite(tile_path, tile)
                        saved_images.append(tile_path)
                elif (
                    idx % 2 == 1
                ):  # For subsequent images: Save only last row (row index = 4)
                    for r, c, tile in tiles:
                        if r == 3:  # Only save tiles from the last row
                            tile_filename = f"{len(saved_images):04d}_tile_{r}_{c}.png"
                            tile_path = os.path.join(output_subdir, tile_filename)
                            cv2.imwrite(tile_path, tile)
                            saved_images.append(tile_path)
                elif idx % 2 == 0 and idx != 0:
                    for r, c, tile in tiles:
                        if r == 4:  # Only save tiles from the last row
                            tile_filename = f"{len(saved_images):04d}_tile_{r}_{c}.png"
                            tile_path = os.path.join(output_subdir, tile_filename)
                            cv2.imwrite(tile_path, tile)
                            saved_images.append(tile_path)

        # Create video from saved images
        if saved_images:
            video_output_path = os.path.join(output_dir, f"{subdir}.mp4")
            create_video_from_images(saved_images, video_output_path, fps=fps)
            print(f"Video created: {video_output_path}")


def create_video_from_images(image_paths, output_video_path, fps=30):
    """
    Creates a video from a sequence of image paths.

    Args:
        image_paths (list): List of paths to images to include in the video.
        output_video_path (str): Path to save the output video file.
        fps (int): Frames per second for the video.
    """
    if not image_paths:
        print("No images provided for video creation.")
        return

    # Read the first image to get dimensions
    first_image = cv2.imread(image_paths[0])
    height, width, _ = first_image.shape

    # Define VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Unable to read {img_path}. Skipping...")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved at {output_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targetDir", type=str, default="target_dir_for_fig")
    parser.add_argument("--inputDir", type=str, default="inpu_dir_for_fig")
    parser.add_argument("--fps", type=int, default="inpu_dir_for_fig")
    parser.add_argument("--condType", type=str, default="inpu_dir_for_fig")
    args = parser.parse_args()
    input_directory = args.inputDir  # Replace with your input directory path
    output_directory = args.targetDir
    frames_per_second = args.fps
    process_and_create_videos(
        input_directory, output_directory, args, fps=frames_per_second
    )
