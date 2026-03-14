import os
from typing import List, Dict
from PIL import Image
import numpy as np
from tqdm import tqdm


def get_sorted_filenames(main_dir: str) -> Dict[str, List[str]]:
    """
    Iterates through every subsubdirectory in the given main directory
    and returns a dictionary where keys are the full paths to subsubdirectories
    and values are sorted lists of filenames in those subsubdirectories.

    Args:
        main_dir (str): Path to the main directory.

    Returns:
        Dict[str, List[str]]: A dictionary with full paths to subsubdirectories as keys
                              and sorted filenames as values.
    """
    sorted_filenames = {}

    # Iterate through all directories and files in main_dir
    for dirpath, dirnames, filenames in os.walk(main_dir):
        # Check if the current directory is a subsubdirectory (two levels deep)
        if dirpath.count(os.sep) == main_dir.count(os.sep) + 2:
            # Sort the filenames alphabetically
            sorted_filenames[dirpath] = sorted(filenames)

    return sorted_filenames


def divide_into_chunks(filenames, chunk_size=16):
    """
    Divides a list of filenames into chunks of specified size, discarding the final chunk if it has fewer elements.
    """
    return [
        filenames[i : i + chunk_size]
        for i in range(0, len(filenames), chunk_size)
        if len(filenames[i : i + chunk_size]) == chunk_size
    ]


def create_image_grid(
    chunk, base_folder_path, image_size=(128, 128), grid_size=(512, 512)
):
    """
    Creates a 512x512 grid image from a chunk of 16 images resized to 128x128.
    """
    # Create a blank grid image
    grid_image = Image.new("RGB", grid_size)

    # Calculate number of images per row and column
    images_per_row = grid_size[0] // image_size[0]

    # Resize and place each image in the grid
    for index, filename in enumerate(chunk):
        # Open and resize the image
        img = Image.open(base_folder_path + "/" + filename).resize(image_size)

        # Calculate position in the grid
        row = index // images_per_row
        col = index % images_per_row
        x = col * image_size[0]
        y = row * image_size[1]

        # Paste the image into the grid
        grid_image.paste(img, (x, y))

    return grid_image


def save_image_grid(grid_image, target_dir, chunk_index, processed_chunks_n):
    """
    Saves a grid image to the target directory with a filename based on the chunk index.
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Save the grid image with the specified filename
    grid_image.save(
        os.path.join(target_dir, f"{int(processed_chunks_n + chunk_index)}.png")
    )


def process_image_chunks(filenames, target_dir, base_folder_path, processed_chunks_n):
    """
    Processes a list of filenames by dividing them into chunks of 16,
    creating a 512x512 grid for each chunk, and saving them to the target directory.
    """
    # Divide filenames into chunks of 16
    chunks = divide_into_chunks(filenames)
    for chunk_index, chunk in enumerate(chunks):
        # Create a grid image for the current chunk
        grid_image = create_image_grid(chunk, base_folder_path)
        # Save the grid image to the target directory
        save_image_grid(grid_image, target_dir, chunk_index, processed_chunks_n)
    processed_chunks_n += len(chunks)
    return processed_chunks_n


if __name__ == "__main__":
    dset_dir = ""  # path to original dataset directory.
    target_dir = ""  # path to the required grid-based dataset directory.

    print("getting fname list ...")
    sorted_filenames = get_sorted_filenames(dset_dir)
    print("got fname list ...")

    n_chunks_processed = 0
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    print("organizing")
    for folder_path, sorted_fname_list in tqdm(sorted_filenames.items()):
        n_chunks_processed = process_image_chunks(
            sorted_fname_list, target_dir, folder_path, n_chunks_processed
        )
