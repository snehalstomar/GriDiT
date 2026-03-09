import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import random


def center_crop_arr(pil_image, image_size):

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def count_files_in_directory(directory):
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
    return total_files


def add_gaussian_noise(image, mean=0, std=10):

    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def process_image_resize_noise_blur(
    image_path,
    sr_factor,
    erosion_iterations=3,
    blur_radius=15,
    brightest_fraction=0.4,
    global_blur_radius=7,
):
    print(f"scale_factor = {sr_factor}.")
    # input()
    if sr_factor == 4:
        out_dim = 512
    elif sr_factor == 2:
        out_dim = 256
    else:
        raise Exception("Invalid Scale factor")
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_resize = cv2.resize(
        cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC),
        # (256, 256),
        (out_dim, out_dim),
        interpolation=cv2.INTER_CUBIC,
    )
    # Convert to grayscale
    noisy_image = add_gaussian_noise(image_resize, std=random.randint(10, 15))

    # Apply a global Gaussian blur to the entire noisy image
    probability_of_zero = 0.5
    probability_of_other_numbers = 0.5

    # Define the list of other numbers
    other_numbers = [9, 11, 13, 15]

    globally_blurred_image = None
    # Generate the random number
    if random.random() < probability_of_zero:
        # result = 0  # Choose 0 with 80% probability
        globally_blurred_image = noisy_image
    else:
        result = random.choice(
            other_numbers
        )  # Choose uniformly from [5, 7, 9, 11] with 20% probability
        global_blur_radius = result
        globally_blurred_image = cv2.GaussianBlur(
            noisy_image, (global_blur_radius, global_blur_radius), 0
        )

    # return image, globally_blurred_image
    return globally_blurred_image


class super_res_dset(Dataset):
    def __init__(self, root_dir, sr_factor):
        """
        Args:
            root_dir (str): Directory containing the dataset.
            transform (callable, optional): Transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.sr_factor = sr_factor
        if sr_factor == 4:
            out_dim = 512
        elif sr_factor == 2:
            out_dim = 256
        else:
            raise Exception("Invalid Scale factor")
        self.transform_in = transforms.Compose(
            [
                transforms.Resize((512, 512), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
        self.transform_degraded = transforms.Compose(
            [   
                transforms.Resize((out_dim, out_dim)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
    
    def __len__(self):
        return len(os.listdir(self.root_dir))


    def __getitem__(self, idx):
        folder_of_choice = self.root_dir
        image = Image.open(folder_of_choice + "/" + str(idx) + ".png").convert("RGB")
        degraded_img = process_image_resize_noise_blur(
            folder_of_choice + "/" + str(idx) + ".png",
            sr_factor=self.sr_factor,
            erosion_iterations=3,
            blur_radius=15,
            brightest_fraction=0.4,
            global_blur_radius=7,
        )
        degraded_img_PIL = Image.fromarray(degraded_img)
        img_tensor = self.transform_in(image)
        degraded_img_tensor = self.transform_degraded(degraded_img_PIL)
        return img_tensor, degraded_img_tensor
