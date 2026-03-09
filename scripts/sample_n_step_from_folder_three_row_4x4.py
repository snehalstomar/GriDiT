import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
# from diffusion_sequential_three_row import create_diffusion
from diffusion import create_diffusion_seq_three_row as create_diffusion
from diffusers.models import AutoencoderKL
from src.utils.download import find_model
from src.models.models_sequential import DiT_models
import argparse
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import os
import torch
from imageio import imread, imsave
import matplotlib.pyplot as plt
from skimage import io, transform
from tqdm import tqdm
import shutil

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


transform = transforms.Compose(
    [
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 512)),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True),
    ]
)
transform_reverse = transforms.Compose(
    [
        transforms.ToPILImage(),
        # Convert tensor to PIL image
    ]
)
device = "cuda" if torch.cuda.is_available() else "cpu"


def sample_n_length_sequences(
    diffusion,
    model,
    vae,
    start_img_path,
    required_length,
    args,
    target_dir,
    grid_size=16,
):

    print(target_dir + "/" + start_img_path.split("/")[-1].split(".")[0])
    if not os.path.exists(
        target_dir + "/" + start_img_path.split("/")[-1].split(".")[0]
    ):
        os.mkdir(target_dir + "/" + start_img_path.split("/")[-1].split(".")[0])
    shutil.copy(
        start_img_path,
        target_dir
        + "/"
        + start_img_path.split("/")[-1].split(".")[0]
        + "/"
        + "sample_0000.png",
    )
    intermediate_dir_path = (
        target_dir + "/" + start_img_path.split("/")[-1].split(".")[0] + "/intermediate"
    )
    os.makedirs(intermediate_dir_path, exist_ok=True)
    latent_size = args.image_size // 8
    prev_img_paths = []
    prev_img_paths.append(start_img_path)
    # num_samples = required_length // grid_size
    num_samples = required_length 
    for sample_idx in tqdm(range(num_samples)):
        img = transform(Image.open(prev_img_paths[-1])).to(device)
        img = img.unsqueeze(0)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            z_gt = vae.encode(img).latent_dist.sample().mul_(0.18215)
        n = 1
        z_noise = torch.randn(n, 4, latent_size, latent_size, device=device).to(device)
        z_gt = torch.cat([z_gt, z_gt], 0)
        z_noise = torch.cat([z_noise, z_noise], 0)
        model_kwargs = dict(cfg_scale=args.cfg_scale)
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z_noise.shape,
            z_noise,
            z_gt,
            False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
            intermediate_dir_path=intermediate_dir_path,
            sample_idx=sample_idx,
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        save_path = (
            target_dir
            + "/"
            + start_img_path.split("/")[-1].split(".")[0]
            + "/"
            + "sample_"
            + str(sample_idx + 1).zfill(4)
            + ".png"
        )

        save_image(samples, save_path, nrow=4, normalize=True, value_range=(-1, 1))
        prev_img_paths.append(save_path)


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    if args.ckpt is None:
        assert (
            args.model == "DiT-XL/2"
        ), "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size, num_classes=args.num_classes
    ).to(device)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    print(type(diffusion), diffusion)
    input()
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    start_img_dir_path = args.input_dir
    target_dir = args.target_dir
    for img_name in os.listdir(start_img_dir_path):

        if not os.path.exists(args.target_dir + "/" + img_name.split(".")[0]):
            os.makedirs(args.target_dir + "/" + img_name.split(".")[0])
        print("img_name", img_name)
        start_path = start_img_dir_path + "/" + img_name

        sample_n_length_sequences(
            diffusion,
            model,
            vae,
            start_path,
            # args.vidLength,
            ((args.vidLength - 8) // 12) + 1,
            args,
            target_dir,
            args.gridSz,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2"
    )
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).",
    )
    parser.add_argument("--target-dir", type=str, default="target_dir_for_fig")
    parser.add_argument("--input-dir", type=str, default="inpu_dir_for_fig")
    parser.add_argument("--vidLength", type=int, default="inpu_dir_for_fig")
    parser.add_argument("--gridSz", type=int, default="inpu_dir_for_fig")
    args = parser.parse_args()
    main(args)
