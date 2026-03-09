import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image

# attempt 1
# from diffusion_sequential_interpolation_sampling import create_diffusion
from diffusion import create_diffusion_seq_interpolation_sampling as create_diffusion
# attempt 2
# from diffusion_sequential_interpolation_sampling_attemp2 import create_diffusion
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
    diffusion, model, vae, stage1_grids_path, args, target_dir, grid_size=16
):
    # print("stage1_grids_path", stage1_grids_path)
    # print(os.listdir(stage1_grids_path))
    latent_size = args.image_size // 8
    stage_1_grid_imgs = sorted(os.listdir(stage1_grids_path))
    # print(stage_1_grid_imgs, type(stage_1_grid_imgs))
    stage_1_grid_imgs = stage_1_grid_imgs[1:]
    # print(stage_1_grid_imgs, type(stage_1_grid_imgs))
    # exit()
    working_grid_img_names = [
        (stage_1_grid_imgs[i], stage_1_grid_imgs[i + 1])
        for i in range(len(stage_1_grid_imgs) - 1)
    ]
    print(f"working_grid_img_name_chunks:{working_grid_img_names}")
    # input()
    for idx, src_tgt_tuple in tqdm(enumerate(working_grid_img_names)):
        z_gt_src = None
        z_gt_tgt = None
        if idx == 0:
            shutil.copy(
                stage1_grids_path + "/" + src_tgt_tuple[0],
                target_dir
                + "/"
                + stage1_grids_path.split("/")[-1]
                + "/"
                + src_tgt_tuple[0],
            )
            print(
                f'copied {stage1_grids_path + "/" + src_tgt_tuple[0]} to {target_dir+ "/"+ stage1_grids_path.split("/")[-1]+ "/"+ src_tgt_tuple[0]}.'
            )
            # input()
        shutil.copy(
            stage1_grids_path + "/" + src_tgt_tuple[1],
            target_dir
            + "/"
            + stage1_grids_path.split("/")[-1]
            + "/sample_"
            + str(int(src_tgt_tuple[1].split(".")[0].split("_")[-1]) * 2).zfill(4)
            + ".png",
        )
        print(
            f'copied {stage1_grids_path + "/" + src_tgt_tuple[1]} to {target_dir + "/" + stage1_grids_path.split("/")[-1] + "/sample_" + str(int(src_tgt_tuple[1].split(".")[0].split("_")[-1])*2).zfill(4) + ".png"}'
        )
        src_img = (
            transform(Image.open(stage1_grids_path + "/" + src_tgt_tuple[0]))
            .unsqueeze(0)
            .to(device)
        )
        print(f"src_img_shape: {src_img.shape}")
        tgt_img = (
            transform(Image.open(stage1_grids_path + "/" + src_tgt_tuple[1]))
            .unsqueeze(0)
            .to(device)
        )
        print(f"tgt_img_shape: {tgt_img.shape}")
        intermediate_dir_path = f"{target_dir}/{stage1_grids_path.split('/')[-1]}/{src_tgt_tuple[0].split('.')[0].split('_')[-1]}_{src_tgt_tuple[1].split('.')[0].split('_')[-1]}"
        os.makedirs(intermediate_dir_path, exist_ok=True)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            z_gt_src = vae.encode(src_img).latent_dist.sample().mul_(0.18215)
            z_gt_tgt = vae.encode(tgt_img).latent_dist.sample().mul_(0.18215)
            print(
                f"z_gt_src.shape: {z_gt_src.shape} | z_gt_src.shape: {z_gt_src.shape}"
            )
        n = 1
        z_noise = torch.randn(n, 4, latent_size, latent_size, device=device).to(device)
        z_noise = torch.cat([z_noise, z_noise], 0)
        z_gt_src = torch.cat([z_gt_src, z_gt_src], 0)
        z_gt_tgt = torch.cat([z_gt_tgt, z_gt_tgt], 0)
        model_kwargs = dict(cfg_scale=args.cfg_scale)
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z_noise.shape,
            z_noise,
            z_gt_src,
            z_gt_tgt,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
            intermediate_dir_path=intermediate_dir_path,
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample
        save_path = (
            target_dir
            + "/"
            + stage1_grids_path.split("/")[-1]
            + "/sample_"
            + str((int(src_tgt_tuple[1].split(".")[0].split("_")[-1]) * 2) - 1).zfill(4)
            + ".png"
        )
        save_image(samples, save_path, nrow=4, normalize=True, value_range=(-1, 1))
        print(
            f'saved interpolated grid img to {target_dir + "/" + stage1_grids_path.split("/")[-1]+ "/sample_"+ str((int(src_tgt_tuple[1].split(".")[0].split("_")[-1])*2)-1).zfill(4)+ ".png"}.'
        )
        # input()


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
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    start_img_dir_path = args.input_dir
    target_dir = args.target_dir
    for img_dir_name in os.listdir(start_img_dir_path):
        if not os.path.exists(args.target_dir + "/" + img_dir_name):
            os.makedirs(args.target_dir + "/" + img_dir_name)
        print(f"running stgage 2 sampling for... {start_img_dir_path}/{img_dir_name}")
        start_path = start_img_dir_path + "/" + img_dir_name
        sample_n_length_sequences(
            diffusion, model, vae, start_path, args, target_dir, args.gridSz
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
