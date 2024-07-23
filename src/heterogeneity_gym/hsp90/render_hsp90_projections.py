import numpy as np
import jax.numpy as jnp
import jax
import os
import argparse
from datetime import datetime
from heterogeneity_gym.hsp90._utils import initialize_latent_code
from heterogeneity_gym.hsp90 import hsp90_jax as hsp90
from heterogeneity_gym.pose_generation import generate_icosahedral_poses
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from typing import Optional, Tuple
from jaxtyping import Array, Int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render noisy images")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for rendering"
    )
    parser.add_argument(
        "--n_render",
        type=int,
        default=24576,
        help="Number of images to render before saving",
    )
    parser.add_argument(
        "--output_folder", type=str, default=None, help="Output containing raw_data"
    )
    # Optionally provide a latent parameter
    parser.add_argument(
        "--latent",
        type=int,
        default=0,
        help="Latent code for images",
    )
    parser.add_argument(
        "--image_width", type=int, default=256, help="Batch size for rendering"
    )
    parser.add_argument(
        "--nu", type=int, default=16, help="Number of subdivisions in the icosphere"
    )
    parser.add_argument(
        "--n_theta", type=int, default=48, help="Number of angles to rotate around axis"
    )
    return parser.parse_args()


def setup_output_folder(output_path: Optional[str] = None) -> str:
    if output_path is None:
        date_time = datetime.now()
        str_date_time = date_time.strftime("%d-%m-%Y_%H:%M:%S")
        output_path = str_date_time

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path


def rendering_epoch(model, poses, latent_code, batch_size):
    all_images = []
    all_rotations = []

    num_images = len(poses)
    num_batches = int(np.ceil(num_images / batch_size))

    for batch_idx in tqdm(range(num_batches), desc="Image batch"):
        i = batch_idx * batch_size
        num_images_in_batch = min(batch_size, num_images - i)

        batch_poses = poses[i : i + num_images_in_batch]
        batch_latent = latent_code[i : i + num_images_in_batch]

        clean_images, __, batch_poses = model.render_projections_from_latent(
            batch_latent, poses=batch_poses
        )

        all_images.append(clean_images)
        all_rotations.append(batch_poses)

        i += num_images_in_batch

    all_images = jnp.concatenate(all_images, axis=0)
    all_rotations = jnp.concatenate(all_rotations, axis=0)
    return all_images, all_rotations


def main():
    args = parse_args()

    poses = generate_icosahedral_poses(args.nu, args.n_theta)
    print(f"Rendering a total of {poses} poses".)
    latent_code = jnp.ones(len(poses), dtype="int32") * args.latent
    # Render Images
    model = hsp90.HSP90_Model(
        defocus_range=(1000, 2000),
        pixel_size=1.1,
        image_width_in_pixels=args.image_width,
        noise_std=0.0,
        seed=0,
    )

    num_images = len(poses)

    # Output Folder
    output_folder = setup_output_folder(args.output_folder)

    # Iterate over rotations, batched by
    num_rotation_batches = int(np.ceil(num_images / args.n_render))
    for i in tqdm(range(num_rotation_batches)):
        poses_to_render = poses[i * args.n_render : (i + 1) * args.n_render]
        images, poses_to_render = rendering_epoch(
            model, poses_to_render, latent_code, batch_size=args.batch_size
        )

        np.save(
            os.path.join(output_folder, f"images_l_{args.latent}_batch_{i}.npy"),
            images,
        )
        np.save(
            os.path.join(output_folder, f"poses_l_{args.latent}_batch_{i}.npy"),
            poses,
        )


if __name__ == "__main__":
    main()
