import numpy as np
import jax.numpy as jnp
import jax
import os
import argparse
from datetime import datetime
from heterogeneity_gym.hsp90 import hsp90_jax as hsp90
from heterogeneity_gym.hsp90._utils import initialize_latent_code
from heterogeneity_gym.pose_generation import generate_random_3d_poses
from tqdm import tqdm
from typing import Optional, Tuple
from jaxtyping import Array, Int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render noisy images")
    parser.add_argument(
        "--num_images", type=int, default=64, help="Number of images to render"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for rendering"
    )
    parser.add_argument(
        "--image_width", type=int, default=256, help="Batch size for rendering"
    )
    parser.add_argument(
        "--output_folder", type=str, default=None, help="Output containing raw_data"
    )
    parser.add_argument(
        "--translation_std_dev",
        type=float,
        default=0.0,
        help="Standard deviation of pose",
    )
    parser.add_argument(
        "--noise_std_dev", type=float, default=10.0, help="Standard deviation of noise"
    )
    # Optionally provide a latent parameter
    parser.add_argument(
        "--latent",
        type=int,
        default=None,
        help="Latent code for images",
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


def main():
    args = parse_args()

    # Initialize random number generators
    seed = 1
    key = jax.random.key(seed + 1)
    rng = np.random.default_rng(seed)  # Needed for scipy Rotation.random

    latent_code = initialize_latent_code(args.num_images, args.latent, key)
    poses = generate_random_3d_poses(args.num_images, args.translation_std_dev, rng)
    poses = jnp.array(poses)

    # Render Images
    model = hsp90.HSP90_Model(
        defocus_range=(1000, 2000),
        pixel_size=1.1,
        image_width_in_pixels=args.image_width,
        noise_std=args.noise_std_dev,
        seed=1,
    )

    all_images = []
    all_poses = []
    all_defoci = []
    all_astigmatisms = []
    num_batches = int(np.ceil(args.num_images / args.batch_size))
    for batch_idx in tqdm(range(num_batches), desc="Image batch"):
        i = batch_idx * args.batch_size
        num_images_in_batch = min(args.batch_size, args.num_images - i)

        batch_poses = poses[i : i + num_images_in_batch]
        batch_latent = latent_code[i : i + num_images_in_batch]

        noisy_images, structures, batch_poses, (defocus, astigmatism) = (
            model.render_images_from_latent(batch_latent, poses=batch_poses)
        )
        print(noisy_images.shape)
        print(noisy_images[:, :2, :2])

        all_images.append(noisy_images)
        all_poses.append(batch_poses)
        all_defoci.append(defocus)
        all_astigmatisms.append(astigmatism)

        i += num_images_in_batch

    all_images = jnp.concatenate(all_images, axis=0)
    all_poses = jnp.concatenate(all_poses, axis=0)
    all_defoci = jnp.concatenate(all_defoci, axis=0)
    all_astigmatisms = jnp.concatenate(all_astigmatisms, axis=0)

    # Save images as numpy files
    output_folder = setup_output_folder(args.output_folder)

    np.save(os.path.join(output_folder, "images.npy"), all_images)
    np.save(os.path.join(output_folder, "poses.npy"), all_poses)
    np.save(os.path.join(output_folder, "defoci.npy"), all_defoci)
    np.save(os.path.join(output_folder, "astigmatisms.npy"), all_astigmatisms)
    np.save(os.path.join(output_folder, "latent.npy"), latent_code)


if __name__ == "__main__":
    main()
