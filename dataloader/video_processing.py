"""Shared GenVideo preprocessing for image lists and directly decoded videos.

Both datasets construct transforms before sampling and use the same RNG order.
Images supplied by readers must be RGB uint8. Padding is zero AFTER normalization.
"""
import math
import random

import albumentations
import numpy as np
import torch

NUM_FRAMES = 8
IMAGE_SIZE = 224


def virtual_frame_indices(frame_count, fps):
    """Approximate the sequence produced by Preprocess/video2frame.py.

    Use 2 FPS for videos >= 4 s, otherwise ceil(8 / duration) FPS.
    Each output timestamp maps to the nearest original frame (half up).
    This is an FPS-based approximation, not FFmpeg's exact -r sync algorithm;
    for VFR input it uses the average FPS reported by OpenCV.
    Repeated source indices represent upsampling BEFORE clip selection.
    """
    if frame_count < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("frame_count and finite fps must be positive")
    duration = frame_count / fps
    output_fps = 2 if duration >= 4 else math.ceil(NUM_FRAMES / duration)
    # nextafter avoids a spurious extra frame at floating-point boundaries.
    output_count = max(1, math.ceil(math.nextafter(duration * output_fps, -math.inf)))
    return [
        min(frame_count - 1, int(i * fps / output_fps + 0.5))
        for i in range(output_count)
    ]


def crop_center_by_percentage(image, percentage):
    height, width = image.shape[:2]

    if width > height:
        left_pixels = int(width * percentage)
        right_pixels = int(width * percentage)
        start_x = left_pixels
        end_x = width - right_pixels
        cropped_image = image[:, start_x:end_x]
    else:
        up_pixels = int(height * percentage)
        down_pixels = int(height * percentage)
        start_y = up_pixels
        end_y = height - down_pixels
        cropped_image = image[start_y:end_y, :]

    return cropped_image


def sample_frame_indices(frame_count, count=NUM_FRAMES):
    """Select a contiguous window in the extracted-frame sequence in every mode."""
    if frame_count < 0 or count < 1:
        raise ValueError("frame_count must be nonnegative and count must be positive")
    start = random.randint(0, frame_count - count) if frame_count >= count else 0
    return list(range(start, min(start + count, frame_count)))


def _compose(augmentations):
    transform = albumentations.Compose(augmentations)
    # Albumentations 2 uses private generators; seed them from the same RNG as
    # clip selection so both datasets remain reproducible with random.seed().
    if hasattr(transform, 'set_random_seed'):
        transform.set_random_seed(random.getrandbits(32))
    return transform


def build_train_transform():
    aug_list = [albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE)]

    if random.random() < 0.5:
        aug_list.append(albumentations.HorizontalFlip(p=1.0))
    if random.random() < 0.5:
        quality_score = random.randint(50, 100)
        aug_list.append(_compression(quality_score))
    if random.random() < 0.3:
        aug_list.append(_noise())
    if random.random() < 0.3:
        aug_list.append(albumentations.GaussianBlur(blur_limit=(3, 5), p=1.0))
    if random.random() < 0.001:
        aug_list.append(albumentations.ToGray(p=1.0))

    aug_list.append(albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0))
    return _compose(aug_list)


def build_eval_transform(task='normal'):
    aug_list = [albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE)]

    if task == 'JPEG_Compress_Attack':
        aug_list.append(_compression(35, p=1.0))
    if task == 'FLIP_Attack':
        if random.random() < 0.5:
            aug_list.append(albumentations.HorizontalFlip(p=1.0))
        else:
            aug_list.append(albumentations.VerticalFlip(p=1.0))
    if task == 'CROP_Attack':
        random_crop_x = random.randint(0, 16)
        random_crop_y = random.randint(0, 16)
        crop_width = random.randint(160, 208)
        crop_height = random.randint(160, 208)
        aug_list.append(albumentations.Crop(x_min=random_crop_x, y_min=random_crop_y, x_max=random_crop_x+crop_width, y_max=random_crop_y+crop_height))
        aug_list.append(albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE))

    if task == 'Color_Attack':
        index = random.choice([i for i in range(4)])
        dicts = {0:[0.5,0,0,0],1:[0,0.5,0,0],2:[0,0,0.5,0],3:[0,0,0,0.5]}
        brightness,contrast,saturation,hue = dicts[index]
        aug_list.append(albumentations.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))

    if task == 'Gaussian_Attack':
        aug_list.append(albumentations.GaussianBlur(blur_limit=(7, 7), p=1.0))

    aug_list.append(albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0))
    return _compose(aug_list)


def prepare_clip(frame_count, read_frame, training=False, video_id='', task='normal'):
    """Read only selected frames and return [8, 3, 224, 224] float32."""
    transform = build_train_transform() if training else build_eval_transform(task)
    indices = sample_frame_indices(frame_count)
    percentage = (0.15 if video_id.startswith('real/youku') else 0.0) if training else 0.1
    frames = []
    for index in indices:
        frame = read_frame(index)
        if percentage:
            frame = crop_center_by_percentage(frame, percentage)
        frame = transform(image=frame)['image']
        frames.append(torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))).float())
    frames.extend(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE) for _ in range(NUM_FRAMES - len(frames)))
    return torch.stack(frames)


def _compression(quality, p=0.5):
    if int(albumentations.__version__.split('.')[0]) >= 2:
        return albumentations.ImageCompression(quality_range=(quality, quality), p=p)
    return albumentations.ImageCompression(quality_lower=quality, quality_upper=quality, p=p)


def _noise():
    if int(albumentations.__version__.split('.')[0]) >= 2:
        return albumentations.GaussNoise(std_range=(10**0.5 / 255, 50**0.5 / 255), p=1.0)
    return albumentations.GaussNoise(var_limit=(10, 50), p=1.0)
