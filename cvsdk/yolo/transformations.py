import albumentations as A

"""
see: https://explore.albumentations.ai
"""

def custom_transforms(
        p_blur: float = 0.0,
        p_noise: bool = 0.0,
        p_compression: bool = 0.0,
        p_hue: bool = 0.0
    ):
    blur = A.OneOf([
        A.MotionBlur(
            blur_limit=7,
            p=1.0
        ),
        A.MedianBlur(
            blur_limit=7,
            p=1.0
        ),
        A.GaussianBlur(
            blur_limit=7,
            p=1.0
        ),
        A.Defocus(
            radius=(3,10), 
            alias_blur=(0.1,0.5), 
            p=1.0
        ),
        A.ZoomBlur(
            max_factor=[1, 1.31],
            step_factor=[0.01, 0.03],
            p=1.0
        )
    ], p=p_blur)

    compression = A.OneOf([
        A.Downscale(
            scale_range=[0.2, 0.75],
            interpolation_pair={"upscale":0,"downscale":0},
            p=1.0
        ),
        A.ImageCompression(
            compression_type="jpeg",
            quality_range=[10, 40],
            p=1.0,
        ),
        A.Posterize(
            num_bits=4,
            p=1.0
        ),
    ], p=p_compression)
    
    noise = A.OneOf([
        A.GaussNoise(
            var_limit=(10.0, 10.0),
            p=1.0
        ),
        A.ISONoise(
            color_shift=(0.01, 0.05),
            intensity=(0.1, 0.5),
            p=1.0,
        ),
        A.SaltAndPepper(
            amount=[0.01, 0.05],
            salt_vs_pepper=[0.5, 0.5],
            p=1.0,
        ),
        A.ShotNoise(
            scale_range=[0.1, 0.2],
            p=1.0
        )
    ], p=p_noise)

    color = A.OneOf([
        A.CLAHE(
            clip_limit=4.0,
            tile_grid_size=(8, 8),
            p=1.0
        ),
        A.HueSaturationValue(
            hue_shift_limit=20, 
            sat_shift_limit=30, 
            val_shift_limit=20, 
            p=1.0
        ),
        A.Illumination(
            mode="linear",
            intensity_range=[0.01, 0.2],
            effect_type="both",
            angle_range=[0, 360],
            center_range=[0.1, 0.9],
            sigma_range=[0.2, 1],
            p=1.0,
        ),
        A.RGBShift(
            r_shift_limit=[-20, 20],
            g_shift_limit=[-20, 20],
            b_shift_limit=[-20, 20],
            p=1.0,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=[-0.2, 0.2],
            contrast_limit=[-0.2, 0.2],
            brightness_by_max=True,
            ensure_safe_range=False,
            p=1.0,
        ),
        A.RandomGamma(
            gamma_limit=[40, 160],
            p=1.0,
        ),
        A.Sharpen(
            alpha=[0.2, 0.5],
            lightness=[0.5, 1],
            method="kernel",
            kernel_size=5,
            sigma=1,
            p=1.0
        ),
        A.ToGray(
            method="weighted_average",
            p=1.0
        ),
        A.ToSepia(
            p=1.0
        )
    ], p=p_hue)

    augmentations = []
    if p_blur:
        augmentations.append(blur)
    if p_noise:
        augmentations.append(noise)
    if p_compression:
        augmentations.append(compression)
    if p_hue:
        augmentations.append(color)

    return augmentations
