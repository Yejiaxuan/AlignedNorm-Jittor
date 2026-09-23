from PIL import Image
from jittor import transform


INTERPOLATION_MODES = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "nearest": Image.NEAREST,
}


class ResizeShortEdge:
    """Match torchvision Resize(int): preserve aspect ratio using the short edge."""

    def __init__(self, size, interpolation):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, image):
        width, height = image.size
        short, long = (width, height) if width <= height else (height, width)
        new_short = self.size
        new_long = int(self.size * long / short)
        new_width, new_height = (
            (new_short, new_long)
            if width <= height
            else (new_long, new_short)
        )
        return image.resize((new_width, new_height), self.interpolation)


def build_transform(cfg, is_train=True, choices=None):
    if cfg.INPUT.NO_TRANSFORM:
        print("Note: no transform is applied!")
        return None
    choices = list(cfg.INPUT.TRANSFORMS if choices is None else choices)
    size = tuple(cfg.INPUT.SIZE)
    interpolation = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    operations = []
    if is_train:
        print("Building transform_train")
        if "random_resized_crop" in choices:
            print(f"+ random resized crop (size={size}, scale={cfg.INPUT.RRCROP_SCALE})")
            operations.append(
                transform.RandomResizedCrop(
                    size,
                    scale=tuple(cfg.INPUT.RRCROP_SCALE),
                    interpolation=interpolation,
                )
            )
        else:
            print(f"+ resize to {size[0]}x{size[1]}")
            operations.append(transform.Resize(size, mode=interpolation))
        if "random_flip" in choices:
            print("+ random flip")
            operations.append(transform.RandomHorizontalFlip())
    else:
        print("Building transform_test")
        print(f"+ resize the smaller edge to {max(size)}")
        operations.append(ResizeShortEdge(max(size), interpolation))
        print(f"+ {size[0]}x{size[1]} center crop")
        operations.append(transform.CenterCrop(size))
    print("+ to Jittor-compatible tensor of range [0, 1]")
    operations.append(transform.ToTensor())
    if "normalize" in choices:
        print(f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})")
        operations.append(transform.ImageNormalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD))
    return transform.Compose(operations)
