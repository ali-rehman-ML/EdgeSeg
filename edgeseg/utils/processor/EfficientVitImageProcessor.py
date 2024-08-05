

def prepare_input_torch(pil_image, crop_size=512):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    import torchvision.transforms.functional as F
    from torchvision import transforms

    class Resize:
        def __init__(self, crop_size, interpolation=cv2.INTER_CUBIC):
            self.crop_size = crop_size
            self.interpolation = interpolation

        def __call__(self, feed_dict):
            if self.crop_size is None or self.interpolation is None:
                return feed_dict

            image, target = feed_dict["data"], feed_dict["label"]
            height, width = self.crop_size

            h, w, _ = image.shape
            if width != w or height != h:
                image = cv2.resize(
                    image,
                    dsize=(width, height),
                    interpolation=self.interpolation,
                )
            return {
                "data": image,
                "label": target,
            }

    class ToTensor:
        def __init__(self, mean, std, inplace=False):
            self.mean = mean
            self.std = std
            self.inplace = inplace

        def __call__(self, feed_dict):
            image, mask = feed_dict["data"], feed_dict["label"]
            image = image.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
            image = torch.as_tensor(image, dtype=torch.float32).div(255.0)
            mask = torch.as_tensor(mask, dtype=torch.int64)
            image = F.normalize(image, self.mean, self.std, self.inplace)
            return {
                "data": image,
                "label": mask,
            }

    image = np.array(pil_image.convert("RGB"))
    transform = transforms.Compose(
        [
            Resize((crop_size, crop_size * 2)),
            ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data = transform({"data": image, "label": np.ones_like(image)})["data"]
    data = torch.unsqueeze(data, dim=0).cpu()
    return data



def prepare_input_numpy(pil_image, crop_size=512):
    import cv2
    import numpy as np
    from PIL import Image
    def resize(image, target_shape, interpolation=cv2.INTER_CUBIC):
        height, width = target_shape
        h, w, _ = image.shape
        if width != w or height != h:
            image = cv2.resize(
                image,
                dsize=(width, height),
                interpolation=interpolation,
            )
        return image

    def to_tensor(image, mean, std):
        # Convert image to float and normalize
        image = image.astype(np.float32) / 255.0
        # Normalize the image
        image -= mean
        image /= std
        # Convert to channel-first format
        image = image.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        # Convert to a PyTorch tensor
        return image

    # Convert PIL Image to numpy array
    image = np.array(pil_image.convert("RGB"))

    # Resize the image
    resized_image = resize(image, (crop_size, crop_size * 2))

    # Define mean and standard deviation for normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert to tensor and normalize
    tensor = to_tensor(resized_image, mean, std)

    # Add batch dimension
    tensor = np.expand_dims(tensor, dim=0)

    return tensor


def prepare_input(image,crop_size=512,type='torch'):
    if type=='torch':
        return prepare_input_torch(image,crop_size=crop_size)
    elif type=='numpy':
        return prepare_input_numpy(image,crop_size=crop_size)
    else:
        print("Inavlid process type ")



