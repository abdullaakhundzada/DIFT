from typing import Literal
import albumentations as alb
import json, cv2, os
from torch.utils.data import Dataset, DataLoader
import torch

def load_default_transform(
        mode: Literal["train", "test"],
        shape: tuple[int] = (32, 32),
        ) -> alb.Compose:
    """
    Creates an image transform object and returns according 
    to the given arguments and default parameters within the function.
    Input data format must be taken into account: no BGR2RGB or
    RGB2BGR conversion is done internally. So if conversion is needed,
    it must be done before the transformation.

    Args:
        mode: Literal["train", "test"]
            Which type of transform to choose. The train 
            transform includes more augmentation strategies
            for covering wider interval of the input distribution.
            The test mode is more suitable for validation and testing
            in which situation augmentation is not needed.
        shape: tuple[int] = (32, 32)
            The target shape of the images (H , W)
    Returns:
        albumentations.Compose:
            A combination of sequentially cascaded image transforms.
    """
    if mode == "train":
        return alb.Compose([
            alb.Resize(*shape),
            alb.HorizontalFlip(p=0.5),
            alb.RandomBrightnessContrast(p=0.8),
            alb.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            alb.Rotate(limit=10, p=0.5),
            alb.GaussianBlur(p=0.4),
            alb.Normalize(
                mean=[0.49186878, 0.48265391, 0.44717728], 
                std =[0.24697121, 0.24338894, 0.26159259]),
            alb.ToTensorV2()
            ])

    else:
        return alb.Compose([
            alb.Resize(*shape),
            alb.Normalize(
                mean=[0.49186878, 0.48265391, 0.44717728], 
                std =[0.24697121, 0.24338894, 0.26159259]),
            alb.ToTensorV2(),
            ])

class CIFARDataset(Dataset):
    def __init__(
            self, 
            image_directory : str, 
            labels_path     : str,
            transform       : alb.Compose
            )       -> None :
        """
        A wrapper class for Torch Dataset. Specifically designed for 
        the CIFAR dataset. 
        The images are expected to be in a folder, whose path is passed 
        as `image_directory` argument. Labels are expected to be stored
        in a JSON file and the path should be passed as `labels_path` 
        argument. Label consistency is automatically checked.

        Args:
            image_directory : str
                Path to the directory containing the images (`.png` or `.jpg`)
            labels_path : str
                The path to the JSON file containing the labels. The image
                file names must be the keys, and the labels must be the values.
            transform: albumentations.Compose
                A transform object to alter an initial image for further processing.
        
        Returns: 
            None
        """
        assert labels_path.endswith(".json"), \
            "Labels must be stored in a JSON file format."

        super().__init__()
        self.directory   = image_directory
        with open(labels_path, "r") as labels_file:
            self.labels_dict = json.load(labels_file)

        self.filenames = [os.path.join(image_directory, filename) 
            for filename in os.listdir(image_directory)
            if filename.endswith("jpg") or filename.endswith("png")]
        
        assert all(
            [filename in self.labels_dict.keys() for filename in self.filenames]), \
            "Not all images have labels given in the labels {} file.".format(labels_path)
        
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """
        Returns:
            Transformed image in torch Tensor format. 
            Label as an interger.
            Image filename as a string
        """
        image_filename  = self.filenames[idx]
        image_file_path = os.path.join(self.directory, image_filename)

        label = self.labels_dict[image_filename]

        image = cv2.imread(image_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed_image = self.transform(image=image)["image"]

        return transformed_image, label, image_filename