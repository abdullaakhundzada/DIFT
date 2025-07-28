import requests, tarfile, pickle, io, os, cv2, json
import numpy as np
from typing import Optional

def devectorize(
        image: np.ndarray | list[np.ndarray], 
        shape: tuple = (3, 32, 32) 
        ) -> np.ndarray | list[np.ndarray]:
    """
    Reshapes the given vectorized image back into CV2 3D image format.
    Also compatable with batch input in numpy.ndarray or list format.

    Args:
        image: np.ndarray
            The input vector or a batch of vectors to process
        shape: tuple = (3, 32, 32) 
            The dimensions of the image before vectorization
    
    Returns:
        np.ndarray | list[np.ndarray]
            np.ndarray: the image or images after reshaping
            list[np.ndarray]: batch of images if the input is a 
                batch of vectors
    """
    if type(image) == np.ndarray:

        if len(image.shape) == 1:
            return image            \
                .reshape  ( shape ) \
                .transpose(1, 2, 0) 

        elif len(image.shape) == 2:
            return np.array(
                [
                    devectorize(image_sample, shape) 
                    for image_sample in image
                ]
            )
        
        else:
            raise ValueError(
                "Dimensions of a vector or a " \
                    "batch of vectors should not exceed 2"
            )

    elif type(image) == list:
        return [
            devectorize(image_sample, shape) 
            for image_sample in image
        ]
    
    else:
        raise ValueError(
            "Input shape and/or type does not " \
                "meet the expectatations."
        )

def load_cifar_dataset(
        url : str ="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        ) -> dict:
    """
    Loads the CIFAR-10 dataset into the memory. 
    Automatically converts the images into OpenCV-compatible format.

    Args:
        url : str
            Link to the dataset. Optional, default is:
                "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    
    Returns:
        dict: a dictionary of dictionaries. There are two inner 
            dictionaries: train and test. Each of them contain 
            training images and labels in numpy NDArray format.
    """

    response = requests.get(url)
    response.raise_for_status()

    tar_file_buffer = io.BytesIO(response.content)

    cifar_dataset = {
        "train" : {
            "images" : None, 
            "labels" : None,
            "files"  : []
        },
        "test"  : {
            "images" : None, 
            "labels" : None,
            "files"  : []
        }
    }

    with tarfile.open(fileobj=tar_file_buffer, mode='r:gz') as tar:
        for member in tar.getmembers():
            member_file_name = os.path.basename(member.name)
            
            if member.isfile() and "_batch" in member_file_name:
                file_obj = tar.extractfile(member=member)

                if not file_obj: continue

                data   = pickle.load(file_obj, encoding="bytes")

                labels = np.array(data[b"labels"])
                images = devectorize(data[b"data"])
                files  = [filename.decode() for filename in data[b"filenames"]]

                batch  = cifar_dataset[
                    "test" if "test" in member_file_name else "train"]
                
                batch["images"] = images if batch["images"] is None \
                    else np.vstack((batch["images"], images))
                
                batch["labels"] = labels if batch["labels"] is None \
                    else np.hstack((batch["labels"], labels))
                
                batch["files"].extend(files)

                print("Loaded {}\tdata into the dataset dictionary" \
                    .format(member_file_name))

    del response, tar_file_buffer

    return cifar_dataset

def store_cifar(cifar_dataset : dict, directory : str) -> None:
    """
    Saves the CIFAR dataset into the disk.

    Args:
        cifar_dataset : dict
            The dataset formatted as a dictionary. Should
            follow the structure of the load_cifar_dataset function.
            Must have the foillowing hierarchy:
                {
                    "train" : {
                        "images" : numpy.ndarray, 
                        "labels" : numpy.ndarray,
                        "files"  : list[str]
                    },
                    "test"  : {
                        "images" : numpy.ndarray, 
                        "labels" : numpy.ndarray,
                        "files"  : list[str]
                    }
                }
        directory : str
            Path to the directory where the images and 
            the labels will be stored in.
    
    Returns:
        None
    """
    assert cifar_dataset.keys(), "Empty dataset dictionary has been passed!"
    # If the files already exist, then the files will not be re-written
    if all([os.path.exists(os.path.join(directory, key)) 
            for key in cifar_dataset.keys()]):
        return 
    
    for sub_n in cifar_dataset.keys():
        sub_directory = os.path.join(directory, sub_n)
        sub_dataset = cifar_dataset[sub_n]
        if not os.path.exists(sub_directory):
            os.makedirs(sub_directory)
        
        for image, filename in zip(sub_dataset["images"], sub_dataset["files"]):
            cv2.imwrite(filename=os.path.join(sub_directory, filename), img=image)

        label_dict = {
            filename : int(label) for filename, label in 
            zip(sub_dataset["files"], sub_dataset["labels"])}
        
        with open(os.path.join(directory, "{}.json".format(sub_n)), "w") as json_file:
            json.dump(label_dict, json_file)

def split_dataset(
        cifar_dataset : dict, 
        ratio         : Optional[dict] = None, 
        shuffle       : bool = True
        )     -> dict :
    """
    Splits the dataset into multiple parts according to the given
    ratio dictionary.

    Args:
        cifar_dataset: dict
            The dataset given as a collection of sub-datasets
            under different keys, i. e. "train", "validate".
            Each sub-dataset must have "images", "labels" and "files" keys.
        ratio: Optional[dict]
            The final ratio of the sub-datasets. New keys can be added
            other than the ones of the cifar_dataset argument.
        shuffle : bool 
            Whether to shuffle the indices or not. 
            Optional, default is `True`

    Returns:
        dict:
            The structure is almost the same as the cifar_dataset 
            input dictionary; however, the values ,ay have been shuffled 
            and the output is the split version of the input dataset
            but split with approximate ratio.
    """
    images = None
    labels = None
    files  = []

    if ratio is None:
        ratio = {
            "train"    : 0.65,
            "val"      : 0.1,
            "test"     : 0.1,
            "finetune" : 0.15
        }

    # collect all portions of the dataset into monoblocks
    for sub_d in cifar_dataset.keys():
        sub_dataset = cifar_dataset[sub_d]
        images = sub_dataset["images"] if images is None else \
            np.vstack((images, sub_dataset["images"]))
        labels = sub_dataset["labels"] if labels is None else \
            np.hstack((labels, sub_dataset["labels"]))
        files.extend(sub_dataset["files"])

    files = np.array(files)

    assert len(images) == len(labels) == len(files), \
        "Sizes of image, label and filename arrays do not match!"

    if shuffle:
        indices = np.arange(len(labels))

        np.random.shuffle(indices)

        images = images[indices]
        labels = labels[indices]
        files  = files [indices]

    key_list = list(ratio.keys())

    ratio_sum = sum(ratio.values())
    ratio_normalized = {key: ratio[key]/ratio_sum for key in key_list}


    total_count = len(labels)
    # assign max values to each key, initially with the minimum index value 0
    max_idx = {key: 0 for key in key_list}

    for key in key_list:
        # each key must occupy an interval starting from the previous one
        max_idx[key] = max(max_idx.values()) + int(total_count * ratio_normalized[key])

    # if not all indexes are taken, assign the remainings to the last key
    max_idx[key_list[-1]] = total_count 

    # create tuples indicating the start and end of intervals
    intervals = {}
    for idx, key in enumerate(key_list):
        intervals[key] = (0 if idx == 0 else max_idx[key_list[idx-1]], max_idx[key])

    mean = np.mean(images / 255, axis=(0, 1, 2), keepdims=True)
    std  = np.sqrt(((images / 255 - mean) ** 2).mean(axis=(0, 1, 2), keepdims=True))

    print("For RGB Images:\nMean:\t{}\nSTD:\t{}".format(mean, std))

    return {
        key: {
            "images" : images[intervals[key][0] : intervals[key][1]],
            "labels" : labels[intervals[key][0] : intervals[key][1]],
            "files"  : files [intervals[key][0] : intervals[key][1]].tolist()
        } for key in key_list
    }
