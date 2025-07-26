import requests, tarfile, pickle, io, os, cv2
import numpy as np

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
            return cv2.cvtColor(
                image                   \
                    .reshape  ( shape ) \
                    .transpose(1, 2, 0) , 
                cv2.COLOR_RGB2BGR
            ) 

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
            "files"  : None
        },
        "test"  : {
            "images" : None, 
            "labels" : None,
            "files"  : None
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
                
                batch["files" ] = files  if batch["files" ] is None \
                    else batch["files"].extend(files)

                print("Loaded {}\tdata into the dataset dictionary" \
                    .format(member_file_name))

    del response, tar_file_buffer

    return cifar_dataset