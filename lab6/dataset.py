import os
import cv2
import wget
import glob
import wandb
import shutil
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import hashlib
import tarfile
import requests
from tqdm import tqdm



class LFWDataset(torch.utils.data.Dataset):
    _DATA = (
        # images
        ("http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz", None),
        # segmentation masks as ppm
        ("https://vis-www.cs.umass.edu/lfw/part_labels/parts_lfw_funneled_gt_images.tgz",
         "3e7e26e801c3081d651c8c2ef3c45cfc"),
    )


    def __init__(self, base_folder, transforms, download=True, split_name: str = 'train'):
        super().__init__()
        self.base_folder = base_folder
        self.transforms = transforms
                # TODO your code here: if necessary download and extract the data

        if download:
            self.download_resources(base_folder)
        
        self.X = []
        self.Y = []
        
        # Get the list of file names from self.X and self.Y
        names_X = [os.path.splitext(os.path.basename(file))[0].split('_')[:2] for file in glob.glob(os.path.join(base_folder, 'lfw_funneled', '*/*'))]
        names_Y = [os.path.splitext(os.path.basename(file))[0].split('_')[:2] for file in glob.glob(os.path.join(base_folder, 'parts_lfw_funneled_gt_images', '*.ppm'))]

        # Create a list of names in the format <first_name>_<last_name>
        names_X = [name[0] + "_" + name[1] for name in names_X]
        names_Y = [name[0] + "_" + name[1] for name in names_Y]

        # Convert the lists to sets
        set_names_X = set(names_X)
        set_names_Y = set(names_Y)

        # Find the common names using set intersection
        common_names = set_names_X & set_names_Y

        X_ap=dict()

        for file in glob.glob(os.path.join(base_folder, 'lfw_funneled', '*/*')):
            name = os.path.splitext(os.path.basename(file))[0].split('_')[:2]
            name = f"{name[0]}_{name[1]}"
            if(name not in X_ap):
                X_ap[name]=0
            X_ap[name]+=1

        Y_ap=dict()
        for file in glob.glob(os.path.join(base_folder, 'parts_lfw_funneled_gt_images', '*.ppm')):
            name = os.path.splitext(os.path.basename(file))[0].split('_')[:2]
            name = f"{name[0]}_{name[1]}"
            if(name not in Y_ap):
                Y_ap[name]=0
            Y_ap[name]+=1

        for file in glob.glob(os.path.join(base_folder, 'lfw_funneled', '*/*')):
            name = os.path.splitext(os.path.basename(file))[0].split('_')[:2]
            name = f"{name[0]}_{name[1]}"
            if name in common_names and X_ap[name]==Y_ap[name]:
                self.X.append(cv2.imread(file))
  
        for file in glob.glob(os.path.join(base_folder, 'parts_lfw_funneled_gt_images', '*.ppm')):
            name = os.path.splitext(os.path.basename(file))[0].split('_')[:2]
            name = f"{name[0]}_{name[1]}"
            if name in common_names and X_ap[name]==Y_ap[name]:
                self.Y.append(cv2.imread(file, cv2.IMREAD_UNCHANGED))

        # convert to numpy arrays
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.X= self.X[..., ::-1]
        self.Y= self.Y[..., ::-1]

        # convert to tensors and
        self.X = torch.tensor(self.X.copy()).float().permute(0, 3, 1, 2)
        self.Y = torch.tensor(self.Y.copy()).float().permute(0, 3, 1, 2)
        
        # self.X = self.X.permute(0, 3, 1, 2)
        # self.Y = self.Y.permute(0, 3, 1, 2)

        # move to GPU if available
        if torch.cuda.is_available():
            self.X = self.X.cuda()
            self.Y = self.Y.cuda()

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.Y[idx]
        # Apply transformations if provided
        # if self.transforms:
        #     x = self.transforms(x)
        #     y = self.transforms(y)

        return x, y

    def __len__(self):
        return len(self.X)

    def download_resources(self, base_folder):
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        self._download_and_extract_archive(url=LFWDataset._DATA[1][0], base_folder=base_folder,
                                           md5=LFWDataset._DATA[1][1])
        self._download_and_extract_archive(url=LFWDataset._DATA[0][0], base_folder=base_folder, md5=None)

    def _download_and_extract_archive(self, url, base_folder, md5) -> None:
        """
          Downloads an archive file from a given URL, saves it to the specified base folder,
          and then extracts its contents to the base folder.

          Args:
          - url (str): The URL from which the archive file needs to be downloaded.
          - base_folder (str): The path where the downloaded archive file will be saved and extracted.
          - md5 (str): The MD5 checksum of the expected archive file for validation.
          """
        base_folder = os.path.expanduser(base_folder)
        filename = os.path.basename(url)

        self._download_url(url, base_folder, md5)
        archive = os.path.join(base_folder, filename)
        print(f"Extracting {archive} to {base_folder}")
        self._extract_tar_archive(archive, base_folder, True)

    def _retreive(self, url, save_location, chunk_size: int = 1024 * 32) -> None:
        """
            Downloads a file from a given URL and saves it to the specified location.

            Args:
            - url (str): The URL from which the file needs to be downloaded.
            - save_location (str): The path where the downloaded file will be saved.
            - chunk_size (int, optional): The size of each chunk of data to be downloaded. Defaults to 32 KB.
            """
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(save_location, 'wb') as file, tqdm(
                    desc=os.path.basename(save_location),
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=chunk_size):
                    file.write(data)
                    bar.update(len(data))

            print(f"Download successful. File saved to: {save_location}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def _download_url(self, url: str, base_folder: str, md5: str = None) -> None:
        """Downloads the file from the url to the specified folder

        Args:
            url (str): URL to download file from
            base_folder (str): Directory to place downloaded file in
            md5 (str, optional): MD5 checksum of the download. If None, do not check
        """
        base_folder = os.path.expanduser(base_folder)
        filename = os.path.basename(url)
        file_path = os.path.join(base_folder, filename)

        os.makedirs(base_folder, exist_ok=True)

        # check if the file already exists
        if self._check_file(file_path, md5):
            print(f"File {file_path} already exists. Using that version")
            return

        print(f"Downloading {url} to file_path")
        self._retreive(url, file_path)

        # check integrity of downloaded file
        if not self._check_file(file_path, md5):
            raise RuntimeError("File not found or corrupted.")

    def _extract_tar_archive(self, from_path: str, to_path: str = None, remove_finished: bool = False) -> str:
        """Extract a tar archive.

        Args:
            from_path (str): Path to the file to be extracted.
            to_path (str): Path to the directory the file will be extracted to. If omitted, the directory of the file is
                used.
            remove_finished (bool): If True , remove the file after the extraction.
        Returns:
            (str): Path to the directory the file was extracted to.
        """
        if to_path is None:
            to_path = os.path.dirname(from_path)

        with tarfile.open(from_path, "r") as tar:
            tar.extractall(to_path)

        if remove_finished:
            os.remove(from_path)

        return to_path

    def _compute_md5(self, filepath: str, chunk_size: int = 1024 * 1024) -> str:
        with open(filepath, "rb") as f:
            md5 = hashlib.md5()
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()

    def _check_file(self, filepath: str, md5: str) -> bool:
        if not os.path.isfile(filepath):
            return False
        if md5 is None:
            return True
        return self._compute_md5(filepath) == md5


if __name__ == '__main__':
    # Initialize the dataset
    dataset = LFWDataset(base_folder='data', transforms=None, download=False)

    # Display some images along with their segmentation masks
    num_samples_to_display = 5


    print(dataset[0][0])
    # exit()

    for i in range(num_samples_to_display):
        # Get a random index
        idx = np.random.randint(len(dataset))

        # Get the image and segmentation mask
        image, mask = dataset[idx]

        # convert to numpy arrays
        image = image.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        mask = mask.cpu().permute(1, 2, 0).numpy().astype(np.uint8)

        # Plot the image
        plt.subplot(2, num_samples_to_display, i + 1)
        plt.imshow(image)
        # plt.imshow(image, vmax=255)
        plt.title('Image')
        plt.axis('off')

        # Plot the segmentation mask
        plt.subplot(2, num_samples_to_display, i + 1 + num_samples_to_display)
        plt.imshow(mask)
        plt.title('Seg Mask')
        plt.axis('off')

    # Show the plot
    plt.show()
