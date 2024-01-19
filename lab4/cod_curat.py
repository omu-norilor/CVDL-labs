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
        # TODO your code here: if necessary download and extract the data

        if download:
            self.download_resources(base_folder)

        
        self.X = [cv2.imread(file) for file in glob.glob(os.path.join(base_folder, 'lfw_funneled', '*/*'))]
        self.Y = [cv2.imread(file) for file in glob.glob(os.path.join(base_folder, 'parts_lfw_funneled_gt_images', '*/*'))]

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.Y[idx]

        # Apply transformations if provided
        if self.transforms:
            x = self.transforms(x)
            y = self.transforms(y)

        return x, y



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


def test_semantic_segmentation_pets(batch_size, num_workers):
    # TODO your code here: return a DataLoader instance for the training set

    # test_data = LFWDataset(download=False, base_folder='lfw_dataset', transforms=None)
    test_data = torchvision.datasets.OxfordIIITPet(root='/content/cvdl_lab_4', split="test", target_types="segmentation", download=False,
                                               transforms= v2.Compose([
                                                            v2.Resize(256),
                                                            v2.CenterCrop(224),
                                                            v2.ToTensor()]))
    bs = batch_size
    dataloader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=True, num_workers=num_workers)

    for i_batch, sample_batched in enumerate(dataloader):
        imgs = sample_batched[0]
        segs = sample_batched[1]

        rows, cols = bs, 2
        figure = plt.figure(figsize=(10, 10))

        for i in range(0, bs):
            figure.add_subplot(rows, cols, 2*i+1)
            plt.title('image')
            plt.axis("off")
            plt.imshow(imgs[i].numpy().transpose(1, 2, 0))

            figure.add_subplot(rows, cols, 2*i+2)
            plt.title('seg')
            plt.axis("off")
            plt.imshow(segs[i].numpy().transpose(1, 2, 0), cmap="gray")
        plt.show()
        # display the first 3 batches
        if i_batch == 2:
            break


def upsample_block(x, filters, f_size, stride=2):
    """
    x - the input of the upsample block
    filters - the number of filters to be applied
    size - the size of the filters
    """

    # Transposed convolution layer
    upsample_conv = torch.nn.ConvTranspose2d(
        in_channels=x.shape[1],
        out_channels=filters,
        kernel_size=f_size,
        stride=stride,
        padding=f_size // 2,
        output_padding=stride - 1  # Additional padding to handle output size
    )

    print("X= ",x.size(1)-1," * ",stride," - 2 * ",f_size," + ", f_size)
    # Batch normalization
    batch_norm = torch.nn.BatchNorm2d(filters)

    # ReLU activation
    relu = torch.nn.ReLU()

    # Apply transposed convolution
    x = upsample_conv(x)

    # Apply batch normalization
    x = batch_norm(x)

    # Apply ReLU activation
    x = relu(x)

    return x

def test_upsample_block():
    in_layer = torch.rand((32, 32, 128, 128))

    filter_sz = 4
    num_filters = 16

    for stride in [2, 4, 8]:
        x = upsample_block(in_layer, num_filters, filter_sz, stride)
        print('in shape: ', in_layer.shape, ' upsample with filter size ', filter_sz, '; stride ', stride, ' -> out shape ', x.shape)

if __name__ == '__main__':
    # test_semantic_segmentation_pets(batch_size=4, num_workers=0)
    test_upsample_block()
