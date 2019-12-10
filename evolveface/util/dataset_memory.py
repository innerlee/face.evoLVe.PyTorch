from torchvision.datasets import VisionDataset


class DatasetMemory(VisionDataset):
    """
    Args:
        imgs (list of array)): List of images in the dataset
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.

     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """
    def __init__(self, imgs, transform=None):
        super().__init__("~")
        self.transform = transform
        self.samples = imgs
        if len(self.samples) == 0:
            raise (RuntimeError("No imgs found in the dataset"))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.samples[index]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, 0

    def __len__(self):
        return len(self.samples)
