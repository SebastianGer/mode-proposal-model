from torchvision.datasets import Cityscapes


class CityscapesDataset(Cityscapes):
    """
    Cityscapes dataset class with an additional index return.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, target, index

    def __len__(self):
        return super().__len__()
