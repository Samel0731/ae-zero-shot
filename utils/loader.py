import os
import glob
import itertools
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import io

class AEDataset(Dataset):
    def __init__(self, root_dir, train=False, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.image_files[index])
        image = Image.open(img_path).convert('L')   # Convert to grayscale ('L' mode)

        if self.transform:
            image = self.transform(image)
        return image, image


class KFoldDataset(Dataset):
    def __init__(self, root_dir, field_names, k=0, fold=5, train=False, transform=None, random_state=403, use_torchio=False):
        super().__init__()
        self.root_dir = root_dir
        self.field_names = field_names
        self.transform = transform
        self.k = k  # Current fold index
        self.fold = fold    # Number of folds
        self.train = train
        self.use_torchio = use_torchio

        # Set random seed for reproducibility
        self.random_state = random_state
        np.random.seed(random_state)

        self.image_files, self.field_idxs = self._get_images_dir(root_dir, field_names)
        self.train_indices, self.val_indices = self._get_kfold_split()
        self.data_indices = self.train_indices if self.train else self.val_indices

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        actual_index = self.data_indices[index]
        # img_path = str(self.image_files[actual_index])
        lbl_path = str(self.image_files[actual_index]).replace('images', 'labels')
        field_idx = self.field_idxs[actual_index]

        # image = Image.open(img_path).convert('L')   # Convert to grayscale ('L' mode)
        label = Image.open(lbl_path).convert('L')   # Convert to grayscale ('L' mode)        
        if self.use_torchio:
            # image = io.read_image(img_path)
            label = io.read_image(lbl_path)
            label = transforms.Grayscale()(label)
        if self.transform:
            # image, label = self.transform(image, label)
            label = self.transform(label)

        return label, label, field_idx

    def _get_images_dir(self, root_path, field_names) -> list:
        number_fields = len(field_names)
        np.random.seed(self.random_state)

        # Pre-allocate field_img array
        field_img = [None] * number_fields

        # Use list comprehension for more efficient file listing and shuffling
        for i, field_name in enumerate(field_names):
            field_path = os.path.join(root_path, 'train_data', field_name)
            images = glob.glob(os.path.join(field_path, 'images', '*.png'))
            np.random.shuffle(images)  # Shuffle the list in-place
            field_img[i] = images

        # Determine minimum number of images available across all fields
        min_images = min(len(img_list) for img_list in field_img)

        # Use itertools to efficiently flatten the shuffled images
        dataset_dir = list(itertools.chain.from_iterable(zip(*[field_img[i][:min_images] for i in range(number_fields)])))
        dataset_field_id = list(itertools.chain.from_iterable(zip(*[np.full(min_images, i) for i,_ in enumerate(field_names)])))

        return dataset_dir, dataset_field_id
    
    def _get_kfold_split(self):
        kf = KFold(n_splits=self.fold, shuffle=False)
        assert len(self.image_files)>0, f"there are no data images in {self.root_dir}"
        indices = np.arange(len(self.image_files))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            if fold_idx == self.k:
                return train_idx, val_idx
        
        raise ValueError(f"Fold {self.k} is out of range for the provided fold number {self.fold}")
