import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class PACSDataloader(Dataset):
    def __init__(self, src_path, meta_filenames, domain_label = -1):
        self.image_transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.src_path = src_path
        self.domain_label = domain_label
        self.sample_paths, self.class_labels, self.domain_labels = self.set_samples_labels(meta_filenames)
        
    def set_samples_labels(self, meta_filenames):
        sample_paths, class_labels, domain_labels = [], [], []
        for idx_domain, meta_filename in enumerate(meta_filenames):
            column_names = ['filename', 'class_label']
            data_frame = pd.read_csv(meta_filename, header = None, names = column_names, sep='\s+')
            sample_paths.extend(data_frame["filename"])
            class_labels.extend(data_frame["class_label"] - 1)
            domain_labels.extend([idx_domain] * len(data_frame))
            
        return sample_paths, class_labels, domain_labels

    def get_image(self, sample_path):
        img = Image.open(sample_path).convert('RGB')
        return self.image_transformer(img)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        sample = self.get_image(self.src_path + self.sample_paths[index])
        class_label = self.class_labels[index]
        domain_label = self.domain_label
        
        return sample, class_label, self.domain_label

class PACS_Test_Dataloader(PACSDataloader):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        def get_val_transformer():
            image_size = 224
            img_tr = [transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            return transforms.Compose(img_tr)
        self._image_transformer = get_val_transformer()

    def __getitem__(self, index):
        sample = self.get_image(self.src_path + self.sample_paths[index])
        class_label = self.class_labels[index]
        domain_label = self.domain_labels[index]
        return sample, class_label, domain_label