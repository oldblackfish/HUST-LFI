import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import h5py


class MyTrainSetLoader_Kfold(Dataset):

    def __init__(self, dataset_dir, test_scene_id):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.to_tensor = ToTensor()

        scene_list = [
            'I01C00', 'I02C00', 'I03C00', 'I04C00', 'I05C00',
            'I06C00', 'I07C00', 'I08C00', 'I09C00', 'I10C00',
            'I11C00', 'I12C00', 'I13C00', 'I14C00', 'I15C00',
            'I16C00', 'I17C00', 'I18C00', 'I19C00', 'I20C00'
        ]

        # K-fold
        scene_list = [
            s for i, s in enumerate(scene_list)
            if i not in test_scene_id
        ]

        all_patch_path = []

        for scene in scene_list:
            scene_dir = os.path.join(dataset_dir, scene)

            for distorted_scene in os.listdir(scene_dir):

                distorted_dir = os.path.join(scene_dir, distorted_scene)

                for patch in os.listdir(distorted_dir):
                    all_patch_path.append(
                        os.path.join(scene, distorted_scene, patch)
                    )

        self.all_patch_path = all_patch_path

    def __getitem__(self, index):

        file_name = os.path.join(
            self.dataset_dir,
            self.all_patch_path[index]
        )

        with h5py.File(file_name, 'r') as hf:
            data = hf['data'][:].astype(np.float32) / 255.0
            score_label = hf['score_label'][()]
            cls = hf['cls'][()]
            level = hf['level'][()]

        data = self.to_tensor(data)
        score_label = torch.tensor(score_label, dtype=torch.float32)
        cls = torch.tensor(cls, dtype=torch.long)
        level = torch.tensor(level, dtype=torch.long)

        return data, score_label, cls, level

    def __len__(self):
        return len(self.all_patch_path)
