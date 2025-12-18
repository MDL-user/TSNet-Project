# --- dataloader.py (Model 2.0 - All Data K-Fold 兼容) ---
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import pandas as pd
import numpy as np


class imgDataset(Dataset):
    def __init__(self, transform=None, split='train', fps=16, qt=20):

        self.split = split
        self.root = 'D://data_new/'
        self.transform = transform
        list_id = []

        # 🚨 关键修改 1: 确保 split='full' 包含 train, val, test 三个分区
        if split == 'full':
            base_splits = ['train', 'val', 'test']
        elif split in ['train', 'val', 'test']:
            base_splits = [split]
        else:
            base_splits = []

        valid_extensions = ('.png', '.jpg', '.jpeg')

        for current_split in base_splits:
            base_dir = os.path.join(self.root, 'images', current_split)

            if os.path.exists(base_dir):
                for dirpath, _, filenames in os.walk(base_dir):
                    # 确定类别标签和患者ID
                    target = -1
                    if 'jixing' in dirpath:
                        target = 2
                    elif 'manji' in dirpath:
                        target = 1
                    elif 'manxing' in dirpath:
                        target = 0

                    # 确保文件夹名是患者ID
                    patient_id = os.path.split(dirpath)[1]

                    if target != -1:
                        for filename in filenames:
                            is_valid = False
                            for ext in valid_extensions:
                                if filename.lower().endswith(ext):
                                    is_valid = True
                                    break

                            if is_valid:
                                full_path = os.path.join(dirpath, filename)
                                # [完整路径, 文件名, 标签, 患者ID]
                                list_id.append([full_path, filename, target, patient_id])

        self.fnames = list_id

        # --- 添加表格数据 (为融合模型准备，Model 2.0 不使用) ---
        xlsx = pd.read_excel(r'C:\Users\admin\Desktop/data-3.xlsx')
        xlsx['性别'] = xlsx['性别'].map({'男': 0, '女': 1}).values
        xlsx['年龄'] = xlsx['年龄'].values / 100.0
        xlsx['白细胞计数（10^9/L）'] = xlsx['白细胞计数（10^9/L）'].values / 10.0
        xlsx['中性分叶核粒细胞百分数(%)'] = xlsx['中性分叶核粒细胞百分数(%)'].values / 100.0
        time_futong = xlsx['腹痛时间（小时）'].tolist()
        time_futong = [float(str(s).replace('+', '')) for s in time_futong]
        xlsx['腹痛时间（小时）'] = np.log(time_futong)

        self.patient_id = [str(s) for s in xlsx.values[:, 0].astype(np.longlong).tolist()]
        self.patient_info = xlsx.values[:, 1:]
        self.NUM_TABULAR_FEATURES = self.patient_info.shape[1]
        print('tabular-data-shape:', xlsx.shape, 'Features:', self.NUM_TABULAR_FEATURES)

        # 定义 Mask 专用变换：只进行 Resize 和 ToTensor
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        # 图像的 Normalize 步骤
        self.img_normalize = transforms.Normalize(mean=[0.1268, 0.1276, 0.1285], std=[0.1778, 0.1785, 0.1800])

    def load_img_mask(self, path):
        img = Image.open(path[0]).convert('RGB')

        # --- Mask 路径推导 (保持不变) ---
        normalized_img_path = os.path.normpath(path[0])
        mask_path_base = normalized_img_path.replace(os.path.join('images'), os.path.join('masks'))
        mask_path_dir = os.path.dirname(mask_path_base)
        first_char = path[1][0]
        mask_filename_with_suffix = first_char + '_filled.png'
        mask_path = os.path.join(mask_path_dir, mask_filename_with_suffix)

        # ⚠️ 确保 mask 文件存在，否则可能抛出异常
        try:
            mask = Image.open(mask_path).convert('L')
        except FileNotFoundError:
            # 如果 mask 不存在，返回一个全零的假 mask
            print(f"Warning: Mask file not found at {mask_path}. Returning dummy mask.")
            mask = Image.new('L', (img.width, img.height), 0)

        # 1. 应用图像几何变换
        if self.transform:
            # transforms.ToTensor() 必须是最后一个变换，因此这里如果包含 ToTensor，则要小心
            # 确保 self.transform 只包含几何/数据增强，不包含 ToTensor 和 Normalize
            #
            # 检查 self.transform 的最后一个元素是否是 ToTensor
            if isinstance(self.transform.transforms[-1], transforms.ToTensor):
                img = self.transform(img)
            else:
                # 如果没有 ToTensor，先进行转换
                img = transforms.ToTensor()(img)
                # 由于您提供的 transform 已经包含 ToTensor，所以这里保持原样，
                # 假设 transform 在 if __name__ == '__main__': 中是正确的
                img = self.transform(img)
                # 为了防止双重 ToTensor，这里调整一下逻辑，避免在函数内重复调用 ToTensor

                # 重新执行图像几何变换，并确保是 PIL Image
                img = Image.open(path[0]).convert('RGB')
                img = self.transform(img)  # 此时 img 是 Tensor 或 PIL Image

        # 2. 对 Mask 显式执行 Resize+ToTensor
        mask_tensor = self.mask_transform(mask)

        # 3. 应用 Normalize 和确保 img 是 Tensor
        if isinstance(img, Image.Image):
            img_tensor = transforms.ToTensor()(img)
        else:
            # 假设 img 已经是 Tensor (来自 self.transform 的 ToTensor)
            img_tensor = img

        img_tensor = self.img_normalize(img_tensor)

        # 确保 img_tensor 是 3D (C, H, W)
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        paths, target, id = self.fnames[index][0:2], self.fnames[index][2], self.fnames[index][3]
        img_tensor, mask_tensor = self.load_img_mask(paths)

        # 提取病人表格信息
        try:
            index_id = self.patient_id.index(id)
        except:
            index_id = -1

        if index_id < 0:
            patient_info = np.zeros(shape=(self.NUM_TABULAR_FEATURES,))
        else:
            patient_info = self.patient_info[index_id]

        info_tensor = torch.from_numpy(patient_info).reshape(-1).float()

        # 返回 (img_tensor, info_tensor, mask_tensor, target, id)
        return img_tensor, info_tensor, mask_tensor, target, id