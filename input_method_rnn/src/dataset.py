from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from config import * 

# 1. 定义Dataset类
class InputMethodDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['input'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['target'], dtype=torch.long)
        return input_tensor, target_tensor
        
# 2. 提供一个获取dataloader的方法
def get_dataloader(train=True):
    path = PROCESSED_DATA_DIR / ('train_dataset.jsonl' if train else 'test_dataset.jsonl')
    dataset = InputMethodDataset(path)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. 

if __name__ == "__main__":
    # 测试Dataset和DataLoader
    # torch.utils.data.DataLoader 对象
    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)
    print(len(train_loader))
    print(len(test_loader))
    
    for input_tensor, target_tensor in train_loader:
        print("输入张量形状:", input_tensor.shape) # [batch_size, seq_length]
        print("目标张量形状:", target_tensor.shape) # [batch_size]
        break