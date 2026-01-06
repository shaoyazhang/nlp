import torch
from dataset import get_dataloader
from model import InputMethodModel
from config import *
from datasets import tqdm

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    '''
    Docstring for train_one_epoch
    
    :param model: 模型
    :param dataloader: 数据集
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device: 设备
    :return: 当前epoch的平均损失
    '''
    model.train()
    total_loss = 0.0
    for inputs, targets in tqdm(dataloader, desc="训练进度"):
        inputs = inputs.to(device)  # [batch_size, seq_length]
        targets = targets.to(device)  # [batch_size]
        # 前向传播
        outputs = model(inputs)  # [batch_size, vocab_size]
        loss_value = loss_fn(outputs, targets)
        # 反向传播和优化
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss_value.item()
    return total_loss / len(dataloader)

def train():
    print("训练开始...")
    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 数据集
    dataloader = get_dataloader()
    
    # 3. 词表
    with open(MODELS_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f.readlines()]
        
    print(vocab_list[:5])
    # 4. 模型
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)  
    
    # 5. 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # 6. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 开始训练
    for epoch in range(EPOCHS):
        print('='*10, f"Epoch: {epoch+1}/{EPOCHS}", '='*10)
        # 训练一个epoch的逻辑
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"loss: {loss:.4f}")

    print("训练结束.")
    
    
if __name__ == "__main__":
    train()