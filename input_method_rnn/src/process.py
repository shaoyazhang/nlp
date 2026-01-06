import pandas as pd
from pathlib import Path
from config import *
from sklearn.model_selection import train_test_split
import jieba
from datasets import tqdm

# print('路径', __file__)

def build_dataset(sentences, word2index):
    
    indexed__sentences = [[word2index.get(token, 0) for token in jieba.lcut(sentence)] for sentence in sentences]
    
    dataset = []
    # [{'input': [1,2,3,4,5], 'target': 6}, {'input': [2,3,4,5,6], 'target': 7}, ...]
    for sentence in tqdm(indexed__sentences, desc="构建数据集"):
        for i in range(len(sentence) - SEQ_LENGTH):
            input = sentence[i:i+SEQ_LENGTH]
            target = sentence[i+SEQ_LENGTH]
            dict_ = {'input': input, 'target': target}
            dataset.append(dict_)
    return dataset

def process():
    print("开始处理数据...")
    # 1. 读取文件
    df = pd.read_json(RAW_DATA_DIR / 'synthesized_.jsonl', lines=True, orient='records').sample(frac=0.1)
    # print(df.head())
    # 2. 提取句子
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split("：")[1].strip())
            
    # 3. 划分训练集和测试集
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42) 
    
    # 4. 构建词表
    vocab_set = set()
    for sentence in tqdm(train_sentences, desc="构建词表"):
        vocab_set.update(jieba.lcut(sentence))
        
    vacab_list = ['<UNK>'] + list(vocab_set)
    print(f"词表大小: {len(vacab_list)}")
    
    # 5. 保存词表
    with open(MODELS_DIR / 'vocab.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(vacab_list))
        
    # 6. 构建训练集
    word2index = {word: idx for idx, word in enumerate(vacab_list)}
    train_dataset = build_dataset(train_sentences, word2index)
    # print(train_dataset[:3])
    
    # 7. 保存训练集
    pd.DataFrame(train_dataset).to_json(PROCESSED_DATA_DIR / 'train_dataset.jsonl', orient='records', lines=True)
    
    # 8. 构建测试集
    test_dataset = build_dataset(test_sentences, word2index)
    # 9. 保存测试集
    pd.DataFrame(test_dataset).to_json(PROCESSED_DATA_DIR / 'test_dataset.jsonl', orient='records', lines=True)
    
    print("数据处理完成。")
    
if __name__ == "__main__":
    process()