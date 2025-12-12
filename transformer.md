# transformer

encoder-decoder 编码

seq2seq模型



-------------------

### encoder-decoder框架

对一个句子模型$X_1,X_2,X_3,...$使用encoder进行非线性编码，获得一个向量C（中间语义表示），decoder根据这个向量和之前生成的历史信息去生成另一个句子$Y_1,Y_2,Y_3,...$。需要注意的是$Y_i$除了受向量C的影响，还受到前序逐步生成的历史信息影响，即：$Y_i= Decoder(C, y_1,...,y_{i-1})$。

四种encoder模式：

![image-20251201092209041](/home/estelle/.config/Typora/typora-user-images/image-20251201092209041.png)

**encoder-decoder框架特点**

1. 是端到端的学习算法
2. 不论输入和输出的长度是什么，中间的向量C都是固定的（导致信息缺失问题）
3. 根据不同任务可以选择不同的编码器和解码器（可以是CNN、RNN、LSTM、GRU等）

**encoder-decoder框架应用**

基于encoder-decoder框架设计的模型可以应用于：机器翻译、对话机器人、诗词生成、代码补全、文章摘要...

**encoder-decoder框架的缺点**

基础encoder-decoder框架存在的最大问题在于信息缺失。

encoder将输入source编码为固定大小的向量的过程是一个“信息有损的压缩过程”，信息量越大，转化得到的固定向量中信息的损失就越大。

## Seq2seq

### 1.简介

seq2seq是一种重要的RNN模型，理解为输入\*输出 = N*M的模型。模型包含两个部分：Encoder是编码器，用于编码序列信息，将任意长度的信息编码到一个向量c里。decoder是解码器，将马得到上下文向量c后可以将信息解码，并输出为序列。

### 2.模型类别

三种seq2seq：seq2seq模型主要区别在于decoder。

encoder与RNN区别一般不大，只是中间神经元没有输出。其中的上下文向量c可以采用多种方式计算：
$$
\alpha_t = softmax(e_t)\\
\alpha_{ti} = \frac{exp(e_{ti})}{\sum_{i=1}^N \alpha_{ti}h_i}\\
c_t = \sum_{i=1}^N \alpha_{ti}h_i
$$
第一种：

![image-20251205143037922](/home/estelle/.config/Typora/typora-user-images/image-20251205143037922.png)

encoder计算公式：
$$
h'_1 = \sigma(W_c + b)\\
h'2 = \sigma(Wh'_{t-1} + b)\\
y't = \sigma(Vh'_t + c)
$$
第二种：

![image-20251205143111479](/home/estelle/.config/Typora/typora-user-images/image-20251205143111479.png)
$$
h't = \sigma(U_c + Wh'_{t-1} + b)\\
y't = \sigma(Vh'_t + c)
$$


第三种：

![image-20251205143132079](/home/estelle/.config/Typora/typora-user-images/image-20251205143132079.png)
$$
h'_t = \sigma(U_c + Wh'_{t-1} + Vy'_{t-1} + b)\\
y'_t = \sigma(Vh'_t + c)
$$

### 3.模型使用技巧

#### 3.1 teacher forcing

teach forcing 用于训练阶段，针对上面三种decoder模型，第三种decoder模型神经元的输入包括了上一个神经元的输出$y'$。如果上一个神经元的输出是错误的，则下一个神经元的输出也很容易错误，导致错误会一直传递下去。

而teacher forcing可以在一定程度上缓解问题，在训练Seq2Seq模型时，decoder的每一个神经元并非一定用上一个神经元的输出，而是有一定比例采用正确的序类作为输入。

代码：
```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# 固定随机种子
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ====================== 1. 定义模型组件 ======================
class Encoder(nn.Module):
    """编码器：将输入序列编码为上下文向量"""
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # LSTM 编码器（输入：embedding，输出：隐藏状态+细胞状态）
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_seq_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_seq_len, emb_dim]
        # LSTM 前向传播
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [batch_size, src_seq_len, hid_dim]（所有时间步输出）
        # hidden/cell: [n_layers, batch_size, hid_dim]（最后一层的隐藏状态）
        return hidden, cell

class Decoder(nn.Module):
    """解码器：自回归模型（输入依赖上一步输出）"""
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # 嵌入层（目标序列嵌入）
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # LSTM 解码器（输入：上一步embedding，初始状态：编码器的hidden/cell）
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # 输出层：将隐藏状态映射到目标词汇表维度
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):
        # input: [batch_size]（单个时间步的输入，如前一步的预测/真实值）
        # hidden/cell: [n_layers, batch_size, hid_dim]
        
        # 增加序列长度维度（LSTM需要seq_len维度）
        input = input.unsqueeze(1)  # [batch_size, 1]
        
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]
        
        # LSTM前向传播（输入：embedded，初始状态：hidden/cell）
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: [batch_size, 1, hid_dim]
        # hidden/cell: [n_layers, batch_size, hid_dim]
        
        # 移除seq_len维度，映射到输出维度
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    """完整的Seq2Seq模型（Encoder + Decoder）"""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # 确保编码器和解码器的隐藏维度/层数一致
        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions must match!"
        assert encoder.n_layers == decoder.n_layers, "Number of layers must match!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        参数：
            src: 输入序列 [batch_size, src_seq_len]
            trg: 目标序列 [batch_size, trg_seq_len]
            teacher_forcing_ratio: 使用真实目标的概率
        返回：
            outputs: 所有时间步的预测 [batch_size, trg_seq_len-1, output_dim]
        """
        batch_size = trg.shape[0]
        trg_seq_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # 初始化输出张量（存储每个时间步的预测）
        outputs = torch.zeros(batch_size, trg_seq_len-1, trg_vocab_size).to(self.device)
        
        # 编码器编码输入序列，得到初始隐藏状态和细胞状态
        hidden, cell = self.encoder(src)
        
        # 解码器初始输入：目标序列的第一个token（如<sos>开始符）
        input = trg[:, 0]  # [batch_size]
        
        # 遍历目标序列的每个时间步（从第1步到最后一步）
        for t in range(1, trg_seq_len):
            # 解码器前向传播（输入：上一步的输入，隐藏状态，细胞状态）
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # 存储当前时间步的预测
            outputs[:, t-1, :] = output
            
            # 决定是否使用Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 选择当前时间步的预测（取概率最大的token）
            top1 = output.argmax(1)  # [batch_size]
            
            # 下一个输入：如果用Teacher Forcing则用真实值，否则用预测值
            input = trg[:, t] if teacher_force else top1
        
        return outputs

# ====================== 2. 训练函数（带Teacher Forcing） ======================
def train(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio=0.5):
    """单轮训练函数"""
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        src = batch[0].to(model.device)  # 输入序列
        trg = batch[1].to(model.device)  # 目标序列
        
        optimizer.zero_grad()
        
        # 前向传播（带Teacher Forcing）
        output = model(src, trg, teacher_forcing_ratio)
        
        # 计算损失：
        # output: [batch_size, trg_seq_len-1, output_dim]
        # trg: [batch_size, trg_seq_len] → 取trg[:,1:]（去掉<sos>）
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)  # [batch_size*(trg_seq_len-1), output_dim]
        trg = trg[:, 1:].reshape(-1)  # [batch_size*(trg_seq_len-1)]
        
        loss = criterion(output, trg)
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# ====================== 3. 测试代码（模拟数据） ======================
if __name__ == "__main__":
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 超参数设置
    INPUT_DIM = 100  # 输入词汇表大小
    OUTPUT_DIM = 100  # 输出词汇表大小
    EMB_DIM = 32  # 嵌入维度
    HID_DIM = 64  # 隐藏层维度
    N_LAYERS = 2  # LSTM层数
    DROPOUT = 0.5  # dropout率
    TEACHER_FORCING_RATIO = 0.5  # Teacher Forcing概率
    LEARNING_RATE = 0.001
    N_EPOCHS = 10
    CLIP = 1  # 梯度裁剪阈值
    
    # 初始化编码器/解码器/Seq2Seq模型
    encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # 优化器和损失函数（忽略<pad>填充符的损失）
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 假设0是<pad>的索引
    
    # 模拟训练数据（批量大小=8，输入序列长度=5，目标序列长度=4）
    batch_size = 8
    src_seq_len = 5
    trg_seq_len = 4
    src = torch.randint(1, INPUT_DIM, (batch_size, src_seq_len)).to(device)  # 输入序列（避开<pad>）
    trg = torch.randint(1, OUTPUT_DIM, (batch_size, trg_seq_len)).to(device)  # 目标序列（避开<pad>）
    trg[:, 0] = 2  # 强制第一个token为<sos>开始符
    
    # 模拟数据迭代器
    train_iterator = [(src, trg) for _ in range(10)]  # 模拟10个batch
    
    # 开始训练
    for epoch in range(N_EPOCHS):
        train_loss = train(
            model, train_iterator, optimizer, criterion, 
            CLIP, teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}')
```

构建真实数据集：

```python
from torch.utils.data import Dataset, DataLoader

# 1. 定义自定义数据集
class Seq2SeqDataset(Dataset):
    def __init__(self, src_data, trg_data):
        self.src_data = src_data  # 真实输入序列列表
        self.trg_data = trg_data  # 真实目标序列列表
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return self.src_data[idx], self.trg_data[idx]

# 2. 加载真实数据（示例：假设已有src_list/trg_list）
src_list = [torch.randint(1, 100, (5,)) for _ in range(80)]  # 80个输入序列
trg_list = [torch.randint(1, 100, (4,)) for _ in range(80)]  # 80个目标序列
dataset = Seq2SeqDataset(src_list, trg_list)

# 3. 构建真实的训练迭代器（按批次加载，自动打乱）
train_iterator = DataLoader(
    dataset,
    batch_size=8,  # 每个批次8条数据
    shuffle=True   # 训练前打乱数据
)
```

结合之前的`train()`函数，有这样的循环：

```python
for batch in iterator:
    src = batch[0].to(model.device)
    trg = batch[1].to(model.device)
    # ... 训练逻辑 ...
```

