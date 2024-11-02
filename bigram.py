import json
import pkuseg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

#----------------------------- 定义超参数 ------------------------

batch_size = 16  # 并行处理的数据数量
block_size = 8  # 输入seq的最大长度
lr = 1e-3
embd_size = 200
max_iters = 20000
eval_interval = 1000 # 每训练1000步评估一次
eval_iters = 200
if torch.cuda.is_available():  
    device = "cuda" 
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


torch.manual_seed(32)

#----------------------------- 构造词表 --------------------------

seg = pkuseg.pkuseg()
with open("datasets/xiaoaojianghu.txt", "r", encoding="utf-8") as fi:
    text = fi.read()
cutted_text = seg.cut(text)
print("length of dataset in characters:" , len(cutted_text))
# 构建词汇表,这里我们采用了最简单的方法，常用的有sentencePiece、BPE等等
vocab = ["<UNK>"]
vocab.extend(list(set(cutted_text)))
vocab_size = len(vocab)
print("vocab size: ", vocab_size)

# 构建字符和索引的映射
wtoi = {ch:i for i, ch in enumerate(vocab)}
itow = {i:ch for i, ch in enumerate(vocab)}
# encode和decode函数
def encode(s:str):
    res = []
    for w in seg.cut(s):
        if w in wtoi:
            res.append(wtoi[w])
        else:
            res.append(wtoi["<UNK>"])
    return res

def decode(s:list):
    return "".join([itow[i] for i in s])

# 测试encode和decode函数
print(encode("少镖头，快来，这里有野鸡"))
print(decode(encode("少镖头，快来，这里有野鸡")))

#----------------------------- 处理训练数据 ------------------------

data = torch.tensor(encode(text), dtype=torch.long) # 利用上述编码器编码整个数据集并存储为tensor
print(data.shape, data.dtype)
print(data[:100])

# 划分数据集
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# # 通常训练transformer并不是用整个训练集，而仅仅采用一段段的数据
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"输入序列为： {context}, 监督： {target}")

def get_batch(split): # 获取batch数据
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

#----------------------------- 定义模型 ------------------------

class BigramLanguageModel(nn.Module): # 二元语言模型
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, embd_size).to(device)
        self.ln = nn.Linear(embd_size, vocab_size)
    
    def forward(self, idx, targets=None):
        """
        idx:[4, 8]
        target:[4, 8]
        """
        embd = self.embedding_table(idx) # [4, 8, 65]
        logits = self.ln(embd)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens): 
        """
        max_new_tokens:最大生成长度int
        idx:输入上下文[B, T]
        """
        # pred_idx = idx.shape[1]
        for _ in range(max_new_tokens):
            # predictions
            logits, _ = self(idx) # logits: [4, 8, 65]
            # focus on the last time step
            logits = logits[:,-1,:] # logits: [4, 65] 取最后一个token的embedding
            probs = F.softmax(logits, dim=-1) # probs:[4,65]
            idx_next = torch.multinomial(probs, num_samples=1) # idx_next:[4, 1]
            idx = torch.cat((idx, idx_next), dim=-1)
        # print(idx)
        # prediction = idx[:,pred_idx:]
        # print(prediction)
        return idx
    

m = BigramLanguageModel(vocab_size=vocab_size) #初始化模型
m.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# 测试模型
xb, yb = get_batch("train")
m.generate(xb, 8)
out, loss = m(xb, yb)
prompt="你好，我是令狐冲"
print(decode(m.generate(torch.tensor([encode(prompt)], dtype=torch.long).to(device), max_new_tokens=500)[0].tolist()))


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]: # 分别在测试集和训练集上测试模型的loss
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 训练模型
optimizer = optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss(model=m)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # get a batch of train data
    xb, yb = get_batch("train")

    # train lm
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

prompt="你好，我是令狐冲"
print(decode(m.generate(torch.tensor([encode(prompt)], dtype=torch.long).to(device), max_new_tokens=500)[0].tolist()))


