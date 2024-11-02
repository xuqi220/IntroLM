import json,time
import torch
import pkuseg
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# 定义超参数
batch_size = 16  # 并行处理的数据数量
block_size = 64  # 输入seq的最大长度(tansformer可以捕获的上下文长度，长度越长上下文信息越丰富，但是计算复杂度也随之上升)
lr = 1e-3
max_iters = 10000 # 训练轮数
eval_interval = 1000 # 每训练1000步评估一次
eval_iters = 500 # 评估轮数
n_embd = 64 # 词向量维度
head_size = 32 # 每个头的维度
num_heads = int(n_embd/head_size) # 头数
dr = 0.5 # dropout rate
num_blocks = 1 # transformer block 层数

if torch.cuda.is_available():  
    device = "cuda" 
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(32)

# 理解数据
seg = pkuseg.pkuseg()
with open("datasets/xiaoaojianghu.txt", "r", encoding="utf-8") as fi:
    text = fi.read()
cutted_text = seg.cut(text)
print("length of dataset in characters:" , len(cutted_text))
# 构建词汇表
vocab = ["<UNK>"]
vocab.extend(list(set(cutted_text)))
vocab_size = len(vocab)
print("vocab size: ", vocab_size)

# 构建字符和索引的映射,这里我们采用了最简单的方法，常用的有sentencePiece、BPE等等
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


# 编码整个数据集并存储为tensor
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])

# 划分数据集
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# # 通常训练transformer并不是用整个训练集，而仅仅采用一段段的数据
# block_size = 8
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"输入序列为： {context}, 监督序列： {target}")

def get_batch(split):
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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

class Head(nn.Module):
    """单头自注意力"""
    def __init__(self, head_size) -> None:
        super().__init__()
        """
        head_size: 单头注意力机制的维度
        """
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 下三角矩阵用于mask future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 

        self.dropout = nn.Dropout(dr)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # 计算注意力分数
        t = k.transpose(-2, -1)
        wei = q @ t * C**-0.5  # (B, T, T)
        # mask future token's attention score
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return v
    
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_head) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dr)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1) # (B, T, C)
        out = self.dropout(self.proj(out))
        return out # (B, T, C)
    
class FeedForward(nn.Module):
    def __init__(self,n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dr)
        )

    def forward(self, x):
        out = self.net(x) # (B, T, C)
        return out

# 将上述的MultiHeadAttention、FeedForward组织为一个Block
class Block(nn.Module):
    def __init__(self, head_size, num_heads, n_embd) -> None:
        super().__init__()
        self.at = MultiHeadAttention(head_size=head_size, num_head=num_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x+self.at(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x


# 简单的、小规模的（Simple, Samll）--> double S GPT 
class minGPT(nn.Module):
    def __init__(self, vocab_size, n_embd) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd).to(device)
        self.position_embedding_table = nn.Embedding(vocab_size, n_embd).to(device)
        self.blocks = nn.Sequential(*[Block(head_size, num_heads, n_embd) for _ in range(num_blocks)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        """
        idx:[4, 8]
        target:[4, 8]
        """
        token_embd = self.token_embedding_table(idx) # [4, 8, 64]
        posi_embd = self.position_embedding_table(idx)
        x = token_embd+posi_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
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
            # 选取最后的block_size窗口大小的idx
            idx_cond = idx[:, -block_size:]
            # predictions
            logits, loss = self(idx_cond) # logits: [4, 8, 64]
            # focus on the last time step
            logits = logits[:,-1,:] # logits: [4, 65]
            probs = F.softmax(logits, dim=-1) # probs:[4,65]
            idx_next = torch.multinomial(probs, num_samples=1) # idx_next:[4, 1]
            idx = torch.cat((idx, idx_next), dim=-1)
        # print(idx)
        # prediction = idx[:,pred_idx:]
        # print(prediction)
        return idx
    
# 定义模型
m = minGPT(vocab_size=vocab_size, n_embd=n_embd)
m.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
# 测试模型
# xb, yb = get_batch("train")
# m.generate(xb, 8)
# out, loss = m(xb, yb)
print(decode(m.generate(torch.tensor([encode("令狐冲")], dtype=torch.long).to(device), max_new_tokens=1000)[0].tolist()))

# 训练模型
optimizer = optim.AdamW(m.parameters(), lr=1e-3)

t1 = time.time()
for iter in range(max_iters):

    if iter % eval_interval == 0:
        t2 = time.time()
        losses = estimate_loss(model=m)
        print(f"step: {iter} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f} | time: {round((t2-t1)/60, 1)} min")

    # get a batch of train data
    xb, yb = get_batch("train")

    # train lm
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


print(decode(m.generate(torch.tensor([encode("令狐冲")], dtype=torch.long).to(device), max_new_tokens=1000)[0].tolist()))


