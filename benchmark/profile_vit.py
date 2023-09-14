import os
import sys
import time
from collections import defaultdict

import crypten
from crypten.config import cfg
from models import ViT

# 2PC setting
rank = sys.argv[1]
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(2)
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["RENDEZVOUS"] = "env://"

# 设置通信后端 
cfg.communicator.verbose = True
crypten.init()

# 生成输入
BATCH_SIZE = 16
IMG_SIZE = 256
dummy_input = crypten.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)

# 定义计时统计
timing = defaultdict(float)

# 加载模型
model = ViT(
    image_size=IMG_SIZE,
    patch_size=16,
    num_classes=1000,
    dim=512,
    depth=6,
    heads=16,
    mlp_dim=512
).encrypt()

# 加密输入 
input = crypten.cryptensor(dummy_input)

# 前向传播计时
for i in range(10):
    start = time.time()

    out = model(input)

    end = time.time()

    timing["total_time"] += (end - start)

# 打印时间统计
print(timing)

# 定义计时字典
timing = defaultdict(float)


# Patch Embedding计时
def profile_embedding(x):
    start = time.time()
    x = model.patch_embedding(x)
    end = time.time()
    timing["patch_embedding_time"] += end - start
    return x


# Attention计时
def profile_attention(x):
    start = time.time()
    x = model.attn(x)
    end = time.time()
    timing["attention_time"] += end - start
    return x


# MLP头计时
def profile_mlp(x):
    start = time.time()
    x = model.mlp(x)
    end = time.time()
    timing["mlp_time"] += end - start
    return x


# 前向传播
for i in range(10):
    start = time.time()

    x = profile_embedding(input)
    x = profile_attention(x)
    x = profile_mlp(x)

    out = model(input)

    end = time.time()
    timing["total_time"] += (end - start)

# 打印结果
print(timing)