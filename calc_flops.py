import torch
import torch.nn as nn
from thop import profile
from network import VLTransformer 

# === 1. 设置真实参数 ===
# 你的模态维度
input_data_dims = [48, 6, 10, 256] 
# 你的节点数
num_nodes = 871 

# === 2. 模拟参数配置 (Mock Args) ===
class MockArgs:
    def __init__(self):
        self.n_head = 8
        self.n_hidden = 16
        self.nlayer = 3
        self.dropout = 0.65
        self.mode = 'pre-train' 
        self.d_model = self.n_head * self.n_hidden 
        
        # 补全缺少的参数
        self.nmodal = 4               
        self.nclass = 2               
        self.th = 0.9                 
        self.GC_mode = 'adaptive-learning' 
        self.MP_mode = 'GAT'          
        self.MF_mode = ''             
        self.alpha = 0.5             

args = MockArgs()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 3. 实例化模型 ===
print('==> Building VLTransformer model...')
model = VLTransformer(input_data_dims, args).to(device)

# === 4. 构造虚拟输入 (关键修正点) ===
total_feat_dim = sum(input_data_dims)

# 【修正】：去掉 Batch 维度，直接使用 (Nodes, Features)
# 之前的 (1, 871, 320) 改为 (871, 320)
dummy_input = torch.randn(num_nodes, total_feat_dim).to(device)

# === 5. 计算 FLOPs 和 Params ===
print('==> Calculating...')
# 注意：profile 会自动处理 forward 过程
flops, params = profile(model, inputs=(dummy_input, ))

print('\n' + '='*30)
print(f'Input Shape: ({num_nodes}, {total_feat_dim})')
print(f'Params: {params / 1e6:.4f} M')
print(f'FLOPs:  {flops / 1e9:.4f} G')
print('='*30 + '\n')
