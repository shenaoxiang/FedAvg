import time

timestamp = time.time()
print(timestamp)

import torch

# 创建两个张量
tensor1 = torch.tensor([1, 2, 3], dtype=torch.float32)
tensor2 = torch.tensor([4, 5, 6], dtype=torch.float32)

# 执行乘法操作
result = torch.mul(tensor1, tensor2)

# 将结果转换为指定的数据类型（例如，Long）
result = result.to(torch.long)

print(result)
