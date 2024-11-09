import torch
from ultralytics import YOLO

# 检查 MPS 设备是否可用，如果不可用则退回到 CPU
if torch.backends.mps.is_available():
    device = 'mps'
    print("Using MPS device for training.")
else:
    device = 'cpu'
    print("MPS device not available. Falling back to CPU.")

# 加载 YOLO 模型
model = YOLO(r'/Users/asdfg/ultralytics-main/ultralytics/cfg/models/11/yolo11l.yaml', task='detect')

# 开始训练
model.train(
    data=r'/Users/asdfg/ultralytics-main/NEU-DET/data.yaml',
    device = device,
    epochs = 300
)
