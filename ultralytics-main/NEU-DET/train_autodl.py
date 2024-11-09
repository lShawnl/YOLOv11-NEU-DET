import os
import torch
from ultralytics import YOLO
from datetime import datetime

# 检查 GPU 是否可用
if torch.cuda.is_available():
    device = 'cuda'
    print("Using GPU device for training.")
else:
    device = 'cpu'
    print("GPU device not available. Falling back to CPU.")

# 验证路径
model_path = '/root/ultralytics-main/ultralytics/cfg/models/11/yolo11.yaml'
data_path = '/root/ultralytics-main/NEU-DET/data.yaml'
project_path = os.path.expanduser("~/asdfg")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model config file not found: {model_path}")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data config file not found: {data_path}")
os.makedirs(project_path, exist_ok=True)  # 自动创建保存目录

# 加载 YOLO 模型
model = YOLO(model_path, task='detect')

# 生成时间戳
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"train_{timestamp}"

# 开始训练
model.train(
    data=data_path,
    device=device,
    epochs=200,
    batch=8,
    project=project_path,
    name=run_name,
    exist_ok=False,
    plots=True,
    verbose=True
)

print(f"Training completed. Results saved to: {project_path}/{run_name}")
