from torch.utils.tensorboard import SummaryWriter
from src.config import *

# 创建一个SummaryWriter对象，指定日志目录
writer = SummaryWriter(log_dir=ROOT_DIR / 'test/logs')