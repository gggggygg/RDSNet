# 触发加载 .so，必须和 extension 名一致
# from .fisheye_conv_ext import *

# import torch
# import fisheye_conv_ext
from . import _C  # 加载 .so 文件，触发 C++ 注册
