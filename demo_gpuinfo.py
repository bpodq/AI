import pynvml


# 初始化模块
pynvml.nvmlInit()
# 获取0号位GPU
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
gpu_info = pynvml.nvmlDeviceGetMemoryInfo(gpu)

print(gpu_info.total)
print(gpu_info.used)
print(gpu_info.free)

