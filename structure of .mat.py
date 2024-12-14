import numpy as np
import scipy.io

# 加载.mat文件
mat_data = scipy.io.loadmat('C:\\Users\\admin\mne_data\MNE-tsinghua-data\\upload\yijun\S1.mat')

# 打印.mat文件中包含的字段（变量名）
print("Fields in the .mat file:")
for field_name in mat_data.keys():
    print(field_name)

# 打印文件中的变量名称
print("Variables in the .mat file:")
for key in mat_data:
    print(key)

# 打印每个变量的形状和数据类型，或标记为字节数据
for key, value in mat_data.items():
    if isinstance(value, np.ndarray):
        print(f"{key} ({value.dtype}): shape={value.shape}")
    else:
        print(f"{key}: bytes data")