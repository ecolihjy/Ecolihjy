import os
import matplotlib.pyplot as plt
import numpy as np
import mne

#下面这句改变文件地址
# sample_data_folder = mne.datasets.sample.data_path()
# sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
sample_data_raw_file = "E:\metaBCI\dynamic stop\FFT-test/EEG_DATA8.cnt"
raw = mne.io.read_raw_cnt(sample_data_raw_file, preload=True, verbose=False)

# 对原始数据应用带通滤波器
raw.filter(l_freq=2.5, h_freq=80, method='iir')

events = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events[0], tmin=-0.01, tmax=1.99)

# 选择第一个通道（索引为0）进行FFT分析
channel_index = 53
channel_name = epochs.ch_names[channel_index]  # 获取通道名称
# 获取所选通道的所有epochs数据
epoch_data = epochs.get_data()[:, channel_index, :]  # 仅提取一个通道的数据，提取所有通道(:)的特定通道(channel_index)的所有时间

# 对所有epochs的数据进行FFT变换
fft_results = np.fft.fft(epoch_data)

# 计算频率轴（基于采样率）
sfreq = epochs.info['sfreq']
freqs = np.fft.fftfreq(epoch_data.shape[1], d=1 / sfreq)

# 由于FFT结果是对称的，只取一半（正频率部分）
half_n_times = epoch_data.shape[1] // 2
fft_magnitude = np.abs(fft_results[:, :half_n_times])

# 绘制所有epochs的FFT结果
plt.figure(figsize=(10, 5))
for i in range(19,20):
    peak_index = np.argmax(fft_magnitude[i])  # 找到最大值的索引
    plt.plot(freqs[:half_n_times], fft_magnitude[i], label=f'Epoch {i+1} FFT of Channel {channel_name}', alpha=0.5)

    # 在峰值位置添加标注
    if fft_magnitude[i, peak_index] > 0:  # 确保有实际的峰值存在
        plt.annotate(f'{freqs[peak_index]:.3f} Hz\n{fft_magnitude[i, peak_index]:.3f}',
                     xy=(freqs[peak_index], fft_magnitude[i, peak_index]),
                     textcoords="offset points",  # 使用偏移点作为文本位置
                     xytext=(0, 10),  # 偏移量，x方向无偏移，y方向向下偏移10个单位
                     ha='center')  # 水平居中对齐文本
plt.title('FFT Analysis of All Epochs - ' + channel_name)
# # 绘制第一个epoch的FFT结果作为示例
# plt.figure(figsize=(10, 5))
# plt.plot(freqs[:half_n_times], fft_magnitude[3], label='FFT of Channel ' + channel_name)
# plt.title('FFT Analysis of First Epoch - ' + channel_name)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.xlim(left=0, right=100)
plt.show()


