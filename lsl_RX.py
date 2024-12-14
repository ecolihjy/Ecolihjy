from pylsl import StreamInlet, resolve_stream
import threading
import time


def receive_data():
    # 查找 'meta_feedback' 类型的数据流
    streams = resolve_stream('type', 'Markers')

    # 创建数据流输入端
    inlet = StreamInlet(streams[0])

    while True:
        # 从数据流中读取样本
        sample, timestamp = inlet.pull_sample()

        # 打印读取到的数据
        print(f"Received data: {sample}, timestamp: {timestamp}")


# 启动接收数据的线程
data_thread = threading.Thread(target=receive_data)
data_thread.start()

# 等待线程结束
data_thread.join()