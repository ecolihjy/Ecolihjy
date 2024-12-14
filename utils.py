import serial
from psychopy import parallel
import numpy as np

from pylsl import StreamInfo, StreamOutlet


class NeuroScanPort:
    """
    Send tag communication Using parallel port or serial port.
    发送标签通信 使用并行端口或串行端口。
    author: Lichao Xu

    Created on: 2020-07-30

    update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        use_serial: bool
            If False, send the tags using parallel port, otherwise using serial port.
        baudrate: int
            The serial port baud rate.

    Attributes
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        use_serial: bool
            If False, send the tags using parallel port, otherwise using serial port.
        baudrate: int
            The serial port baud rate.
        port:
            Send tag communication Using parallel port or serial port.

    Tip
    ----
    .. code-block:: python
       :caption: An example of using port to send tags
        使用 brainstim.utils 模块中的 NeuroScanPort 类在实验过程中发送标签或标记。
        NeuroScanPort 类用于与设备（如 EEG 或 MEG 系统）进行接口，以在实验过程中发送标签或标记。
        这些标签或标记可用于将实验事件与记录的数据进行同步，从而实现对数据的准确分析和解释。

        from brainstim.utils import NeuroScanPort
        port = NeuroScanPort(port_addr, use_serial=False) if port_addr else None
        VSObject.win.callOnFlip(port.setData, 1)
        port.setData(0)

    """

    def __init__(self, port_addr, use_serial=False, baudrate=115200):
        self.use_serial = use_serial
        if use_serial:
            self.port = serial.Serial(port=port_addr, baudrate=baudrate)
            self.port.write([0])
        else:
            self.port = parallel.ParallelPort(address=port_addr)

    def setData(self, label):
        """Send event labels
        用于向连接的设备发送事件标签或标记
        Parameters
        ----------
            label:
                The label sent.

        """
        if self.use_serial:
            self.port.write([int(label)])
        else:
            self.port.setData(int(label))


class NeuraclePort:
    """
    Send trigger to Neuracle device.The Neuracle device uses serial
    port for writing trigger, so it does not need to write a 0 trigger
    before a int trigger. This class is writen under the Trigger box instruction.

    author: Jie Mei

    Created on: 2022-12-05

    update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        baudrate: int
            The serial port baud rate.

        该类的目的是提供一种方便的方式来向 Neuracle 设备发送触发信号。该类抽象了 Neuracle 设备所需的特定通信协议的细节。
        该类与先前的 NeuroScanPort 类的主要区别在于:
        Neuracle 设备使用串行端口,而先前的类支持串行和并行端口。
        Neuracle 设备不需要在发送实际触发信号之前发送"0"触发信号,这是先前类的要求。
    """

    def __init__(self, port_addr, baudrate=115200) -> None:
        # The only choice for neuracle is using serial for writting trigger
        self.port = serial.Serial(port=port_addr, baudrate=baudrate)

    def setData(self, label):
        # Neuracle doesn't need 0 trigger before a int trigger.
        if str(label) != '0':
            head_string = '01E10100'
            hex_label = str(hex(label))
            if len(hex_label) == 3:
                hex_value = hex_label[2]
                hex_label = '0'+hex_value.upper()
            else:
                hex_label = hex_label[2:].upper()
            send_string = head_string+hex_label
            send_string_byte = [int(send_string[i:i+2], 16)
                                for i in range(0, len(send_string), 2)]
            self.port.write(send_string_byte)


class LsLPort:
    """
    Creating a lab streaming layer marker, which could align with the
    stream which retriving stream from devices.

    """

    def __init__(self) -> None:
        self.info = StreamInfo(
            name='LSLMarkerStream',
            type='Marker',
            channel_count=1,
            nominal_srate=0,
            channel_format='cf_int16')
        self.outlet = StreamOutlet(self.info)

    def setData(self, label):
        # We don't need 0 trigger before a int trigger
        if str(label) != '0':
            self.outlet.push_sample(str(label))


def _check_array_like(value, length=None):
    """
    Check array dimensions.

    -author: Lichao Xu

    -Created on: 2020-07-30

    -update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        value: ndarray,
            The array to check.
        length: int,
            The array dimension.

    """
    # 该函数使用isinstance()函数检查输入value是否为列表、元组或NumPy ndarray的实例。
    # 如果提供了length参数，该函数会检查输入value的长度是否等于指定的length。
    flag = isinstance(value, (list, tuple, np.ndarray))
    return flag and (len(value) == length if length is not None else True)


def _clean_dict(old_dict, includes=[]):
    """
    Clear dictionary.

    -author: Lichao Xu

    -Created on: 2020-07-30

    -update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        old_dict: dict,
            The dict to clear.
        includes: list,
            Key-value indexes that need to be preserved.

    """

    names = list(old_dict.keys())
    for name in names:
        if name not in includes:
            old_dict[name] = None
            del old_dict[name]
    return old_dict
