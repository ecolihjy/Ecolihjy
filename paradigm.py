# -*- coding: utf-8 -*-
import datetime
import math

# load in basic modules
import os
import os.path as op
import queue
import string
from datetime import time

import numpy as np
from math import pi
from psychopy import data, visual, event
from psychopy.visual.circle import Circle
from pylsl.pylsl import StreamInlet, resolve_byprop
from sympy import false

from Mycode.utils import NeuroScanPort, NeuraclePort, _check_array_like
import threading
from copy import copy
import random
from scipy import signal
from PIL import Image

from MyTrigger import Trigger

# prefunctions


def sinusoidal_sample(freqs, phases, srate, frames, stim_color):#stim_color 是一个用来指定每个频率的 RGB 颜色值的参数,可以用来灵活地控制最终显示的颜色。
    """
    具有不同频率和相位的多个刺激被呈现给受试者,并且需要以正弦波的方式调制刺激的颜色。

    Sinusoidal approximate sampling method.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        freqs: list of float
            Frequencies of each stimulus.
        phases: list of float
            Phases of each stimulus.
        srate: int or float
            Refresh rate of screen.
        frames: int
            Flashing frames.
        stim_color: list
            Color of stimu.

    Returns
    ----------
        color: ndarray
            shape(frames, len(fre), 3)

    """
    # 这句话创建了一个时间向量 time,其中包含了从 0 到整个时间序列持续时间之间的 frames 个等间隔的时间点。
    # 这个时间向量将用于后续的计算,例如计算每个时间点的正弦波形。(frames - 1) / srate: 这个表达式计算了总的时间长度。
    # frames 是总的帧数,srate 是屏幕的刷新率(每秒帧数)。因此,(frames - 1) / srate 就是整个时间序列的持续时间。
    time = np.linspace(0, (frames - 1) / srate, frames)
    color = np.zeros((frames, len(freqs), 3)) #这个三维数组 color 可以用来存储每个时间帧和每个频率下的 RGB 颜色值
    for ne, (freq, phase) in enumerate(zip(freqs, phases)):
        sinw = np.sin(2 * pi * freq * time + pi * phase) + 1
        color[:, ne, :] = np.vstack(#创建一个 3xN 的数组
            (sinw * stim_color[0], sinw * stim_color[1], sinw * stim_color[2])
        ).T
        # 这在用户想要显示只有部分色彩通道的刺激物，或者想要创建灰度刺激物（将所有三个通道设置为相同值）
        if stim_color == [-1, -1, -1]:
            pass
        else:
            if stim_color[0] == -1:
                color[:, ne, 0] = -1
            if stim_color[1] == -1:
                color[:, ne, 1] = -1
            if stim_color[2] == -1:
                color[:, ne, 2] = -1

    return color


def wave_new(stim_num, type):
    """determine the color of each offset dot according to "type".
        根据指定的"类型"参数确定每个偏移点的颜色

    author: Jieyu Wu

    Created on: 2022-12-14

    update log:
        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        stim_num: int
            Number of stimuli dots of each target.
        type: int
            avep code.

    Returns
    ----------
        point: ndarray
            (stim_num, 3)

    """
    point = [[-1, -1, -1] for i in range(stim_num)]
    if type == 0:
        pass
    else:
        point[type - 1] = [1, 1, 1]
    point = np.array(point)
    return point


def pix2height(win_size, pix_num):
    height_num = pix_num / win_size[1]
    return height_num


def height2pix(win_size, height_num):
    pix_num = height_num * win_size[1]
    return pix_num


def code_sequence_generate(basic_code, sequences):
    """Quickly generate coding sequences for sub-stimuli using basic endcoding units and encoding sequences.
        使用基本的终端编码单元和编码序列快速生成子刺激的编码序列
    author: Jieyu Wu

    Created on: 2023-09-18

    update log:
        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        basic_code: list
            Each basic encoding unit in the encoding sequence.
        sequences: list of array
            Encoding sequences for basic_code.

    Returns
    ----------
        code: ndarray
            coding sequences for sub-stimuli.

    """

    code = []
    for seq_i in range(len(sequences)):
        code_list = []
        seq_length = len(sequences[seq_i])
        for code_i in range(seq_length):
            code_list.append(basic_code[sequences[seq_i][code_i]])
        code.append(code_list)
    code = np.array(code)
    return code#每一行表示一个子刺激的编码序列


# create interface for VEP-BCI-Speller


class KeyboardInterface(object):
    """Create the interface to the stimulus interface and initialize the window parameters.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        win:
            The window object.
        win_size: ndarray, shape(width, high)
            The size of the window in pixels.
        stim_length: int
            The length of the stimulus block in pixels.
        stim_width: int
            The width of the stimulus block in pixels.
        n_elements: int
            Number of stimulus blocks.
        stim_pos: ndarray, shape([x, y],...)
            Customize the position of the stimulus blocks with an array length
            that corresponds to the number of stimulus blocks.
        stim_sizes: ndarray, shape([length, width],...)
            The size of the stimulus block, the length of which corresponds to the number of stimulus blocks.
        symbols: str
            Stimulate the text of characters in the block.
        text_stimuli:
            Configuration information required for paradigm characters.
        rect_response:
            Configuration information required for the rectangular feedback box.
        res_text_pos: tuple, shape (x, y)
            The character position of the online response.
        symbol_height: int
            The height of the feedback character.
        symbol_text: str
            The character text of the online response.
        text_response:
            Configuration information for the feedback character.

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        self.win = win
        win.colorSpace = colorSpace
        win.allowGUI = allowGUI
        win_size = win.size
        self.win_size = np.array(win_size)  # e.g. [1920,1080]

    def config_pos(
        self,
        n_elements=40,
        rows=5,
        columns=8,
        stim_pos=None,
        stim_length=150,
        stim_width=150,
    ):
        """Set the number, position, and size parameters of the stimulus block.

        update log:
            2022-06-26 by Jianhang Wu

            2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

        Parameters
        ----------
            n_elements: int
                Number of stimulus blocks, default is 40.
            rows: int
                Sets the number of stimulus block rows.
            columns: int
                Set the number of stimulus block columns.
           stim_pos: ndarray, shape(x,y)
                自定义刺激块的位置，如果为“无”，则将其排列成矩形阵列。
                Customize the position of the stimulus block, if None then it will be arranged in a rectangular array.
            stim_length: int
                Length of stimulus.
            stim_width: int
                Width of stimulus.

        Raises
        ----------
            Exception: Inconsistent numbers of stimuli and positions

        """

        self.stim_length = stim_length
        self.stim_width = stim_width
        self.n_elements = n_elements
        # highly customizable position matrix
        if (stim_pos is not None) and (self.n_elements == stim_pos.shape[0]):
        # if stim_pos is not None:# 原来是if (stim_pos is not None) and (self.n_elements == stim_pos.shape[0])
            # note that the origin point of the coordinate axis should be the center of your screen
            # (so the upper left corner is in Quadrant 2nd), and the larger the coordinate value,
            # the farther the actual position is from the center
            self.stim_pos = stim_pos
        # conventional design method
        elif (stim_pos is None) and (rows * columns >= self.n_elements):
            # according to the given rows of columns, coordinates will be automatically converted
            # 这行代码的效果是将实验中所有刺激物的位置初始化为(0, 0), 即显示屏或坐标系的原点或左上角
            stim_pos = np.zeros((self.n_elements, 2))
            # divide the whole screen into rows*columns' blocks, and pick the center of each block
            first_pos = (
                np.array([self.win_size[0] / columns, self.win_size[1] / rows]) / 2
            )
            if (first_pos[0] < stim_length / 2) or (first_pos[1] < stim_width / 2):
                raise Exception("Too much blocks or too big the stimulus region!")
            for i in range(columns):
                for j in range(rows):
                    stim_pos[i * rows + j] = first_pos + [i, j] * first_pos * 2
            # note that those coordinates are still not the real ones that
            # need to be set on the screen
            # 将计算出的刺激位置转换为实际的屏幕坐标
            stim_pos -= self.win_size / 2  # from Quadrant 1st to 3rd
            stim_pos[:, 1] *= -1  # invert the y-axis
            self.stim_pos = stim_pos
        else:
            raise Exception("Incorrect number of stimulus!")

        # check size of stimuli
        stim_sizes = np.zeros((self.n_elements, 2))
        stim_sizes[:] = np.array([stim_length, stim_width])
        self.stim_sizes = stim_sizes
        self.stim_width = stim_width
        self.columns = columns
        self.rows = rows

    def config_text(
        self, unit="pix", symbols=None, symbol_height=0, tex_color=[1, 1, 1]
    ):
        """Sets the characters within the stimulus block.

        update log:
            2022-06-26 by Jianhang Wu

            2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

        Parameters
        ----------
            symbols: str
                Edit character text.
            symbol_height: int
                The height of the character in pixels.
            tex_color: list, shape(red, green, blue)
                Set the character color, the value is between -1.0 and 1.0.

        Raises
        ----------
            Exception: Insufficient characters

        """

        # check number of symbols
        if (symbols is not None) and (len(symbols) >= self.n_elements):
            self.symbols = symbols
        elif self.n_elements <= 40:
            self.symbols = "".join([string.ascii_uppercase, "1234567890+-*/"])
        else:
            raise Exception("Please input correct symbol list!")

        # add text targets onto interface
        if symbol_height == 0:
            symbol_height = self.stim_width / 2
        self.text_stimuli = []
        for symbol, pos in zip(self.symbols, self.stim_pos):
            self.text_stimuli.append(
                visual.TextStim(
                    win=self.win,
                    text=symbol,
                    font="Times New Roman",
                    pos=pos,
                    color=tex_color,
                    units=unit,
                    height=symbol_height,
                    bold=True,
                    name=symbol,
                )
            )

    def config_response(
        self,
        symbol_text="Speller:  ",#这是 symbol_text 参数的默认值，表示将显示为在线响应的文本。如果在调用函数时未提供此参数的值，它将默认为 "Speller: "
        symbol_height=0,#表示响应符号的高度（响应符号是什么？）
        symbol_color=(1, 1, 1),#如果在调用函数时未提供此参数的值，它将默认为 (1, 1, 1)，即白色
        bg_color=[-1, -1, -1],#如果在调用函数时未提供此参数的值，它将默认为 (-1, -1, -1)，即黑色
    ):
        """Sets the character of the online response.

        update log:
            2022-08-10 by Wei Zhao

            2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

        Parameters
        ----------
            symbol_text: str
                Online response string.
            symbol_height: int
                Height of response symbol.
            symbol_color: tuple, shape (red, green, blue)
                The character color of the online response, the value is between -1.0 and 1.0.
            bg_color: list
                Color of background symbol.

        Raises
        ----------
            Exception: Insufficient characters
        """
        # brige_length的计算方式是取窗口宽度的中心，加上刺激位置的x坐标，减去刺激长度的一半。这有效地计算了从窗口中心到刺激左边缘的桥的长度
        brige_length = self.win_size[0] / 2 + self.stim_pos[0][0] - self.stim_length / 2
        brige_width = self.win_size[1] / 2 - self.stim_pos[0][1] - self.stim_width / 2

        # 创建了一个跨越窗口宽度（不包括"桥"区域）、高度为"桥"区域三倍的矩形视觉元素。该矩形垂直居中于窗口中，填充颜色为背景色，边框颜色为白色。
        # 这个视觉元素可能用作实验或模拟中的响应区域或视觉提示
        self.rect_response = visual.Rect(#创建在线反馈框
            win=self.win,
            units="pix",
            width=self.win_size[0] - brige_length,
            height=brige_width * 3 / 3,
            pos=(0, self.win_size[1] / 2 - brige_width / 2),#这将矩形垂直居中于窗口中
            fillColor=bg_color,
            lineColor=[1, 1, 1],
        )

        self.res_text_pos = (#定义了文本元素的位置
            -self.win_size[0] / 2 + brige_length * 3 / 2,
            self.win_size[1] / 2 - brige_width / 2,
        )
        self.reset_res_pos = (#定义了文本元素的位置
            -self.win_size[0] / 2 + brige_length * 3 / 2,
            self.win_size[1] / 2 - brige_width / 2,
        )
        self.reset_res_text = ">:  "
        if symbol_height == 0:
            self.symbol_height = brige_width
        self.symbol_text = symbol_text
        self.text_response = visual.TextStim(
            win=self.win,
            text=symbol_text,
            font="Times New Roman",
            pos=self.res_text_pos,
            color=symbol_color,
            units="pix",
            height=self.symbol_height,
            bold=True,
        )


# config visual stimuli


class VisualStim(KeyboardInterface):
    """Create various visual stimuli.

    The subclass VisualStim inherits from the parent class KeyboardInterface, duplicate properties are not listed.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        index_stimuli:
            Configuration information for the target prompt.

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)
        self._exit = threading.Event()

    def config_index(self, index_height=0, units="pix"):
        """Config index stimuli: downward triangle (Unicode: \u2BC6)

        Parameters
        ----------
            index_height: int
                The height of the cue symbol, which defaults to half the height of the stimulus block.

        """

        # add index onto interface, with positions to be confirmed.
        if index_height == 0:
            index_height = copy(self.stim_width / 3 * 2)
        self.index_stimuli = visual.TextStim(
            win=self.win,
            text="\u2BC6",
            font="Arial",
            color=[1.0, -1.0, -1.0],
            colorSpace="rgb",
            units=units,
            height=index_height,
            bold=True,
            autoLog=False,
        )


# standard SSVEP paradigm


class SemiCircle(Circle):
    """
    A SemiCircle class inherited from Circle.
    """

    def _calcVertices(self):
        # only draw half of a circle
        d = np.pi / self.edges
        self.vertices = np.asarray(
            [
                np.asarray((np.sin(e * d), np.cos(e * d))) * self.radius
                for e in range(int(round(self.edges) + 1))
            ]
        )


# standard SSVEP paradigm


class SSVEP(VisualStim):
    """Create SSVEP stimuli.

    The subclass SSVEP inherits from the parent class VisualStim, and duplicate properties are not listed.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        refresh_rate: int
            Screen refresh rate.
        stim_time: float
            Time of stimulus flash
        stim_color: list, shape(red, green, blue)
            The color of the stimulus block, taking values between -1.0 and 1.0.
        stim_opacities: float
            Opacity, default opaque.
        stim_frames: int
            The number of frames contained in a single-trial stimulus.
        stim_oris: ndarray
            刺激快的方向
            Orientation of the stimulus block.
        stim_sfs: ndarray
            刺激块的空间频率。
            Spatial frequency of the stimulus block.
        stim_contrs: ndarray
            Stimulus block contrast.
        freqs: list, shape(fre, …)
            Stimulus block flicker frequency, length consistent with the number of stimulus blocks.
        phases: list, shape(phase, …)
             Stimulus block flicker phase, length consistent with the number of stimulus blocks.
        stim_colors: list, shape(red, green, blue)
            The color configuration required for the stimulus block flashing.
        flash_stimuli:
            刺激块闪烁所需的配置信息。
            The configuration information required for the flashing of the stimulus block.

    Tip
    ----
     .. code-block:: python
        :caption: An example of creating SSVEP stimuli.

        from psychopy import monitors
        import numpy as np
        from brainstim.framework import Experiment
        from brainstim.paradigm import SSVEP,paradigm

        win = ex.get_window()

        # press q to exit paradigm interface
        n_elements, rows, columns = 20, 4, 5
        stim_length, stim_width = 150, 150
        stim_color, tex_color = [1,1,1], [1,1,1]
        fps = 120                                                   # screen refresh rate
        stim_time = 2                                               # stimulus duration
        stim_opacities = 1                                          # stimulus contrast
        freqs = np.arange(8, 16, 0.4)                               # Frequency of instruction
        phases = np.array([i*0.35%2 for i in range(n_elements)])    # Phase of the instruction
        basic_ssvep = SSVEP(win=win)
        basic_ssvep.config_pos(n_elements=n_elements, rows=rows, columns=columns,
            stim_length=stim_length, stim_width=stim_width)
        basic_ssvep.config_text(tex_color=tex_color)
        basic_ssvep.config_color(refresh_rate=fps, stim_time=stim_time, stimtype='sinusoid',
            stim_color=stim_color, stim_opacities=stim_opacities, freqs=freqs, phases=phases)
        basic_ssvep.config_index()
        basic_ssvep.config_response()
        bg_color = np.array([-1, -1, -1])                           # background color
        display_time = 1
        index_time = 0.5
        rest_time = 0.5
        response_time = 1
        port_addr = None 			                                 # Collect host ports
        nrep = 1
        lsl_source_id = None
        online = False
        ex.register_paradigm('basic SSVEP', paradigm, VSObject=basic_ssvep, bg_color=bg_color,
            display_time=display_time,  index_time=index_time, rest_time=rest_time, response_time=response_time,
            port_addr=port_addr, nrep=nrep,  pdim='ssvep', lsl_source_id=lsl_source_id, online=online)

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        """Item class from VisualStim.

        Args:

        """
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)

    def config_color(
        self,
        refresh_rate,
        stim_time,
        stim_color,
        stimtype="sinusoid",
        stim_opacities=1,
        **kwargs
    ):
        """Config color of stimuli.

        Parameters
        ----------
            refresh_rate: int
                Refresh rate of screen.
            stim_time: float
                Time of each stimulus.
            stim_color: int
                The color of the stimulus block.
            stimtype: str
                Stimulation flicker mode, default to sine sampling flicker.正弦刺激
            stim_opacities: float
                Opacity, default to opaque.不透明度
            freqs: list, shape(fre, …)
                刺激块闪烁频率、长度与刺激块数量一致。
                Stimulus block flicker frequency, length consistent with the number of stimulus blocks.
            phases: list, shape(phase, …)
                Stimulus block flicker phase, length consistent with the number of stimulus blocks.

        Raises
        ----------
            Exception: Inconsistent frames and color matrices

        """

        # initialize extra inputs
        self.refresh_rate = refresh_rate
        self.stim_time = stim_time
        self.stim_color = stim_color
        self.stim_opacities = stim_opacities
        self.stim_frames = int(stim_time * self.refresh_rate)

        # 这段代码的目的是自动检测窗口的实际帧率并相应地设置 self.refresh_rate 变量。这在帧率事先未知或可能变化的情况下很有用，例如在不同硬件或环境上运行时。
        # nIdentical 参数指定用于计算的相同帧的数量，nWarmUpFrames 参数指定用于预热计算的帧数
        if refresh_rate == 0:
            self.refresh_rate = np.floor(
                self.win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20)
            )

        self.stim_oris = np.zeros((self.n_elements,))  # orientation用于存储每个刺激块的方向
        self.stim_sfs = np.zeros((self.n_elements,))  # spatial frequency用于存储每个刺激块的空间频率
        self.stim_contrs = np.ones((self.n_elements,))  # contrast所有元素都初始化为1 用于存储每个刺激块的对比度

        # check extra inputs允许在创建对象时通过 kwargs 参数来覆盖这些属性的默认值
        if "stim_oris" in kwargs.keys():
            self.stim_oris = kwargs["stim_oris"]
        if "stim_sfs" in kwargs.keys():
            self.stim_sfs = kwargs["stim_sfs"]
        if "stim_contrs" in kwargs.keys():
            self.stim_contrs = kwargs["stim_contrs"]
        if "freqs" in kwargs.keys():
            self.freqs = kwargs["freqs"]
        if "phases" in kwargs.keys():
            self.phases = kwargs["phases"]

        # check consistency
        if stimtype == "sinusoid":
            self.stim_colors = (
                sinusoidal_sample(
                    freqs=self.freqs,
                    phases=self.phases,
                    srate=self.refresh_rate,
                    frames=self.stim_frames,
                    stim_color=stim_color,
                )
                - 1
            )
            if self.stim_colors[0].shape[0] != self.n_elements:
                raise Exception("Please input correct num of stims!")

        incorrect_frame = self.stim_colors.shape[0] != self.stim_frames
        incorrect_number = self.stim_colors.shape[1] != self.n_elements
        if incorrect_frame or incorrect_number:
            raise Exception("Incorrect color matrix or flash frames!")

        # add flashing targets onto interface，使用 PsychoPy 库在屏幕上添加闪烁的刺激目标
        self.flash_stimuli = []
        # 在 for 循环中,它会创建 self.stim_frames 个闪烁刺激目标,并将它们添加到 self.flash_stimuli 列表中
        for sf in range(self.stim_frames):
            self.flash_stimuli.append(
                visual.ElementArrayStim(
                    win=self.win,#将刺激目标显示在 self.win 窗口上
                    units="pix",#使用像素作为单位
                    nElements=self.n_elements,#设置刺激目标的数量
                    sizes=self.stim_sizes,#设置每个刺激目标的大小
                    xys=self.stim_pos,#设置每个刺激目标的位置
                    #self.stim_colors[sf, ...] 是一种 NumPy 数组的高级索引方式。
                    # 它会选择 self.stim_colors 数组中第 sf 行的所有列(用 ... 表示)。
                    # 即对于每个闪烁刺激目标,它的颜色会根据当前处理的帧序号 sf 而变化。每个刺激目标在不同的帧中会显示不同的颜色。
                    colors=self.stim_colors[sf, ...],#设置每个刺激目标的颜色,根据当前帧 sf 进行变化
                    opacities=self.stim_opacities,
                    oris=self.stim_oris,
                    sfs=self.stim_sfs,
                    contrs=self.stim_contrs,
                    elementTex=np.ones((64, 64)),# 创建了一个64x64的NumPy数组，填充值为1 每个刺激元素将具有均匀的实心纹理
                    elementMask=None,# 不会应用任何遮罩，整个纹理将可见
                    texRes=48,# 64x64的纹理将缩放到48x48像素渲染在屏幕上
                )
            )


# standard P300 paradigm


class P300(VisualStim):
    """Create P300 stimuli.

    The subclass P300 inherits from the parent class VisualStim, and duplicate properties are no longer listed.

    author: Shengfu Wen

    Created on: 2022-07-04

    update log:
        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        stim_duration: float
            Flashing time interval between rows and columns.
        refresh_rate: int
            Screen refresh rate.
        stim_frames: int
            Total stimulation frames.
        order_index: ndarray
            Row and column labels.
        stim_colors: list, (red, green, blue)
            The color of the stimulus block, ranging from -1.0 to 1.0.
        flash_stimuli: int
            The configuration information required for a trial stimulus.

    Tip
    ----
    .. code-block:: python
        :caption: An example of creating P300 stimuli.

        from psychopy import monitors
        import numpy as np
        from brainstim.framework import Experiment
        from brainstim.paradigm import P300,paradigm

        win = ex.get_window()

        # press q to exit paradigm interface
        n_elements, rows, columns = 20, 4, 5
        tex_color = [1,1,1]                                         # color of text
        fps = 120                                                   # screen refresh rate
        stim_duration = 0.5
        basic_P300 = P300(win=win)
        basic_P300.config_pos(n_elements=n_elements, rows=rows, columns=columns)
        basic_P300.config_text(tex_color=tex_color)
        basic_P300.config_color(refresh_rate=fps, stim_duration=stim_duration)
        basic_P300.config_index()
        basic_P300.config_response(bg_color=[0,0,0])
        bg_color = np.array([0, 0, 0])                              # background color
        display_time = 1
        index_time = 0.5
        response_time = 2
        rest_time = 0.5
        port_addr = None
        nrep = 1                                                    # Number of blocks
        lsl_source_id = None
        online = False
        ex.register_paradigm('basic P300', paradigm, VSObject=basic_P300, bg_color=bg_color, display_time=display_time,
            index_time=index_time, rest_time=rest_time, response_time=response_time, port_addr=port_addr, nrep=nrep,
            pdim='p300', lsl_source_id=lsl_source_id, online=online)

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)

    def config_color(
        self, refresh_rate=0, stim_duration=0.1, stim_ISI=0.025, stim_round=1
    ):
        """Configure P300 paradigm interface parameters, including screen refresh rate
        and row to column transition time interval.

        Parameters
        ----------
            refresh_rate: int
                Refresh rate of screen.
            stim_ duration: float
                The time interval for row and column transformation.

        """
        self.stim_duration = stim_duration
        self.stim_ISI = stim_ISI
        self.stim_round = stim_round
        self.refresh_rate = refresh_rate
        if refresh_rate == 0:
            self.refresh_rate = np.floor(
                self.win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20)
            )

        # highlight one row/ column onto interface
        row_pos = np.unique(self.stim_pos[:, 0])
        col_pos = np.unique(self.stim_pos[:, 1])
        [row_num, col_num] = [len(col_pos), len(row_pos)]
        # complete single trial
        self.stim_frames = int(
            (row_num * (stim_duration + stim_ISI) * refresh_rate)
        ) + int((col_num * (stim_duration + stim_ISI) * refresh_rate))

        # back png
        self.back = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
            "textures" + os.sep + "back.png",
        )
        self.back_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.back,
            elementMask=None,
            texRes=2,
            nElements=1,
            sizes=[self.win_size],
            xys=np.array([[0, 0]]),
            oris=[0],
            opacities=[1],
            contrs=[-1],
        )

        # start
        self.flash_stimuli = []
        self.order_index = np.zeros([int((row_num + col_num) * self.stim_round)])
        for round_num in range(self.stim_round):
            row_order_index = list(range(0, row_num))
            np.random.shuffle(row_order_index)
            col_order_index = list(range(0, col_num))
            np.random.shuffle(col_order_index)
            # reset event label
            l_row_order_index = [x + 1 for x in row_order_index]
            l_col_order_index = [x + row_num + 1 for x in col_order_index]

            order_row_col = np.array(l_row_order_index + l_col_order_index)
            # print(order_row_col.shape)
            self.order_index[
                (round_num * (row_num + col_num)): (
                    (round_num + 1) * (row_num + col_num)
                )
            ] = order_row_col[
                :
            ]  # event label
            # print(self.order_index)

            # Determine row and column char status
            stim_colors_row = np.zeros(
                [
                    (row_num * col_num),
                    int(row_num * refresh_rate * (stim_duration + stim_ISI)),
                    3,
                ]
            )
            stim_colors_col = np.zeros(
                [
                    (row_num * col_num),
                    int(col_num * refresh_rate * (stim_duration + stim_ISI)),
                    3,
                ]
            )  #
            row_label = np.zeros(
                [int(row_num * refresh_rate * (stim_duration + stim_ISI))]
            )
            col_label = np.zeros(
                [int(col_num * refresh_rate * (stim_duration + stim_ISI))]
            )

            tmp = 0
            for col_i in col_order_index:
                stim_colors_col[
                    (col_i * row_num): ((col_i + 1) * row_num),
                    int(tmp * refresh_rate * (stim_duration + stim_ISI)): int(
                        tmp * refresh_rate * (stim_duration + stim_ISI)
                        + refresh_rate * (stim_duration)
                    ),
                ] = [-1, -1, -1]
                col_label[int(tmp * refresh_rate * (stim_duration + stim_ISI))] = 1
                tmp += 1

            tmp = 0
            for row_i in row_order_index:
                for col_i in range(col_num):
                    stim_colors_row[
                        (row_i + row_num * col_i),
                        int(tmp * refresh_rate * (stim_duration + stim_ISI)): int(
                            tmp * refresh_rate * (stim_duration + stim_ISI)
                            + refresh_rate * stim_duration
                        ),
                    ] = [-1, -1, -1]
                    row_label[int(tmp * refresh_rate * (stim_duration + stim_ISI))] = 1
                tmp += 1

            stim_colors = np.concatenate((stim_colors_row, stim_colors_col), axis=1)
            self.roworcol_label = np.concatenate(
                (row_label, col_label), axis=0
            )  # each round is the same
            self.stim_colors = np.transpose(stim_colors, [1, 0, 2])

            # add flashing targets onto interface
            for sf in range(self.stim_frames):
                self.flash_stimuli.append(
                    visual.ElementArrayStim(
                        win=self.win,
                        units="pix",
                        nElements=self.n_elements,
                        opacities=np.ones((self.n_elements,)) * 0.7,
                        sizes=self.stim_sizes,
                        xys=self.stim_pos,
                        colors=self.stim_colors[sf, ...],
                        elementTex=np.ones((64, 64)),
                        elementMask=None,
                        texRes=48,
                    )
                )


# standard MI paradigm


class MI(VisualStim):
    """
    Create MI stimuli.

    The subclass MI inherits from the parent class VisualStim, and duplicate properties are no longer listed.

    author: Wei Zhao

    Created on: 2022-06-30

    update log:
        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        tex_left: str
            Obtain the image path for left hand stimulation.
        tex_right: str
            Obtain the image path for right hand stimulation.
        left_pos: list, shape(x, y)
            The position of left hand stimulation. Only exists in config_color().
        right_pos: list, shape(x, y)
            The position of right hand stimulation. Only exists in config_color().
        tex_left: str
            Obtain the image path for left hand stimulation. Only exists in config_color().
        refresh_rate: int
            The refresh rate of the screen. Only exists in config_color().
        text_stimulus: object
            Stimulus text, display "start" on the screen. Only exists in config_color().
        image_left_stimuli: object
            Left hand stimulation image, with colors indicating or starting to imagine.
            Only exists in config_color().
        image_right_stimuli: object
            Stimulate the image with the right hand, with colors indicating or starting to imagine.
            Only exists in config_color().
        normal_left_stimuli: object
            Left hand stimulation image, default color. Only exists in config_color().
        normal_right_stimuli: object
            Right hand stimulation image, default color. Only exists in config_color().
        response_left_stimuli: object
            Left hand stimulation image, color for online feedback. Only exists in config_color().
        response_right_stimuli: object
            Right hand stimulation image, color for online feedback. Only exists in config_color().

    Tip
    ----
    .. code-block:: python
        :caption: An example of creating MI stimuli.

        from psychopy import monitors
        import numpy as np
        from brainstim.framework import Experiment
        from brainstim.paradigm import MI,paradigm

        win = ex.get_window()

        # press q to exit paradigm interface
        fps = 120                                                   # Screen refresh rate
        text_pos = (0.0, 0.0)                                       # Prompt text position
        left_pos = [[-480, 0.0]]                                    # Left hand position
        right_pos = [[480, 0.0]]                                    # Right hand position
        tex_color = 2*np.array([179, 45, 0])/255-1                  # Prompt text color
        normal_color = [[-0.8,-0.8,-0.8]]                           # Default color
        image_color = [[1,1,1]]
        symbol_height = 100
        n_Elements = 1                                              # One on each hand
        stim_length = 288                                           # Length
        stim_width = 288                                            # Width
        basic_MI = MI(win=win)
        basic_MI.config_color(refresh_rate=fps, text_pos=text_pos, left_pos=left_pos, right_pos=right_pos, .
            tex_color=tex_color, normal_color=normal_color, image_color=image_color, symbol_height=symbol_height,
            n_Elements=n_Elements, stim_length=stim_length, stim_width=stim_width)
        basic_MI.config_response()
        bg_color = np.array([-1, -1, -1])                           # Background color
        display_time = 1
        index_time = 1
        rest_time = 1
        image_time = 4
        response_time = 2
        port_addr = None
        nrep = 10
        lsl_source_id =  None
        online = False
        ex.register_paradigm('basic MI', paradigm, VSObject=basic_MI, bg_color=bg_color, display_time=display_time,
            index_time=index_time, rest_time=rest_time, response_time=response_time, port_addr=port_addr,
            nrep=nrep, image_time=image_time, pdim='mi',lsl_source_id=lsl_source_id, online=online)

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)

        self.tex_left = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
            "textures" + os.sep + "left_hand.png",
        )
        self.tex_right = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
            "textures" + os.sep + "right_hand.png",
        )

    def config_color(
        self,
        refresh_rate=60,
        text_pos=(0.0, 0.0),
        left_pos=[[-480, 0.0]],
        right_pos=[[480, 0.0]],
        tex_color=(1, -1, -1),
        normal_color=[[-0.8, -0.8, 0.8]],
        image_color=[[1, 1, 1]],
        symbol_height=100,
        n_Elements=1,
        stim_length=288,
        stim_width=162,
    ):
        """Config color of stimuli.

        Parameters
        ----------
            refresh_rate: int
                Refresh rate of screen.
            text_pos: ndarray, shape(x, y)
                The position of the prompt text ("start").
            left_pos: ndarray, shape(x, y)
                The position of left hand stimulation.
            right _pos: ndarray, shape(x, y)
                The position of right hand stimulation.
            tex_color: ndarray, shape(red, green, blue)
               The color of the stimulating text, ranging from -1.0 to 1.0.
            normal_color: ndarray, shape(red, green, blue)
                The stimulating color during rest.
            image_color: ndarray, shape(red, green, blue)
                The stimulating color during imaging.
            symbol_height: float
                The height of the prompt text.
            n_Elements: int
                The number of left and right hand stimuli.
            stim_length: float
                The length of left and right hand stimulation
            stim_width=162: float
                The width of left and right hand stimulation.

        """

        self.n_Elements = n_Elements
        self.stim_length = stim_length
        self.stim_width = stim_width
        self.left_pos = left_pos
        self.right_pos = right_pos
        self.refresh_rate = refresh_rate
        if refresh_rate == 0:
            refresh_rate = np.floor(
                self.win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20)
            )

        if symbol_height == 0:
            symbol_height = int(self.win_size[1] / 6)
        self.text_stimulus = visual.TextStim(
            self.win,
            text="start",
            font="Times New Roman",
            pos=text_pos,
            color=tex_color,
            units="pix",
            height=symbol_height,
            bold=True,
        )

        self.image_left_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_left,
            elementMask=None,
            texRes=2,
            nElements=n_Elements,
            sizes=[[stim_length, stim_width]],
            xys=np.array(left_pos),
            oris=[0],
            colors=np.array(image_color),
            opacities=[1],
            contrs=[-1],
        )
        self.image_right_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_right,
            elementMask=None,
            texRes=2,
            nElements=n_Elements,
            sizes=[[stim_length, stim_width]],
            xys=np.array(right_pos),
            oris=[0],
            colors=np.array(image_color),
            opacities=[1],
            contrs=[-1],
        )

        self.normal_left_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_left,
            elementMask=None,
            texRes=2,
            nElements=n_Elements,
            sizes=[[stim_length, stim_width]],
            xys=np.array(left_pos),
            oris=[0],
            colors=np.array(normal_color),
            opacities=[1],
            contrs=[-1],
        )
        self.normal_right_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_right,
            elementMask=None,
            texRes=2,
            nElements=n_Elements,
            sizes=[[stim_length, stim_width]],
            xys=np.array(right_pos),
            oris=[0],
            colors=np.array(normal_color),
            opacities=[1],
            contrs=[-1],
        )

    def config_response(self, response_color=[[-0.5, 0.9, 0.5]]):
        self.response_left_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_left,
            elementMask=None,
            texRes=2,
            nElements=self.n_Elements,
            sizes=[[self.stim_length, self.stim_width]],
            xys=np.array(self.left_pos),
            oris=[0],
            colors=np.array(response_color),
            opacities=[1],
            contrs=[-1],
        )
        self.response_right_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_right,
            elementMask=None,
            texRes=2,
            nElements=self.n_Elements,
            sizes=[[self.stim_length, self.stim_width]],
            xys=np.array(self.right_pos),
            oris=[0],
            colors=np.array(response_color),
            opacities=[1],
            contrs=[-1],
        )


# standard AVEP paradigm


class AVEP(VisualStim):
    """Create AVEP stimuli.

    The subclass AVEP inherits from the parent class VisualStim, and duplicate attributes are no longer listed.

    author: Jieyu Wu

    Created on: 2022-12-14

    update log:
        2022-12-17 by Jieyu Wu

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        dot_shape: str
            The shape of the stimulus point can be set to 'circle' - circular, 'sqr' - rectangular,
            'cluster' - cluster of dots, etc.
        n_rep: int
            repetitions of stimulation.
        duty: float
            PWM of a single flicker.
        cluster_num: int
            The number of individual stimuli in cluster stimuli
        colorSpace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.
        cluster_number: int
            Number of dots in cluster of stimuli.

    Attributes
    ----------
        refresh_rate: int
            Screen refresh rate.
        stim_time: float
            Time of stimulus flicker.
        stim_color: list, shape(red, green, blue)
            The color of the stimulus block, ranging from -1.0 to 1.0.
        stim_opacities: float
            Opacity, default to opaque.
        stim_frames: int
            The number of frames included in a single trial stimulus.
        stim_oris: ndarray
            The direction of the stimulus block.
        stim_sfs: ndarray
            The spatial frequency of the stimulus block.
        stim_contrs: ndarray
            The contrast of the stimulus block.
        freqs: list, shape(fre, …)
            Stimulus block flicker frequency, length consistent with the number of stimulus blocks.
        stim_colors: list, shape(red, green, blue)
            The color configuration required for the stimulus block flashing.
        flash_stimuli:
            The configuration information required for the flashing of the stimulus block.
        stim_ary:
            Stimulus flicker sequences generated based on encoding.
        stim_dot_pos:
            The location of all stimulation points.

    Tip
    ----
    .. code-block:: python

        from psychopy import monitors
        import numpy as np
        from brainstim.framework import Experiment
        from brainstim.paradigm import AVEP,paradigm

        win = ex.get_window()

        # press q to exit paradigm interface
        n_elements, rows, columns = 32, 4, 8  # n_elements number of instructions
        stim_length, stim_width = 3, 3                               # the size of AVEP stimulation points
        tex_height = 40											# size of the AVEP instruction
        stim_color, tex_color = [0.7, 0.7, 0.7], [1, 1, 1]      # AVEP stimulus and text color
        fps = 60													# screen refresh rate
        stim_time = 1  											# stimulus duration
        stim_opacities = 1  										# stimulus contrast
        freqs = np.ones((n_elements)) * 10 							# frequency of instructions
        stim_num = 2												# number of stimulation points
        avep = AVEP(win=win,dot_shape='cluster')
        sequence = [avep.num2bin_ary(i,5) for i in range(n_elements)] # stimulus sequence
        if len(sequence) != n_elements:
            raise Exception('Incorrect spatial code amount!')
        avep.tex_height = tex_height
        avep.config_pos(n_elements=n_elements, rows=rows, columns=columns, stim_length=stim_length,
        stim_width=stim_width)
        avep.config_color(refresh_rate=fps, stim_time=stim_time, stimtype='sinusoid',
                    stim_color=stim_color,sequence=sequence,
                    stim_opacities=stim_opacities, freqs=freqs, stim_num=stim_num)

        avep.config_text(symbol_height = tex_height, tex_color=tex_color)
        avep.config_index(index_height=40)
        avep.config_response()
        bg_color = np.array([-1, -1, -1])						# background color
        display_time = 1									# Paradigm start 1s of warm duration
        index_time = 0.5										# cue length, distraction
        rest_time = 0.5										# length of rest after cueing
        response_time = 1										# online Feedback
        port_addr = None #  0xdefc								# capture host port
        nrep = 1												# number of blocks
        lsl_source_id = 'meta_online_worker'					# source id
        online = False # True									# sign of online experiments
        ex.register_paradigm('avep', paradigm, VSObject=avep, bg_color=bg_color, display_time=display_time,
            index_time=index_time, rest_time=rest_time, response_time=response_time, port_addr=port_addr,
            nrep=nrep, pdim='avep', lsl_source_id=lsl_source_id, online=online)

    """

    def __init__(
        self,
        win,
        dot_shape="circle",
        n_rep=5,
        duty=0.5,
        cluster_num=1,
        colorSpace="rgb",
        allowGUI=True,
    ):
        """Item class from VisualStim.

        Args:

        """
        self.dot_shape = dot_shape
        self.n_rep = n_rep
        if dot_shape == "cluster" and cluster_num == 1:
            self.cluster_num = 6
            dot_shape == "circle"
        elif dot_shape == "cluster":
            self.cluster_num = cluster_num
            dot_shape == "circle"
        else:
            self.cluster_num = cluster_num

        self.duty = duty
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)

    def config_array(self, frequencies=None):
        """Config the dot array according to the code sequences.

        Parameters
        ----------
            frequencies: array
                frequencies of each target.

        """
        stim_time = self.stim_time
        stim_frames = self.stim_frames
        n_element = self.n_elements
        sequence = self.sequence
        if frequencies is None:
            frequencies = 10 * np.ones((n_element, 1))
        t = np.linspace(0, stim_time, stim_frames, endpoint=False)
        stim_ary = [[] for i in range(n_element)]
        for target_i in range(n_element):
            tar_fre = frequencies[target_i]
            avep_num = int(tar_fre * stim_time)
            fold_num = int(np.ceil(avep_num / len(sequence[target_i])))
            tar_seq = np.tile(sequence[target_i], fold_num)[0:avep_num]
            sample = (signal.square(2 * pi * tar_fre * t, duty=self.duty) + 1) / 2
            sample = sample.astype(int)
            a = np.append(0, sample)
            b = np.diff(a)
            c = np.where(b == 1)
            c = np.append(c, sample.shape[0])
            d = np.array([], "int")
            for avep_i in range(avep_num):
                d = np.append(d, sample[c[avep_i]: c[avep_i + 1]] * tar_seq[avep_i])
            stim_ary[target_i] = d
        self.stim_ary = []
        for clu_i in range(self.cluster_num):
            self.stim_ary.extend(stim_ary)

    def config_dot_pos(self, offset=None):
        """Config the position of each single dot.

        Parameters
        ----------
            offset: list
                offset distance between stimulus point and target center.

        """
        if offset is None and self.stim_num == 2:
            offset = [[-20, -20], [20, -20]]
        elif offset is None and self.stim_num == 4:
            offset = [[20, 20], [-20, 20], [-20, -20], [20, -20]]
        elif offset is None and self.stim_num == 8:
            offset = [
                [20, 20],
                [0, 20],
                [-20, 20],
                [-20, 0],
                [-20, -20],
                [0, -20],
                [20, -20],
                [20, 0],
            ]
        elif len(offset) == self.stim_num:
            pass
        else:
            raise Exception("Please confirm the offset position list!")
        dot_pos = np.zeros((self.n_elements, self.stim_num, 2))
        for dot_i in range(self.stim_num):
            dot_pos[:, dot_i, :] = self.stim_pos
            dot_pos[:, dot_i, 0] = dot_pos[:, dot_i, 0] + offset[dot_i][0]
            dot_pos[:, dot_i, 1] = dot_pos[:, dot_i, 1] + offset[dot_i][1]
        if self.cluster_num == 1:
            self.stim_dot_pos = np.tile(
                dot_pos[np.newaxis, ...], (self.stim_frames, 1, 1, 1)
            )
        else:
            self.stim_dot_pos = np.zeros(
                (self.stim_frames, self.cluster_num * self.n_elements, self.stim_num, 2)
            )
            for stim_i in range(self.stim_frames):
                for clu_i in range(self.cluster_num):
                    width_rand = random.randint(-3, 3)
                    height_rand = random.randint(-3, 3)
                    self.stim_dot_pos[
                        stim_i,
                        clu_i * self.n_elements: (clu_i + 1) * self.n_elements,
                        :,
                        0,
                    ] = (
                        dot_pos[..., 0] + width_rand
                    )
                    self.stim_dot_pos[
                        stim_i,
                        clu_i * self.n_elements: (clu_i + 1) * self.n_elements,
                        :,
                        1,
                    ] = (
                        dot_pos[..., 1] + height_rand
                    )

    def config_dot_color(self):
        """Config color array according to dot array."""
        stim_num = self.stim_num
        stim_ary = self.stim_ary
        stim_colors = np.zeros(
            (self.stim_frames, self.n_elements * self.cluster_num, stim_num, 3)
        )
        for tar_i in range(self.n_elements * self.cluster_num):
            for frame_i in range(self.stim_frames):
                dot_type = stim_ary[tar_i][frame_i]
                stim_colors[frame_i, tar_i, :, :] = wave_new(
                    stim_num=stim_num, type=dot_type
                )
        self.stim_colors = stim_colors

    def config_color(
        self, refresh_rate, stim_time, stim_color, sequence, stim_opacities=1, **kwargs
    ):
        """Set AVEP paradigm interface parameters, including screen refresh rate, stimulus time, and stimulus color.

        Parameters
        ----------
            refresh_rate: int
                Refresh rate of screen.
            stim_time: float
                Time of each stimulus.
            stim_color: int
                The color of the stimulus point.
            sequence: list
                The encoding sequence that stimulates flickering, with a length consistent with
                the number of instructions.
            stim_opacities: float
                Opacities of each stimulus, default to opaque.
            freqs: list, shape(fre, …)
                The flicker frequency of the stimulus point, with length consistent with the number of instructions.
            stim_num: int
                The number of stimulus points around characters.

        Raises
        ----------
            Exception: Inconsistent frames and color matrices

        """

        # initialize extra inputs
        all_shapes = [
            "sin",
            "sqr",
            "saw",
            "tri",
            "sinXsin",
            "sqrXsqr",
            "circle",
            "gauss",
            "cross",
        ]
        self.refresh_rate = refresh_rate
        self.stim_time = stim_time
        self.stim_color = stim_color
        self.stim_opacities = stim_opacities
        self.stim_frames = int(stim_time * self.refresh_rate)
        self.sequence = sequence

        if refresh_rate == 0:
            self.refresh_rate = np.floor(
                self.win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20)
            )

        self.stim_oris = np.zeros((self.n_elements,))  # orientation
        self.stim_sfs = np.zeros((self.n_elements,))  # spatial frequency
        self.stim_contrs = np.ones((self.n_elements,))  # contrast

        # check extra inputs
        if "stim_oris" in kwargs.keys():
            self.stim_oris = kwargs["stim_oris"]
        if "stim_sfs" in kwargs.keys():
            self.stim_sfs = kwargs["stim_sfs"]
        if "stim_contrs" in kwargs.keys():
            self.stim_contrs = kwargs["stim_contrs"]
        if "freqs" in kwargs.keys():
            self.freqs = kwargs["freqs"]
        if "phases" in kwargs.keys():
            self.phases = kwargs["phases"]
        if "stim_num" in kwargs.keys():
            self.stim_num = kwargs["stim_num"]
        # config the location of each dot
        self.config_dot_pos()
        # create the coding array according to the input spatial codes
        self.config_array(frequencies=self.freqs)
        # generate the color array according to coding array of each stimulus
        self.config_dot_color()
        # the dot number equal to number of targets multiplied by the number of stimuli of avep
        # and the number of cluster dots
        all_dot_num = self.n_elements * self.stim_num * self.cluster_num
        stim_colors = np.concatenate(
            [self.stim_colors[:, :, i, :] for i in range(self.stim_num)], axis=1
        )
        stim_dot_pos = np.concatenate(
            [self.stim_dot_pos[:, :, i, :] for i in range(self.stim_num)], axis=1
        )
        stim_size = np.concatenate(
            [self.stim_sizes for i in range(self.stim_num * self.cluster_num)], axis=0
        )
        stim_oris = np.concatenate(
            [self.stim_oris for i in range(self.stim_num * self.cluster_num)], axis=0
        )
        stim_sfs = np.concatenate(
            [self.stim_sfs for i in range(self.stim_num * self.cluster_num)], axis=0
        )
        stim_contrs = np.concatenate(
            [self.stim_contrs for i in range(self.stim_num * self.cluster_num)], axis=0
        )
        if self.dot_shape == "cluster":
            dot_shape = "circle"
        elif self.dot_shape == "square":
            dot_shape = None
        elif self.dot_shape in all_shapes:
            dot_shape = self.dot_shape
        else:
            raise Exception("Please input the correct shape!")

        incorrect_frame = stim_colors.shape[0] != self.stim_frames
        incorrect_number = stim_colors.shape[1] != all_dot_num
        if incorrect_frame or incorrect_number:
            raise Exception("Incorrect color matrix or flash frames!")
        # add flashing targets onto interface
        self.flash_stimuli = []
        for sf in range(self.stim_frames):
            self.flash_stimuli.append(
                visual.ElementArrayStim(
                    win=self.win,
                    units="pix",
                    nElements=all_dot_num,
                    sizes=stim_size,
                    xys=stim_dot_pos[sf, ...],
                    colors=stim_colors[sf, ...],
                    opacities=self.stim_opacities,
                    oris=stim_oris,
                    sfs=stim_sfs,
                    contrs=stim_contrs,
                    elementTex=np.ones((64, 64)),
                    elementMask=dot_shape,
                    texRes=48,
                )
            )

    def num2bin_ary(self, num, n_elements, type="0-1"):
        """Converts a decimal number to a binary sequence of specified bit.
        The byte-codes of the binary sequence are 1 and 2

        author: Jieyu Wu

        Created on: 2022-12-16

        update log:
            2023-3-27 by Shihang Yu

            2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

        Parameters
        ----------
            num : int
                A decimal number.
            n_elements : int
                Num of stimulus.
            type : int
                if type is '0-1',convert each '0' to '1-2' and each '1' to '2-1'.
                else convert each '0' to '1' and each '1' to '2'.

        Returns
        -------
            bin_ary2: list
                (stim_num, 3)

        """
        bit = int(math.ceil(math.log(n_elements) / math.log(2)))
        bin_ary = np.zeros(bit, "int")
        quo = num
        i = -1
        while quo != 0:
            (quo, mod) = divmod(quo, 2)
            bin_ary[i] = mod
            i -= 1
        if type == "0-1":
            bin_ary2 = np.zeros(bit * 2, "int")
            elements = np.array([[0, 1], [1, 0]])
            for j in range(bit):
                bin_ary2[j * 2: (j + 1) * 2] = elements[int(bin_ary[j])]
        else:
            bin_ary2 = bin_ary

        return list(bin_ary2 + 1)


# standard SSaVEP paradigm


class SSAVEP(VisualStim):
    """
    Create SSAVEP stimuli.

    The subclass SSVEP is inherited from the parent class VisualStim, and the repeated attributes are no longer listed.

    author: Jieyu Wu

    Created on: 2023-09-11

    update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        n_elements: int
            The number of unique stimuli.
        n_members: int
            The number of sub-stimuli in an individual command.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        refresh_rate: int
            Screen refresh rate.
        stim_time: float
            Stimulate the time of flicker.
        stim_color: list, shape(red, green, blue)
            The color of the stimulus block was between-1.0 and 1.0.
        stim_opacities: float
            Opacity, the default opacity.
        stim_frames: int
            The number of frames contained in a single trial stimulus.
        stim_oris: ndarray
            The direction of the stimulus block.
        stim_sfs: ndarray
            The spatial frequency of the stimulus block.
        stim_contrs: ndarray
            Contrast of the stimulator block.
        freqs: list, shape(fre, …)
            The blinking frequency and length of the stimulus block are consistent
            with the number of stimulus blocks.
        phases: list, shape(phase, …)
            The stimulus block flashed phase, and the length was consistent with
            the number of stimulus blocks.
        stim_colors: list, shape(red, green, blue)
            The color configuration required for the flicker of the stimulus block.
        flash_stimuli:
            The configuration information required for the stimulus block flicker.

    Tip
    ----
    .. code-block:: python
       :caption: An example of using SSVEP Paradigm.

        from psychopy import monitors
        import numpy as np
        from brainstim.framework import Experiment
        from brainstim.paradigm import SSAVEP,paradigm

        # ex:Examples in the above, no longer repeat
        win = ex.get_window()

        # q:Quit paradigm interface
        n_elements, rows, columns = 20, 4, 5
        stim_length, stim_width = 150, 150
        stim_color, tex_color = [1,1,1], [1,1,1]
        fps = 120                  # screen refresh rate
        stim_time = 2              # stimulation duration
        stim_opacities = 1         # stimulation contrast
        freqs = np.arange(8, 16, 0.4) # the frequency of the instruction
        phases = np.array([i*0.35%2 for i in range(n_elements)]) # the phase of the instruction
        basic_ssvep = SSVEP(win=win)
        basic_ssvep.config_pos(n_elements=n_elements, rows=rows, columns=columns,
        stim_length=stim_length, stim_width=stim_width)
        basic_ssvep.config_text(tex_color=tex_color)
        basic_ssvep.config_color(refresh_rate=fps, stim_time=stim_time, stimtype='si
        nusoid', stim_color=stim_color, stim_opacities=stim_opacities, freqs=freqs,
        phases=phases)
        basic_ssvep.config_index()
        basic_ssvep.config_response()
        bg_color = np.array([-1, -1, -1])     # background color
        display_time = 1
        index_time = 0.5
        rest_time = 0.5
        response_time = 1
        port_addr = None           # acquisition host port
        nrep = 1
        lsl_source_id = None
        online = False
        ex.register_paradigm('basic SSVEP', paradigm, VSObject=basic_ssvep, bg_color
        =bg_color, display_time=display_time, index_time=index_time, rest_time=rest
        _time, response_time=response_time, port_addr=port_addr, nrep=nrep, pdim='s
        svep', lsl_source_id=lsl_source_id, online=online)

    """

    def __init__(
        self, win, n_elements=20, n_members=8, colorSpace="rgb", allowGUI=True
    ):
        self.n_members = n_members
        self.n_elements = n_elements
        self.n_groups = self.n_members * self.n_elements
        super().__init__(win, colorSpace, allowGUI)

    def config_member_pos(
        self,
        win,
        radius=0.1,
        angles=[0],
        outter_deg=4,
        inner_deg=1.5,
        tex_pix=128,
        sep_line_pix=16,
    ):
        """
        Config color of stimuli.

        Parameters
        ----------
            win: psychopy.visual.Window
                window this shape is being drawn to.
            radius: float
                related to sizes of stimulus.
            angles: list of float
                initial rotation angle of each sub-stimulus.
            outter_deg: float
                the ratio determining the size of two circles, used in self.generate_octants.
            inner_deg: float
                the ratio determining the size of two circles, used in self.generate_octants.
            tex_pix: int
                size of sub-stimulus, used in self.generate_octants

        """
        win_size = np.array(win.size)
        lpad, rpad = int(0.15 * win_size[1]), int(0.15 * win_size[1])
        upad, dpad = int(0.1 * win_size[1]), int(0.1 * win_size[1])
        x_stride = (win_size[0] - lpad - rpad) // (self.columns - 1)
        y_stride = (win_size[1] - upad - dpad) // (self.rows - 1)
        positions = np.array(
            [
                (
                    (lpad + (i % self.columns) * x_stride - win_size[0] / 2)
                    / win_size[1],
                    (win_size[1] / 2 - upad - (i // self.columns) * y_stride)
                    / win_size[1],
                )
                for i in range(self.n_elements)
            ]
        )
        self.positions_rad = positions
        self.stim_pos = self.positions_rad
        self.element_mask = np.zeros((self.n_groups, self.n_members), dtype=np.bool)
        self.radius = radius
        self.angles = angles
        self.angles = np.tile(np.reshape(self.angles, (-1, 1)), (1, self.n_members))
        self.stim_pos = np.array(self.stim_pos)
        self.positions = np.tile(positions, (1, self.n_members))
        self.member_angles = np.append(
            360 / self.n_members * np.arange(1, self.n_members), 0
        )
        # 计算每个圆环的位置
        oris = self.angles + self.member_angles.reshape((1, -1))
        oris = np.reshape(oris, (-1))
        self.oris = oris
        thetas = np.radians(oris)
        rotate_mat = np.array(
            [[np.cos(thetas), np.sin(thetas)], [-np.sin(thetas), np.cos(thetas)]]
        )
        self.member_positions = np.tensordot(
            rotate_mat, np.array([-0.5, 0.5]) * self.radius, axes=((1), (0))
        ).T
        self.member_positions = np.reshape(self.member_positions, (self.n_elements, -1))
        xys = self.positions + self.member_positions
        xys = np.reshape(xys, (-1, 2))
        self.element_pos = xys
        self.outter_deg = outter_deg
        self.inner_deg = inner_deg
        self.tex_pix = tex_pix
        self.sep_line_pix = sep_line_pix

    def config_stim(
        self,
        win,
        sizes=[[0.1, 0.1]],
        stim_color=[[1.0, 1.0, 1.0]],
        stim_opacities=[1],
        member_degree=None,
    ):
        """
        Config color of stimuli.

        Parameters
        ----------
            win: psychopy.visual.Window
                window this shape is being drawn to.
            sizes: list of float
                sizes of the state stimuli.
            stim_color: list of float
                rgb value of the state stimuli.
            stim_opacities: list of float
                opacities of the state stimuli.
            member_degree: list of float
                rotation angle of each sub-stimulus

        """
        self.generate_octants(
            win,
            outter_deg=self.outter_deg,
            inner_deg=self.inner_deg,
            member_degree=member_degree,
        )
        self.elementTex = "adaptive_octants.png"
        self.stim_oris = np.zeros((self.n_elements,))  # orientation
        self.stim_sfs = np.zeros((self.n_elements,))  # spatial frequency
        self.stim_contrs = [1]  # contrast
        self.stim_opacities = stim_opacities
        colors = np.tile(stim_color, (1, self.n_groups, 1))

        self.state_stim = self.create_elements(
            win,
            units="height",
            elementTex=self.elementTex,
            elementMask=None,
            nElements=self.n_groups,
            frames=1,
            sizes=sizes,
            xys=self.element_pos,
            oris=self.oris,
            colors=colors,
            opacities=[1],
            contrs=[1],
            texRes=2,
        )

    def config_ring(
        self, win, sizes=[[0.3, 0.3]], ring_colors=[1, 1, 1], opacities=[1.0]
    ):
        """
        Config color of rings around the stimuli.

        Parameters
        ----------
            win: psychopy.visual.Window
                window this shape is being drawn to.
            sizes: list of float
                sizes of each ring.
            ring_colors: list of float
                rgb value of rings.
            opacities: list of float
                opacities of center target.

        """
        _TEX = op.join(
            op.abspath(op.dirname(op.abspath(__file__))), "textures", "ring.png"
        )

        sizes = sizes
        ring_colors1 = np.tile(ring_colors, (1, self.n_elements, 1))
        self.ring = self.create_elements(
            win,
            units="height",
            elementTex=_TEX,
            elementMask=None,
            nElements=self.n_elements,
            frames=1,
            sizes=sizes,
            xys=self.positions_rad,
            colors=ring_colors1,
            opacities=opacities,
            contrs=[1],
            texRes=2,
        )

    def config_target(
        self, win, sizes=[[0.2, 0.2]], target_colors=[1, 0, 0], opacities=[1.0]
    ):
        """
        Config color of targets at the center of each stimulus.

        Parameters
        ----------
            win: psychopy.visual.Window
                window this shape is being drawn to.
            sizes: list of float
                sizes of each center target.
            target_colors: list of float
                rgb value of center targets.
            opacities: list of float
                opacities of center targets.

        """
        _TEX = op.join(
            op.abspath(op.dirname(op.abspath(__file__))), "textures", "centroid.png"
        )
        target_colors1 = np.tile(target_colors, (1, self.n_elements, 1))
        self.center_target = self.create_elements(
            win,
            units="height",
            elementTex=_TEX,
            elementMask=None,
            nElements=self.n_elements,
            frames=1,
            sizes=sizes,
            xys=self.positions_rad,
            oris=[45],
            colors=target_colors1,
            opacities=opacities,
            contrs=[1],
            texRes=2,
        )

    def config_flash_array(
        self,
        refresh_rate=60,
        freqs=[15],
        phases=[0],
        codes=[[0], [1], [2], [3]],
        stim_time_member=0.5,
        stim_color=[1, 1, 1],
        stimtype="sinusoid",
    ):
        """
        Config flash sequence array of stimuli.

        Parameters
        ----------
            refresh_rate: int or float
                Refresh rate of screen.
            freqs: list of float
                Frequencies of each stimulus.
            phases: list of float
                Phases of each stimulus.
            codes: list of list
                code sequences of each stimulus.
            stim_time_member: float
                Time of each sub-stimulus.
            stim_color: list of float
                Maximum rgb value during stimuli flicker.
            stimtype: str
                flashing mode of stimuli.

        """
        self.codes = np.array(codes)
        self.freqs = np.array(freqs)
        self.freqs = np.reshape(
            np.tile(np.reshape(self.freqs, (-1, 1)), (1, self.n_members)),
            (self.freqs.size * self.n_members, 1),
        )
        self.phases = np.array(phases)
        self.phases = np.reshape(
            np.tile(np.reshape(self.phases, (-1, 1)), (1, self.n_members)),
            (self.phases.size * self.n_members, 1),
        )
        self.refresh_rate = refresh_rate
        self.stim_frames_member = int(stim_time_member * self.refresh_rate)
        self.stim_time_member = stim_time_member
        self.stimtype = stimtype
        if stimtype == "sinusoid":
            self.stim_colors_member = (
                sinusoidal_sample(
                    freqs=self.freqs,
                    phases=self.phases,
                    srate=self.refresh_rate,
                    frames=self.stim_frames_member,
                    stim_color=stim_color,
                )
                - 1
            )
        self.n_sequence = np.shape(self.codes)[1]
        self.stim_time = self.stim_time_member * self.n_sequence
        self.stim_frames = self.stim_frames_member * self.n_sequence
        self.stim_colors1 = np.ones(
            (self.stim_frames_member, self.n_groups, self.n_sequence, 3)
        )
        for tar_idx in range(self.n_elements):
            tar_codes = self.codes[tar_idx]
            for seq_idx in range(self.n_sequence):
                for seq_group_idx in range(len(tar_codes[seq_idx])):
                    self.stim_colors1[
                        :,
                        tar_idx * self.n_members + tar_codes[seq_idx][seq_group_idx],
                        seq_idx,
                        :,
                    ] = self.stim_colors_member[
                        :,
                        tar_idx * self.n_members + tar_codes[seq_idx][seq_group_idx],
                        :,
                    ]
        self.stim_colors = np.concatenate(
            [self.stim_colors1[:, :, i, :] for i in range(self.n_sequence)], axis=0
        )

    def config_color(
        self,
        win,
        refresh_rate=60,
        freqs=[15],
        phases=[0],
        codes=[[0], [1], [2], [3]],
        stim_time_member=0.5,
        stim_color=[1.0, 1.0, 1.0],
        stimtype="sinusoid",
        sizes=[0.1, 0.1],
    ):
        """
        Config color of stimuli.

        Parameters
        ----------
            refresh_rate: int or float
                Refresh rate of screen.
            freqs: list of float
                Frequencies of each stimulus.
            phases: list of float
                Phases of each stimulus.
            codes: list of list
                code sequences of each stimulus.
            stim_time_member: float
                Time of each sub-stimulus.
            stim_color: list of float
                Maximum rgb value during stimuli flicker.
            stimtype: str
                flashing mode of stimuli .
            sizes: list of float
                sizes of each sub-stimulus.

        """
        self.config_flash_array(
            refresh_rate, freqs, phases, codes, stim_time_member, stim_color, stimtype
        )
        self.flash_stimuli = self.create_elements(
            win,
            units="height",
            elementTex=self.elementTex,
            elementMask=None,
            nElements=self.n_groups,
            frames=self.stim_frames,
            sizes=sizes,
            xys=self.element_pos,
            oris=self.oris,
            colors=self.stim_colors,
            opacities=self.stim_opacities,
            contrs=self.stim_contrs,
            texRes=2,
        )

    def generate_octants(self, win, outter_deg=4, inner_deg=2, member_degree=None):
        """
        Generate the sub-stimulus and save the .png file.

        Parameters
        ----------
            win: psychopy.visual.Window
                window this shape is being drawn to.
            radius: float
                related to sizes of stimulus.
            angles: list of float
                initial rotation angle of each sub-stimulus.
            outter_deg: float
                the ratio determining the size of two circles, used in self.generate_octants.
            inner_deg: float
                the ratio determining the size of two circles, used in self.generate_octants.
            member_degree: list of float
                rotation angle of differen sub-stimuli in a single stimulus.

        """
        if member_degree is None:
            member_degree = self.member_angles[0]
        win_size = win.size
        win_size = [1600, 900]
        win.color = [-1, -1, -1]
        radius = self.tex_pix / win_size[1]
        sep_line_height = self.sep_line_pix / win_size[1]
        ratio = np.tan(np.radians(inner_deg)) / np.tan(np.radians(outter_deg))
        edges = 256
        outter_circle = visual.Circle(
            win,
            units="height",
            radius=radius,
            edges=edges,
            fillColor="white",
            lineColor="white",
        )
        inner_circle = visual.Circle(
            win,
            units="height",
            radius=radius * ratio,
            edges=edges,
            fillColor="black",
            lineColor="black",
        )
        h_line = visual.Rect(
            win,
            units="height",
            size=(win_size[0] / win_size[1], sep_line_height),
            fillColor="black",
            lineColor=None,
            ori=0,
        )
        v_line = visual.Rect(
            win,
            units="height",
            size=(win_size[0] / win_size[1], sep_line_height),
            fillColor="black",
            lineColor=None,
            ori=90,
        )
        diag_line1 = visual.Rect(
            win,
            units="height",
            size=(win_size[0] / win_size[1], sep_line_height),
            fillColor="black",
            lineColor=None,
            ori=90 - member_degree + 0.001,
        )
        diag_line2 = visual.Rect(
            win,
            units="height",
            size=(win_size[0] / win_size[1], sep_line_height),
            fillColor="black",
            lineColor=None,
            ori=-90 + member_degree - 0.001,
        )
        semi_circle1 = SemiCircle(
            win,
            units="height",
            radius=radius * 1.2,
            edges=edges,
            fillColor="black",
            lineColor="black",
            ori=180 - member_degree,
        )
        semi_circle2 = SemiCircle(
            win,
            units="height",
            radius=radius * 1.2,
            edges=edges,
            fillColor="black",
            lineColor="black",
            ori=0,
        )
        stims = [
            outter_circle,
            inner_circle,
            h_line,
            v_line,
            diag_line1,
            diag_line2,
            semi_circle1,
            semi_circle2,
        ]
        # stims = [
        #     outter_circle, inner_circle
        # ]
        rect = [-2 * radius * win_size[1] / win_size[0], 0, 0, -2 * radius]
        screenshot = visual.BufferImageStim(win, stim=stims, buffer="back", rect=rect)
        image = Image.fromarray(np.array(screenshot.image), mode="RGB")
        image.putalpha(image.convert("L"))
        image.save("adaptive_octants.png")
        win.clearBuffer()

    def create_elements(
        self,
        win,
        units="pix",
        elementTex=None,
        elementMask=None,
        nElements=1,
        frames=1,
        sizes=[[0.1, 0.1]],
        xys=[[0, 0]],
        oris=[0],
        colors=[[1, 1, 1]],
        contrs=[1],
        opacities=[1],
        texRes=48,
    ):
        """
        create the specific elements.

        Parameters
        ----------
            win: psychopy.visual.Window
                window this shape is being drawn to.
            units: str
                units to use when drawing.
            elementTex: object
                texture data of the elements.
            nElements: int
                number of the elements.
            frames: int
                number of frames drawn.
            sizes: list of float
                size of the elements.
            xys: list of float
                position of the elements.
            oris: list of float
                rotation angle of the elements.
            colors: list of float
                colors of the elements.
            contrs: list of float
                contrast ratio of the elements.
            opacities: list of float
                opacities ratio of the elements.
            texRes: int
                the resolution of the texture

        """
        sizes = (
            np.repeat(sizes, nElements, axis=0) if len(sizes) == 1 else np.array(sizes)
        )
        xys = np.repeat(xys, nElements, axis=0) if len(xys) == 1 else np.array(xys)
        oris = np.repeat(oris, nElements, axis=0) if len(oris) == 1 else np.array(oris)

        contrs = (
            np.repeat(contrs, nElements, axis=0)
            if len(contrs) == 1
            else np.array(contrs)
        )
        opacities = (
            np.repeat(opacities, nElements, axis=0)
            if len(opacities) == 1
            else np.array(opacities)
        )
        stim = []
        for frame_i in range(frames):
            stim.append(
                visual.ElementArrayStim(
                    win,
                    units=units,
                    elementTex=elementTex,
                    elementMask=elementMask,
                    texRes=texRes,
                    nElements=nElements,
                    sizes=sizes,
                    xys=xys,
                    oris=oris,
                    colors=colors[frame_i, ...],
                    opacities=opacities,
                    contrs=contrs,
                )
            )

        return stim


class GetPlabel_MyTherad:
    """
    Open the sub-thread to obtain the online feedback label,
    which is used in the ' con-ssvep ' paradigm. In the traditional
    BCI online experiment, the feedback result is blocked after the
    single trial stimulation, and then the next trial stimulation is started.
    However, the continuous control paradigm does not need to wait for the online
    result, so the sub-thread is opened to receive the online feedback result.
    打开子线程获取在线反馈标签，该标签在 'con-ssvep ' 范式中使用。在传统的 BCI 在线实验中，反馈结果在单次试验刺激后被阻塞，然后开始下一次试验刺激。
    但是，continu-control 范式不需要等待在线结果，所以打开子线程接收在线反馈结果。
    author: Wei Zhao

    Created on: 2022-07-30

    update log:
        2022-08-10 by Wei Zhao

        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        inlet:
            Stream data online.

    Attributes
    ----------
        inlet:
            pylsl: The data flow realizes the communication between the online
            processing program and the stimulus presentation program.
            pylsl：数据流实现了在线处理程序和刺激呈现程序之间的通信。
        _exit:
            Make a thread wait for the notification of other threads.
            使线程等待其他线程的通知。
        online_text_pos: ndarray
            The corresponding position of online prediction results in Speller.
            拼写器中在线预测结果的相应位置！
        online_symbol_text:
            The corresponding letter in Speller for online prediction results.
            拼写器中在线预测结果的相应文本
        samples: list, shape(label)
            Online processing of predictive labels passed to stimulus programs.
            在线处理传递给刺激计划的预测标签。
        predict_id: int
            Online prediction labels.

    Tip
    ----
    .. code-block:: python
       :caption: An example of Opening the sub-thread to receive online feedback results

        MyTherad = GetPlabel_MyTherad(inlet)
        MyTherad.feedbackThread()
        MyTherad.stop_feedbackThread()

    """

    def __init__(self, inlet):
        self.inlet = inlet#用于接收来自在线处理程序的数据流
        self._exit = threading.Event()#用于向子线程发送停止信号


    def feedbackThread(self):
        """Start the thread."""
        self._t_loop = threading.Thread(
            target=self._inner_loop, name="get_predict_id_loop"
        )
        self._t_loop.start()

    def _inner_loop(self):
        """The inner loop in the thread.
        不断从inlet接收在线预测标签，并相应地更新online_text_pos和online_symbol_text变量"""
        self._exit.clear()#清除_exit事件
        #确保在循环开始之前,这些变量的初始值被正确设置
        global online_text_pos, online_symbol_text
        online_text_pos = copy(self.res_text_pos)
        online_symbol_text = copy(self.symbol_text)
        while not self._exit.is_set():
            try:
                samples, _ = self.inlet.pull_sample()#尝试从self.inlet对象(来自在线处理程序的数据流)拉取一个样本
                if samples:#所以动态停止传出的数据是存在这里
                    # online predict id
                    predict_id = int(samples[0]) - 1
                    online_text_pos = (
                        online_text_pos[0] + self.symbol_height / 3,
                        online_text_pos[1],
                    )
                    online_symbol_text = online_symbol_text + self.symbols[predict_id]

            except Exception:
                pass

    def stop_feedbackThread(self):
        """Stop the thread."""
        self._exit.set()
        self._t_loop.join()#t_loop完成后才进行关闭子线程

flag = []
class My_Therad:

    def __init__(self, inlet):
        self.inlet = inlet#用于接收来自在线处理程序的数据流

    def myrun(self):
        while True:
            samples, timestamp = self.inlet.pull_sample()
            if len(samples) >= 1:
                # 如果第二位为 True,则中断当前刺激并开始下一轮
                global flag
                flag = samples
                break


    # def mystop(self):
    #     global flag
    #     key, value = flag.popitem()
    #     flag["False"] = value

# basic experiment control


def paradigm(
    VSObject,#代表实验视觉刺激的对象
    win,
    bg_color,
    display_time=1.0,
    index_time=1.0,
    rest_time=0.5,
    response_time=2,#在线实验期间反馈显示的持续时间
    image_time=2,
    port_addr=9045,
    nrep=1,
    pdim="ssvep",
    lsl_source_id=None,#在线处理程序的通信ID
    online=None,
    device_type="NeuroScan",
):
    """
    The classical paradigm is implemented, the task flow is defined, the ' q '
    exit paradigm is clicked, and the start selection interface is returned.

    author: Wei Zhao

    Created on: 2022-07-30

    update log:

        2022-08-10 by Wei Zhao

        2022-08-03 by Shengfu Wen

        2022-12-05 by Jie Mei

        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        VSObject:
            Examples of the three paradigms.
        win:
            window.
        bg_color: ndarray
            Background color.
        fps: int
            Display refresh rate.
        display_time: float
            Keyboard display time before 1st index.
        index_time: float
            Indicator display time.
        rest_time: float, optional
            SSVEP and P300 paradigm: the time interval between the target cue and the start of the stimulus.
            MI paradigm: the time interval between the end of stimulus presentation and the target cue.
        respond_time: float, optional
            Feedback time during online experiment.
        image_time: float, optional,
            MI paradigm: Image time.
        port_addr:
             Computer port , hexadecimal or decimal.
        nrep: int
            Num of blocks.
        pdim: str
            One of the three paradigms can be 'ssvep ', ' p300 ', ' mi ' and ' con-ssvep '.
        mi_flag: bool
            Flag of MI paradigm.
        lsl_source_id: str
            The id of communication with the online processing program needs to be consistent between the two parties.
        online: bool
            Flag of online experiment.
        device_type: str
            See support device list in brainstim README file

    """

    if not _check_array_like(bg_color, 3):
        raise ValueError("bg_color should be 3 elements array-like object.")
    win.color = bg_color
    fps = VSObject.refresh_rate

    if device_type == "NeuroScan":
        port = NeuroScanPort(port_addr, use_serial=True) if port_addr else None
    elif device_type == "Neuracle":
        port = NeuraclePort(port_addr) if port_addr else None
    else:
        raise KeyError(
            "Unknown device type: {}, please check your input".format(device_type)
        )
    port_frame = int(0.05 * fps)

    # 该代码的目的是为实验设置视觉刺激和响应处理,并在实验在线进行时建立与外部数据源（如 EEG 放大器）的连接。
    # 采取的具体操作取决于范式维度和是否有可用的有效数据源。
    inlet = False
    if online:
        if (
            pdim == "ssvep"
            or pdim == "p300"
            or pdim == "con-ssvep"
            or pdim == "avep"
            or pdim == "ssavep"
        ):
            VSObject.text_response.text = copy(VSObject.reset_res_text)
            VSObject.text_response.pos = copy(VSObject.reset_res_pos)
            VSObject.res_text_pos = copy(VSObject.reset_res_pos)
            VSObject.symbol_text = copy(VSObject.reset_res_text)
            res_text_pos = VSObject.reset_res_pos
        if lsl_source_id:
            inlet = True
            streams = resolve_byprop(
                "source_id", lsl_source_id, timeout=5
            )  # Resolve all streams by source_id这通过 source_id 属性解析所有流,超时时间为 5 秒
            if not streams:
                return
            inlet = StreamInlet(streams[0])  # receive stream data

        # # 设置标签串口
        # trigger = Trigger('COM4')


    if pdim == "ssvep":
        # config experiment settings
        # 设置了实验的条件,并创建了一个试次处理器对象,用于在实验过程中按照随机顺序呈现这些条件
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        # 在每一帧中,会执行以下操作:
        # 如果 online 标志为 True,则绘制 VSObject.rect_response（矩形）和 VSObject.text_response（文本）这两个视觉刺激。
        # 遍历 VSObject.text_stimuli 列表,并绘制其中的所有文本刺激。
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)#如果有端口对象 port 存在,则将其设置为 0。

        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            # 线程方法
            # MyTherad = My_Therad(inlet)
            # MyTherad.daemon = True
            # MyTherad.myrun()

            # initialise index position（初始化提示索引的起始位置）
            # 在刺激位置的基础上,向上偏移 VSObject.stim_width / 2 的距离,得到最终的索引刺激位置。
            # 将 VSObject.index_stimuli 对象的位置设置为上一步计算得到的位置。
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + np.array([0, VSObject.stim_width / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting（在盯刺激快之前移动提示索引的位置）)
            iframe = 0
            while iframe < int(fps * index_time):
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            # phase II: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    if online:
                        VSObject.rect_response.draw()#响应矩形
                        VSObject.text_response.draw()#响应文本
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    iframe += 1
                    win.flip()

            # 打标签
            # trigger.write_event_with0(1)

            # 创建队列并打开inlet
            q = queue.Queue()
            inlet.open_stream()
            # print(datetime.datetime.now())
            # phase III: target stimulating
            # 在一定时间内(由 VSObject.stim_frames 决定)持续显示目标刺激,
            # 并在特定的帧数(由 port_frame 决定)向外部设备(由 port 表示)发送触发信号
            for sf in range(VSObject.stim_frames):
                if sf == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                elif sf == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if sf == port_frame and port:
                    port.setData(0)

                # start_time = datetime.datetime.now()  # 记录开始时间
                # 队列方法 直接接收
                sample_count = inlet.samples_available()
                if sample_count:
                    for i in range(sample_count):
                        samples, _ = inlet.pull_sample()
                        q.put(samples)
                    mark = q.get(block=False)[0]
                    if mark > 0:
                        break
                # end_time = datetime.datetime.now()   # 记录结束时间
                # print(f"耗时：{end_time - start_time}秒")
                VSObject.flash_stimuli[sf].draw()
                win.flip()

            # print(datetime.datetime.now())
            # phase IV: respond
            # 如果inlet对象可用,接收在线预测,更新VSObject.symbol_text和res_text_pos变量,并显示指定的response_time内的响应
            if inlet:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()

                predict_id = mark - 1
                mark = 0
                # samples, timestamp = inlet.pull_sample()
                # predict_id = int(samples[0]) - 1  # online predict id
                VSObject.symbol_text = (
                    VSObject.symbol_text + VSObject.symbols[predict_id]
                )
                res_text_pos = (
                    res_text_pos[0] + VSObject.symbol_height / 3,
                    res_text_pos[1],#更新 res_text_pos 变量,用于控制文本响应的位置
                )
                iframe = 0
                while iframe < int(fps * response_time):
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = VSObject.symbol_text
                    VSObject.text_response.pos = res_text_pos
                    VSObject.text_response.draw()
                    iframe += 1
                    win.flip()


    elif pdim == "avep":
        # config experiment settings
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="random")
        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            iframe += 1
            win.flip()
        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break
            # initialise index position
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + np.array([0, VSObject.tex_height / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            for round_i in range(VSObject.n_rep):
                # phase II: rest state
                if rest_time != 0:
                    iframe = 0
                    while iframe < int(fps * rest_time):
                        if iframe == 0 and port and online and round_i == 0:
                            VSObject.win.callOnFlip(port.setData, 2)
                        elif iframe == 0 and port and round_i == 0:
                            VSObject.win.callOnFlip(port.setData, id + 1)
                            print(id + 1)
                        if iframe == port_frame and port and round_i == 0:
                            port.setData(0)

                        if online:
                            VSObject.rect_response.draw()
                            VSObject.text_response.draw()
                        for text_stimulus in VSObject.text_stimuli:
                            text_stimulus.draw()
                        iframe += 1
                        win.flip()

                # phase III: target stimulating
                for sf in range(VSObject.stim_frames):
                    if sf == 0 and port and online:
                        VSObject.win.callOnFlip(port.setData, 2)
                    elif sf == 0 and port:
                        VSObject.win.callOnFlip(port.setData, round_i + 1)
                    if sf == port_frame and port:
                        port.setData(0)
                    VSObject.flash_stimuli[sf].draw()
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    win.flip()

            # phase IV: respond
            if inlet:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()
                samples, timestamp = inlet.pull_sample()
                predict_id = int(samples[0]) - 1  # online predict id
                VSObject.symbol_text = (
                    VSObject.symbol_text + VSObject.symbols[predict_id]
                )
                res_text_pos = (
                    res_text_pos[0] + VSObject.symbol_height / 3,
                    res_text_pos[1],
                )
                iframe = 0
                while iframe < int(fps * response_time):
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = VSObject.symbol_text
                    VSObject.text_response.pos = res_text_pos
                    VSObject.text_response.draw()
                    iframe += 1
                    win.flip()

    elif pdim == "p300":
        # config experiment settings
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            VSObject.back_stimuli.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            # initialise index position
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + np.array([0, VSObject.stim_width / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                if iframe == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, id + 21)
                elif iframe == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 21)
                if iframe == port_frame and port:
                    port.setData(0)
                VSObject.back_stimuli.draw()
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            # phase II: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    VSObject.back_stimuli.draw()
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    iframe += 1
                    win.flip()

            # phase III: target stimulating

            tmp = 0
            nonzeros_label = 0
            for round_num in range(VSObject.stim_round):
                for sf in range(VSObject.stim_frames):
                    if VSObject.roworcol_label[sf] > 0 and port:
                        VSObject.win.callOnFlip(
                            port.setData,
                            VSObject.order_index[tmp],
                        )
                        nonzeros_label = sf
                        tmp += 1

                        # time_recrod.append(time.time())
                        # T = time_recrod[-1] - time_recrod[-2]
                        # print('P3:%s毫秒' % ((T)*1000))
                    if (sf - nonzeros_label) > port_frame and port:
                        port.setData(0)

                    # for text_stimulus in VSObject.text_stimuli:
                    #     text_stimulus.draw()
                    VSObject.back_stimuli.draw()
                    VSObject.flash_stimuli[
                        round(round_num * VSObject.stim_frames + sf)
                    ].draw()
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    win.flip()

            # phase IV: respond
            if inlet:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()
                samples, timestamp = inlet.pull_sample()
                predict_id = int(samples[0]) - 1  # online predict id
                VSObject.symbol_text = (
                    VSObject.symbol_text + VSObject.symbols[predict_id]
                )
                res_text_pos = (
                    res_text_pos[0] + VSObject.symbol_height / 3,
                    res_text_pos[1],
                )
                iframe = 0
                while iframe < int(fps * response_time):
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = VSObject.symbol_text
                    VSObject.text_response.pos = res_text_pos
                    VSObject.text_response.draw()
                    iframe += 1
                    win.flip()

    elif pdim == "mi":
        # config experiment settings
        conditions = [
            {"id": 0, "name": "left_hand"},
            {"id": 1, "name": "right_hand"},
            # {"id": 2, "name": "both_hands"},
        ]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            VSObject.normal_left_stimuli.draw()
            VSObject.normal_right_stimuli.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            # initialise index position
            id = int(trial["id"])
            if id == 0:
                image_stimuli = [VSObject.image_left_stimuli]
                normal_stimuli = [VSObject.normal_right_stimuli]
            elif id == 1:
                image_stimuli = [VSObject.image_right_stimuli]
                normal_stimuli = [VSObject.normal_left_stimuli]
            else:
                image_stimuli = [
                    VSObject.image_left_stimuli,
                    VSObject.image_right_stimuli,
                ]
                normal_stimuli = []

            # phase I: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    VSObject.normal_left_stimuli.draw()
                    VSObject.normal_right_stimuli.draw()
                    iframe += 1
                    win.flip()

            # phase II: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                for _image_stimuli in image_stimuli:
                    _image_stimuli.draw()
                if normal_stimuli:
                    for _normal_stimuli in normal_stimuli:
                        _normal_stimuli.draw()
                iframe += 1
                win.flip()

            # phase III: target stimulating
            iframe = 0
            while iframe < int(fps * image_time):
                if iframe == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                elif iframe == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if iframe == port_frame and port:
                    port.setData(0)
                VSObject.text_stimulus.draw()
                for _image_stimuli in image_stimuli:
                    _image_stimuli.draw()
                if normal_stimuli:
                    for _normal_stimuli in normal_stimuli:
                        _normal_stimuli.draw()
                iframe += 1
                win.flip()

            # phase IV: respond
            if inlet:
                VSObject.normal_left_stimuli.draw()
                VSObject.normal_right_stimuli.draw()
                win.flip()

                samples, timestamp = inlet.pull_sample()
                predict_id = int(samples[0]) - 1  # online predict id

                if predict_id == 0:
                    response_stimuli = [VSObject.response_left_stimuli]
                    normal_stimuli = [VSObject.normal_right_stimuli]
                elif predict_id == 1:
                    response_stimuli = [VSObject.response_right_stimuli]
                    normal_stimuli = [VSObject.normal_left_stimuli]
                else:
                    response_stimuli = [
                        VSObject.response_left_stimuli,
                        VSObject.response_right_stimuli,
                    ]
                    normal_stimuli = []

                iframe = 0
                while iframe < int(fps * response_time):
                    for _response_stimuli in response_stimuli:
                        _response_stimuli.draw()
                    if normal_stimuli:
                        for _normal_stimuli in normal_stimuli:
                            _normal_stimuli.draw()
                    iframe += 1
                    win.flip()

    elif pdim == "con-ssvep":
        global online_text_pos, online_symbol_text

        if inlet:
            MyTherad = GetPlabel_MyTherad(inlet)
            MyTherad.feedbackThread()

        # config experiment settings
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                MyTherad.stop_feedbackThread()
                break

            # initialise index position
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + np.array([0, VSObject.stim_width / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = online_symbol_text
                    VSObject.text_response.pos = online_text_pos
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            # phase II: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.text = online_symbol_text
                        VSObject.text_response.pos = online_text_pos
                        VSObject.text_response.draw()
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    iframe += 1
                    win.flip()

            # phase III: target stimulating
            for sf in range(VSObject.stim_frames):
                if sf == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, 1)
                elif sf == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if sf == port_frame and port:
                    port.setData(0)
                VSObject.flash_stimuli[sf].draw()
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = online_symbol_text
                    VSObject.text_response.pos = online_text_pos
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()

        if inlet:
            MyTherad.stop_feedbackThread()

    elif pdim == "ssavep":
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            VSObject.ring[0].draw()
            VSObject.state_stim[0].draw()
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            VSObject.ring[0].draw()
            VSObject.state_stim[0].draw()
            # VSObject.center_target[0].draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            # initialise index position
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + np.array([0, VSObject.stim_width / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.ring[0].draw()
                VSObject.state_stim[0].draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            # phase II: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    VSObject.ring[0].draw()
                    VSObject.state_stim[0].draw()
                    iframe += 1
                    win.flip()

            # phase III: target stimulating
            for sf in range(VSObject.stim_frames):
                if sf == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                elif sf == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if sf == port_frame and port:
                    port.setData(0)
                VSObject.flash_stimuli[sf].draw()
                VSObject.ring[0].draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()

            # phase IV: respond
            if inlet:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
                VSObject.ring[0].draw()
                VSObject.state_stim[0].draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()
                samples, timestamp = inlet.pull_sample()
                predict_id = int(samples[0]) - 1  # online predict id
                VSObject.symbol_text = (
                    VSObject.symbol_text + VSObject.symbols[predict_id]
                )
                res_text_pos = (
                    res_text_pos[0] + VSObject.symbol_height / 3,
                    res_text_pos[1],
                )
                iframe = 0
                while iframe < int(fps * response_time):
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = VSObject.symbol_text
                    VSObject.text_response.pos = res_text_pos
                    VSObject.text_response.draw()
                    iframe += 1
                    win.flip()
