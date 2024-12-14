from psychopy import monitors
import numpy as np
from framework import Experiment
from paradigm import SSVEP, paradigm, GetPlabel_MyTherad
from psychopy.tools.monitorunittools import deg2pix
from math import sin, cos, pi

if __name__ == "__main__":
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
        verbose=False,
    )
    mon.setSizePix([1920, 1080])  # 显示器的分辨率
    mon.save()
    bg_color_warm = np.array([0, 0, 0])#设置背景颜色
    win_size = np.array([1920, 1080])
    # esc/q退出开始选择界面
    # Experiment 对象，并配置了监视器、背景颜色、屏幕 ID、窗口大小和其他设置。
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0, # 值 0 通常指主要或默认显示器
        win_size=win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
        is_fullscr=False,  # True全窗口,此时win_size参数默认屏幕分辨率
        record_frames=False,  # 实验将记录在实验过程中显示的帧
        disable_gc=False,
        process_priority="normal", # 设置了实验进程的优先级
        use_fbo=False,
    )
    win = ex.get_window() # 允许实验代码访问和操作与 Experiment 实例关联的窗口

    # press q to exit paradigm interface
    n_elements, rows, columns = 20, 4, 5
    stim_length, stim_width = 150, 150
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]
    fps = 60                                                   # screen refresh rate
    stim_time = 2                                               # stimulus duration
    stim_opacities = 1                                          # stimulus contrast
    freqs = np.arange(8, 16, 0.4)                               # Frequency of instruction
    phases = np.array([i*0.35 % 2 for i in range(n_elements)])    # Phase of the instruction

    # # 计算每个刺激块的坐标(圆形)
    # side_length = 600
    # stim_pos = []
    # for i in range(n_elements):
    #     theta = 2 * pi * i / n_elements
    #     stim_pos.append([side_length / 2 * cos(theta), side_length / 2 * sin(theta)])
    #
    # # symbols的内容
    # symbols = ['', 'I', '', 'L', 'O', 'V', 'E', '', 'A', 'P', 'P', 'L', 'E', '', 'O', 'H', 'Y', 'E', 'A', 'H']

    basic_ssvep = SSVEP(win=win) #
    basic_ssvep.config_pos(n_elements=n_elements, rows=rows, columns=columns,
        stim_length=stim_length, stim_width=stim_width)
    # basic_ssvep.config_pos(n_elements=n_elements, rows=rows, columns=columns,
    #     stim_pos=stim_pos, stim_length=stim_length, stim_width=stim_width)
    basic_ssvep.config_text(tex_color=tex_color)
    basic_ssvep.config_color(refresh_rate=fps, stim_time=stim_time, stimtype='sinusoid',
        stim_color=stim_color, stim_opacities=stim_opacities, freqs=freqs, phases=phases)
    basic_ssvep.config_index()
    basic_ssvep.config_response()#
    bg_color = np.array([-1, -1, -1])                           # background color
    display_time = 1
    index_time = 0.5
    rest_time = 0.5
    response_time = 1
    port_addr = None 			                                 # Collect host ports
    nrep = 1 # 初始化为 1 意味着默认情况下只需要执行一次
    lsl_source_id = 'meta_online_worker'
    # lsl_source_id = None#实验室流媒体层（LSL）流的源ID，用于实时数据采集
    online = True
    ex.register_paradigm('basic SSVEP', paradigm, VSObject=basic_ssvep, bg_color=bg_color,
        display_time=display_time,  index_time=index_time, rest_time=rest_time, response_time=response_time,
        port_addr=port_addr, nrep=nrep,  pdim='ssvep', lsl_source_id=lsl_source_id, online=online)

    ex.run()