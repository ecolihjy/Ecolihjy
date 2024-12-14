# -*- coding: UTF-8 -*-
import viz
import vizshape
import vizact          
import math
from my_package.Dynamic_program import MazeSolver
import threading
import viztask
import time
import serial
import numpy as np
# 初始化 Vizard
viz.go(viz.FULLSCREEN)

#生成障碍物的函数
def generate_barrier(position):
    '''
    create red barrier and replace it 
    size:[length,width,height]
    position:[X,Y,Z]
    x朝向北，Z朝向西，Y朝向正上
    '''
    # 创建一个红色的圆柱
    circle = vizshape.addCylinder(height=0.2, radius=0.3, axis=vizshape.AXIS_Y, color=viz.RED)
    # 将圆柱放置在(0, 0, 0)的位置
    circle.setPosition(position)

#生成路径的函数
def generate_path(position):
    # 创建一个绿色的圆柱
    circle = vizshape.addCylinder(height=0.2, radius=0.3, axis=vizshape.AXIS_Y, color=viz.GREEN)
    # 将圆柱放置在(0, 0, 0)的位置
    circle.setPosition(position)
    
# 创建环境地图的函数
def map_generate(size,scale):
    grid = vizshape.addGrid(size=[size, size], step=size/scale)
    return grid

#grid = vizshape.addGrid(size=[4,4], step=1)
#障碍物坐标
obstacle_positions = [(0, 0, 16), (2, 0, 4), (4, 0, 12), (6, 0, 0), (6, 0, 20), (10, 0, 2), (10, 0, 16), (10, 0, 18), (12, 0, 12), 
(14, 0, 10), (14, 0, 14), (14, 0, 18), (16, 0, 4), (16, 0, 8), (16, 0, 14), (18, 0, 20), (20, 0, 8), (20, 0, 14)]

#路径
path_positions = [(0, 0, 0), (2, 0, 2), (4, 0, 4), (6, 0, 6), (8, 0, 8), (10, 0, 10), (10, 0, 12), (12, 0, 14), (14, 0, 16), (16, 0, 18), (18, 0, 18), (20, 0, 20)]
'''for position in obstacle_positions:
    generate_barrier(position)
for position in path_positions:
    generate_path(position)'''
    

#grid = map_generate(20,10)

#设置地图坐标
#grid.setPosition([2, 0, 2])
#添加小车
car = viz.addChild('Porsche_911_GT2.obj')
car.setPosition([0, 0, 0])
car.setScale(0.25, 0.25, 0.25)

# 设置相机的位置为在网格上方，朝向网格中心
viz.MainView.setPosition([1, 10, 1])
viz.MainView.lookAt([0, 0, 0])

# 创建一个链接将主视图链接到小车
viewLink = viz.link(car, viz.MainView)
# 设置视图相对于小车的位置，例如10单位高，正上方
#viewLink.setOffset([1.5, 10, 1.5])
viewLink.setOffset([0, 10, 0])

# 设置视图的旋转角度，使其始终向下看
viewLink.setEuler([270, 90, 0])

# 创建八个刺激块并链接到小车
 
cues = []
for i in range(8):
    cue = vizshape.addCylinder(height=0.01, radius=0.5, axis=vizshape.AXIS_Y, color=viz.RED)  # 创建一个圆形的色块
    cues.append(cue)
    link = viz.link(car, cue)  # 将小车链接到每一个色块
    # 设置链接的偏移量，使色块围绕小车形成一个圆形
    link.setOffset([1.5*math.sin(i * math.pi / 4), 0, 1.5*math.cos(i * math.pi / 4)])  
    for cue in cues:
        cue.visible(viz.OFF)
    
blocks = []
for i in range(8):
    block = vizshape.addCylinder(height=0.01, radius=0.5, axis=vizshape.AXIS_Y, color=viz.GRAY)  # 创建一个圆形的色块
    blocks.append(block)
    link = viz.link(car, block)  # 将小车链接到每一个色块
    # 设置链接的偏移量，使色块围绕小车形成一个圆形
    link.setOffset([1.5*math.sin(i * math.pi / 4), 0, 1.5*math.cos(i * math.pi / 4)]) 


class Trigger():
    def __init__(self, port, bps=115200):
        """ 
        Send events using serial port.
        :param:
            port: port name for writing events
            bps: baud rate
        """
        super().__init__()
        self._port = port
        self._bps = bps
        self._serial = serial.Serial(port=self._port, baudrate=self._bps, timeout=1)
        self._serial.write([0])
        print(f"Trigger serial initialized with port ({self._port}) and baudrate ({self._bps})")

    def write_event(self, event):
        """ """
        self._serial.write([event])
        if event != 0:
            print(event, flush=True)
    
    def write_event_with0(self, event):
        """ """
        self._serial.write([event])
        if event != 0:
            print(event, flush=True)
        time.sleep(0.001)
        self._serial.write([0])


# 定义闪烁的频率和初相位
frequencies = [8.4, 9.0, 9.6, 10.2, 10.8, 11.4, 12.0, 12.6]    # 频率
phases =  [1, 0.5, 0, 1.5, 1, 0.5, 0, 1.5]  # 初相位
labels=[1,2,3,4,5,6,7,8]
trigger = Trigger('COM4')
# 创建一个任务来更新色块的灰度
def get_filcker_code(duration, freq, phase, fps):
        """ Generate SSVEP flickering code using joint frequency-phase modulation (JFPM). """
        idx = np.arange(int(duration*fps))
        code = (np.sin(2*np.pi*freq*(idx/fps) + phase*np.pi)+1)/2
        return code
        
        
def updateBlocksTask(label):
    print('updateBlocksTask label:{}'.format(label))
    trigger.write_event_with0(label)
    codes = [get_filcker_code(0.4, freq, phase, 60) for freq, phase in zip(frequencies, phases)]
    for intensity_values in zip(*codes):
        for block, intensity in zip(blocks, intensity_values):
            block.color([intensity, intensity, intensity])
        yield
        for block in blocks:
            block.color(viz.GRAY)
            

def cueTask(labels): 
    lables = list(labels)  # 创建labels的副本
    for label in labels:
        for i, cue in enumerate(cues):
            if i == label-1:
                cue.visible(viz.ON)
            else:
                cue.visible(viz.OFF)
        yield viztask.waitTime(1)  # 等待1秒
        cues[label-1].visible(viz.OFF)
        #print('labels:{}'.format(label))
        yield from updateBlocksTask(label)


def mainTask():
    while True:
        yield viztask.waitKeyDown(' ')  # 等待按下空格键
        for _ in range(6):
            yield cueTask(labels)
            #print("block:{}".format(_))
        #yield viztask.waitFrame(0.2)

viztask.schedule(mainTask())
 
