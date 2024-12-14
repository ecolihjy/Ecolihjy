import time
import serial
import numpy as np

class Trigger():
    def __init__(self, port, bps=115200):
        """ """
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