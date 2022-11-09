"""
Created on Wed Nov  9 16:42:23 2022

@author: anish narkar
"""

import logging
import numpy as np

from pupil_detectors import DetectorBase
from pyglui import ui

from methods import normalize

from plugin import Plugin

import zmq
import msgpack

logger = logging.getLogger(__name__)


class MyCustomPlugin(Plugin):
    
    
    def __init__(self, g_pool):
        super().__init__(g_pool)
        
        
    def recent_events(self, events):


        if self.run == True:
            if 'frame' in events:
                frame = events['frame']
                img = frame.img
                self.frame = img           
                self.input_q.put(img)
        
        ctx = zmq.Context()
        pupil_remote = ctx.socket(zmq.REQ)
        ip = 'localhost'
        port = 50020
        pupil_remote.connect(f'tcp://{ip}:{port}')
        
        # Request 'SUB_PORT' for reading data
        pupil_remote.send_string('SUB_PORT')
        sub_port = pupil_remote.recv_string()
        
        subscriber = ctx.socket(zmq.SUB)
        subscriber.connect(f'tcp://{ip}:{sub_port}')
        subscriber.subscribe('gaze.')  # receive all gaze messages
        
        
        # Request 'PUB_PORT' for writing data
        pupil_remote.send_string('PUB_PORT')
        pub_port = pupil_remote.recv_string()
         
        
                
                
                