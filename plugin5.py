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
import cv2
import numpy as np

logger = logging.getLogger(__name__)

CUSTOM_TOPIC = "all_events"

class AllEvents(Plugin):
    
    
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.order = 0.001
        self.frame = None
        
        
    def recent_events(self, events):
        
        keys = events.keys()
        custom_datum = {
        "topic": CUSTOM_TOPIC,
        "timestamp": self.g_pool.get_timestamp(),  # Timestamp in pupil time
        "oldImage": msgpack.packb(keys, use_bin_type=True),
        # Further fields can be added here.
        # Their values should be serializable with msgpack.
    }
        events[CUSTOM_TOPIC] = [custom_datum]
                      
            
        
        
         
        
                
                
                