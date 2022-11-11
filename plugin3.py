import cv2
import msgpack
import numpy as np

from plugin import Plugin
from pyglui import ui
from pupil_apriltags import Detector


CUSTOM_TOPIC = 'anish_clahe'
class CLAHE_Anish(Plugin):
    order = 0.1

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return

        gray = cv2.cvtColor(frame.img, cv2.COLOR_BGR2GRAY)

        
        

        old = np.asarray(frame.img).tolist()
        frame_rgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        frame_rgb = frame_rgb*0
        frame.img[:] = frame_rgb
        new = np.asarray(frame.img).tolist()
        
        custom_datum = {
            "topic": CUSTOM_TOPIC,
            "oldImage": msgpack.packb(old, use_bin_type=True),
            "newImage": msgpack.packb(new, use_bin_type=True),
            # Further fields can be added here.
            # Their values should be serializable with msgpack.
        }
            
        events[CUSTOM_TOPIC] = [custom_datum]

    