from plugin import Plugin
import cv2
import numpy as np

CUSTOM_TOPIC = "custom_topic"


class AnishNarkar(Plugin):
    order = 0.01
    def recent_events(self, events):
       
        if 'frame' in events:
            frame = events['frame']
            img = frame.img
            #self.frame = img           
            #self.input_q.put(img)
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            np.save('AnishNarkarArray.npy', np.asanyarray(frame_rgb)) 
                
                