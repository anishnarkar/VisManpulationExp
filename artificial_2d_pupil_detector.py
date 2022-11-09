import logging
import numpy as np

from pupil_detectors import DetectorBase
from pyglui import ui

from methods import normalize

from pupil_detector_plugins.detector_base_plugin import PupilDetectorPlugin
from pupil_detector_plugins.visualizer_2d import draw_pupil_outline


logger = logging.getLogger(__name__)


class Detector(DetectorBase):
    """
    This is a custom DetectorBase subclass that returns an artificial pupil
    centre that moves around on a circle
    """

    def __init__(self):
        # here we will generate points on a circle with radius 20
        pi = np.pi
        r = 10
        n = 100
        self.pnts = [
            (np.cos(2 * pi / n * x) * r, np.sin(2 * pi / n * x) * r)
            for x in range(0, n + 1)
        ]
        self.stop_ind = len(self.pnts) - 1
        self.ind = 0

    def get_circle_coord(self):
        # retreive a new point from the circle on every frame refresh
        coords = self.pnts[self.ind]
        if self.ind < self.stop_ind:
            self.ind += 1
        else:
            self.ind = 0
        return coords

    def detect(self):
        # here we override the detect method with our own custom detector
        # this is a random artificial pupil center
        center = (90, 90)
        # we move the center around the circle here
        center = tuple(sum(x) for x in zip(center, self.get_circle_coord()))
        return center


class ArtificialDetector2DPlugin(PupilDetectorPlugin):
    uniqueness = "by_class"
    icon_chr = "2D"

    label = "Artificial  2d detector"
    identifier = "custom-2d"
    order = 0.5

    @property
    def pretty_class_name(self):
        return "Artificial Anish Pupil Detector 2D"

    @property
    def pupil_detector(self) -> Detector:
        return self.__detector

    def __init__(self, g_pool=None):
        super().__init__(g_pool=g_pool)
        self.__detector = Detector()
        self._stop_other_pupil_detectors()

    def _stop_other_pupil_detectors(self):
        plugin_list = self.g_pool.plugins

        # Deactivate other PupilDetectorPlugin instances
        for plugin in plugin_list:
            if isinstance(plugin, PupilDetectorPlugin) and plugin is not self:
                plugin.alive = False

        # Force Plugin_List to remove deactivated plugins
        plugin_list.clean()

    def detect(self, frame, **kwargs):
        # our DetectorBase subclass detect method returns a tuple here
        # containing an artificial pupil center
        center = self.__detector.detect()

        # add the artificial pupil center and other variables to a Pupil
        # datum, i.e. dict with Pupil specific keys
        diameter = 30.0
        result = {
            "ellipse": {"center": center, "axes": (diameter, diameter), "angle": 90}
        }

        result["confidence"] = 1.0
        eye_id = self.g_pool.eye_id
        result["id"] = eye_id
        result["topic"] = f"pupil.{eye_id}.{self.identifier}"
        result["method"] = "artificial-2d"
        result["timestamp"] = frame.timestamp
        result["diameter"] = diameter

        # add artificial pupil center to norm_pos calculation
        result["norm_pos"] = normalize(center, (frame.width, frame.height), flip_y=True)
        return result

    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name
        info = ui.Info_Text("Artificial 2D Pupil Detector Plugin")
        self.menu.append(info)

    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(self._recent_detection_result, color_rgb=(0.3, 1.0, 0.1))