import cv2
import numpy as np

from plugin import Plugin
from pyglui import ui
from pupil_apriltags import Detector


class CLAHE_Preview(Plugin):
    order = 0.1

    def __init__(
        self,
        g_pool,
        clip_limit: float = 40.0,
        tile_grid_size_row: int = 8,
        tile_grid_size_col: int = 8,
    ):
        super().__init__(g_pool)
        self.clip_limit = clip_limit
        self.tile_grid_size_row = tile_grid_size_row
        self.tile_grid_size_col = tile_grid_size_col
        self.enabled = True
        self.at_detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def init_ui(self):
        self.add_menu()
        self.menu.label = "CLAHE Post-Processing Preview"
        self.menu.append(
            ui.Info_Text("Contrast Limited Adaptive Histogram Equalization")
        )
        self.menu.append(
            ui.Info_Text(
                "Read more about Histogram Equalization here: "
                "https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html"
            )
        )
        self.menu.append(ui.Switch("enabled", self, label="Enable CLAHE"))
        self.menu.append(ui.Text_Input("clip_limit", self, label="Clip Limit"))
        self.menu.append(
            ui.Text_Input("tile_grid_size_row", self, label="Tile Grid Size (rows)")
        )
        self.menu.append(
            ui.Text_Input("tile_grid_size_col", self, label="Tile Grid Size (columns)")
        )

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return

        gray = cv2.cvtColor(frame.img, cv2.COLOR_BGR2GRAY)

        if self.enabled:
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=(self.tile_grid_size_row, self.tile_grid_size_col),
            )

            gray = clahe.apply(gray)

        detections = self.at_detector.detect(gray)
        gray = np.repeat(gray[..., np.newaxis], 3, axis=2)

        for detection in detections:
            cv2.polylines(
                gray,
                [detection.corners.reshape((-1, 1, 2)).astype("int32")],
                True,
                (0, 0, 255),
            )

        frame.img[:] = gray

    def get_init_dict(self):
        return {
            "clip_limit": self.clip_limit,
            "tile_grid_size_row": self.tile_grid_size_row,
            "tile_grid_size_col": self.tile_grid_size_col,
        }