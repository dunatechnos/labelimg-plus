# visualizer.py
import os
from annotation_helper.annotation_helper_current import AnnotationApp, set_paths


def open_helper(parent, images_dir, labels_dir, models_dir):
    set_paths(images_dir, labels_dir, models_dir)

    win = AnnotationApp()
    win.setWindowTitle("Annotation Helper")
    win.show()
    win.raise_()
    return win
