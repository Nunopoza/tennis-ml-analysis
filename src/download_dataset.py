
from roboflow import Roboflow
rf = Roboflow(api_key="ZGLuDrCQpJdtcfEuYUFL")
project = rf.workspace("tennisgo").project("tennis-objects-rw5zf")
version = project.version(1)
dataset = version.download("yolov8")
                
