import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import time


def frameProcessing(im, predictor, Visualizer, cfg):
    tick0 = time.time()
    outputs = predictor(im)
    tick1 = time.time()
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv_bgr = out.get_image()[:, :, ::-1]
    tick2 = time.time()
    print(tick1 - tick0, tick2 - tick1)
    return cv_bgr

def Video(openpath, savepath = None):
    cap = cv2.VideoCapture(openpath)
    if cap.isOpened():
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    #fourcc = cv2.VideoWriter_fourcc('m','p','4','v') with *.mp4 save
    out = None
    if savepath is not None:
        out = cv2.VideoWriter(savepath, fourcc, fps, (width, height), True)
    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            output = frameProcessing(frame, predictor, Visualizer, cfg)
            # Write frame-by-frame
            if out is not None:
                out.write(output)
            # Display the resulting frame
            cv2.imshow("Input", frame)
            cv2.imshow("Output", output)
        else:
            break
        # waitKey(int(1000.0/fps)) for matching fps of video
        if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    return

def main(VideoPath):
    Video(VideoPath)    
    
if __name__=='__main__':
    videoPath = 'challenge.mp4'
    main(videoPath)
