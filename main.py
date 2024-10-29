import argparse
import cv2
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import math
import pandas as pd

from sklearn.preprocessing import StandardScaler


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  


from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams

from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics import YOLO
from trackers.multi_tracker_zoo import create_tracker
from yolov8.ultralytics.yolo.utils.ops import Profile

@torch.no_grad()
def run(
        source='0',
        save_vid=False,
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        project=ROOT / 'runs' / 'results',
        imgsz=(640,640) # inference size (height, width)
        
):
    speed_check1=[]
    speed_check2=[]
    device='cpu'
    model1 = YOLO("yolov8n.pt")
    model2 = YOLO("best.pt")
    source = str(source)
    exp_name='result'
    save_dir = increment_path(Path(project) / exp_name, exist_ok=False)
    if save_vid:
        (save_dir / 'tracks').mkdir(parents=True, exist_ok=True)
    imgsz = check_imgsz(imgsz, stride=32)  # check image size

    # Dataloader
    bs = 1
    if source=='0':
        
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=32,
            auto=True,
            transforms=getattr(model1.model, 'transforms', None),
            vid_stride=1
        )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=32,
            auto=True,
            transforms=getattr(model1.model, 'transforms', None),
            vid_stride=1
        )
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    
    tracker_list = []
    
    tracking_config='trackers\\strongsort\\configs\\strongsort.yaml'
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, False)
        tracker_list.append(tracker, )
    

    outputs1 = [None] * bs
    outputs2 = [None] * bs
   
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        
        if source=='0':  
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)
            save_path = str(save_dir / p.name)
        else:
            p, im0 = path, im0s.copy()
            p = Path(p)
            save_path = str(save_dir / p.name)
            
        
        results1 = model1.predict(source=im0,stream=True,verbose=False,conf=0.5)
        results2 = model2.predict(source=im0,stream=True,verbose=False,conf=0.8)
        
        for r in results1:
            boxes=r.boxes
            det1=boxes.data
            
        
        for r in results2:
            boxes=r.boxes
            det2=boxes.data
        print(det1,"\t",det2)
        curr_frames[i] = im0
  
        if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
            if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

       
        outputs1[i] = tracker_list[i].update(det1.cpu(), im0)
        outputs2[i] = tracker_list[i].update(det2.cpu(), im0)
        bbox_final1=[]
        bbox_final2=[]
        
        for j, (output) in enumerate(outputs1[i]):
            bbox=output[:7]
            if bbox[5]==2 or 3 or 5 or 7:
                bbox_final1.append(bbox)
        
        for t in bbox_final1:
            
            x1,y1,x2,y2=t[0],t[1],t[2],t[3]
            cv2.rectangle(im0,(x1,y1),(x2,y2), (255, 0, 0), 3)
            
        for j, (output) in enumerate(outputs2[i]):
            bbox=output[:7]
            
            bbox_final2.append(bbox)
        
        for t in bbox_final2:
            
            x1,y1,x2,y2=t[0],t[1],t[2],t[3]
            cv2.rectangle(im0,(x1,y1),(x2,y2), (255, 0, 0), 3)
       
        speed_check1.append(bbox_final1)
        speed_check2.append(bbox_final2)
       
        if cv2.waitKey(1) == ord('q'):  
            exit()

        speeds1={}
        speeds2={}
        prev_frames[i] = curr_frames[i]
        
        
        speeds1=speed_collect(speed_check1,speeds1)
        speeds2=speed_collect(speed_check2,speeds2)
        
        #df=pd.DataFrame(columns=['xmin','ymin','xmax','ymax','scaled_xmin','scaled_ymin','scaled_xmax','scaled_ymax'])
        for t in bbox_final1:
            distance=calc_dist(t,im0.shape[1],im0.shape[0])
            x1,y1,x2,y2,track=t[0],t[1],t[2],t[3],t[4]

            if speeds1 is not None and track in speeds1:
                text="speed="+str(int(speeds1[track]))+"kmph"+"  distance="+str(int(distance))
                cv2.putText(im0, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
                
        for t in bbox_final2:
            distance=calc_dist(t,im0.shape[1],im0.shape[0])
            
            x1,y1,x2,y2,track=t[0],t[1],t[2],t[3],t[4]
            

            if speeds2 is not None and track in speeds2:
                text="speed="+str(int(speeds2[track]))+"kmph"+"  distance="+str(int(distance))
                cv2.putText(im0, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)       
        
        if len(speed_check1)==2:
            del speed_check1[0]
        if len(speed_check2)==2:
            del speed_check2[0]
        
        if save_vid is False:
            cv2.imshow("Frame", im0)
        
        
        if save_vid:
            
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0)
        
    if save_vid is True:
        print("\nResults saved in ",save_path)



def calc_dist(t,wd,ht):
    x3,y3=t[0],t[3]-(t[3]-t[1])/7
    x1,y1=wd/2,0.9*ht
    x2,y2=t[2],t[3]-(t[3]-t[1])/7
    angle_x1_x2 = math.degrees(math.atan2(x1 - x2, y1 - y2))
    angle_x1_x3 = math.degrees(math.atan2(x1 - x3, y1 - y3))
    angle_right = 90 + angle_x1_x2
    angle_left = 90 - angle_x1_x3
    total_angle = angle_right + angle_left
    
    length=5000
    distance = (length *  total_angle * 0.01745329) / 1000
    return distance
    

def speed_collect(speed_check,speeds):
    try:
        x=speed_check[0]
        y=speed_check[1]
    
        for k in x:
            for j in y:
                if k[4]==j[4]:
                    speed=calc_speed(k,j)
                    speeds[k[4]]=speed
        return speeds
    except:
        return None

def calc_speed(location1, location2):
	d_pixels = math.sqrt(math.pow(location2[2] - location1[2], 2) + math.pow(location2[3] - location1[3], 2))
	ppm = (location2[2]-location2[0])/3
	
	d_meters = d_pixels / ppm
	
	fps = 30
	speed = d_meters * fps * 3.6
	
	return speed
    
    
def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    opt = parser.parse_args()
    
    
    print_args(vars(opt))
    return opt

def main(opt):
    
    run(**vars(opt))
    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)