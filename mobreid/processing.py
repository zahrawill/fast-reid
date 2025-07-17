import cv2 as cv
from ultralytics import YOLO
import numpy as np
import torch
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
import time

#TO RUN: python3 -m mobreid.processing
thres = -625
class FastReIDExtractor:
    """Helper for extracting features with FastReID."""

    def __init__(self, cfg_path: str, weight_path: str):
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.merge_from_list(["MODEL.WEIGHTS", weight_path])
        cfg.freeze()
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)

    def __call__(self, image_bgr: np.ndarray):
        # FastReID expects RGB images
        image = image_bgr[:, :, ::-1]
        image = cv.resize(image, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv.INTER_CUBIC)
        tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
        feat = self.predictor(tensor)
        return feat.squeeze(0).cpu() #.numpy() to return array

#to configure reid: "anaconda3\Lib\site-packages\ultralytics\cfg\trackers\botsort.yaml"

# Represents each object to be tracked and projected
class TrackedObject:
    def __init__(self, center_pt, id, timestamp, keypoint, feature=None):
        self.center_pt = center_pt
        # Store both position and timestamp as tuples
        self.timestp_ctr_pts = [(timestamp, center_pt)]
        self.id = id
        self.timestp_keypts = [(timestamp, keypoint)]
        # Keep a history of extracted ReID features
        self.features = []
        if feature is not None:
            self.features.append((timestamp, feature))

    # Updating list of center point locations with timestamp
    def update_pts(self, new_pt, timestamp, new_keypoint, feature=None):
        self.timestp_ctr_pts.append((timestamp, new_pt))
        self.timestp_keypts.append((timestamp, new_keypoint))
        if feature is not None:
            self.features.append((timestamp, feature))

def remap_id(track_id, conf) -> str | int:
    return f"ID*: {track_id} | conf: {conf:.2f}"

#Gets center of pose estimation block (not provided by yolo's keypoint object)
def get_center(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2, (y1 + y2) / 2]

#Matches pose block center to center points found by tracking model to match IDs
def find_closest_keypoints(center, keypoints_data):
    min_dist = float('inf')
    best_keypoints = None
    for kp in keypoints_data:
        kps = kp.xy[0].cpu().numpy()
        kp_center = kps.mean(axis=0)
        dist = np.linalg.norm(np.array(center) - kp_center)
        if dist < min_dist:
            min_dist = dist
            best_keypoints = kps.tolist()
    return best_keypoints



#Person tracking and pose estimation
def transform_real_time(video_path, cfg_path, weight_path):
    # Load tracking, pose estimation, and ReID models
    detect_model = YOLO("yolo11n.pt")
    pose_model = YOLO("yolo11n-pose.pt")
    reid_extractor = FastReIDExtractor(cfg_path, weight_path)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_number = 0
    tracked_objects = {}
    crops = []
    features = []
    #cv.namedWindow("Detection + Pose Frame", cv.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        timestamp = "%.2f" % (frame_number / fps)
        frame_number += 1

        # Person detection + tracking
        #imgsz=(1088,1920)
        results = detect_model.track(source=frame, persist = True, tracker="botsort.yaml", classes=[0])
        # Pose estimation
        pose_results = pose_model(frame)

        #Combine visualization: draw pose results on top of detection frame
        #annotated_frame = pose_results[0].plot()  # base pose visualization  
        #annotated_frame = results[0].plot() # base tracking visualization

        # Process each tracked detection before drawing results so we can
        # override IDs if necessary
        if results[0].boxes is not None:
            for detection in results[0].boxes:
                if detection.id is None:
                    continue

                obj_id = int(detection.id.item())
                cls = int(detection.cls.item())
                if cls != 0: #Only tracking people, cls 0 means person
                    continue

                box = detection.xyxy.cpu().numpy()[0]
                center = get_center(box)

                best_kps = find_closest_keypoints(center, pose_results[0].keypoints) #could be helpful for
                if best_kps is None:
                    continue

                if obj_id in tracked_objects: 
                    '''Update with timestamp, coords, and features'''
                    # Crop bounding box image for feature extraction
                    x1, y1, x2, y2 = [int(i) for i in box]
                    h, w = frame.shape[:2]
                    x1 = max(0, x1);  y1 = max(0, y1)
                    x2 = min(w, x2);  y2 = min(h, y2)

                    if x2 > x1 and y2 > y1:
                        crop_img = frame[y1:y2, x1:x2].copy()
                        crops.append(crop_img)
                        feature = reid_extractor(crop_img) #returns numpy array
                    tracked_objects[obj_id].update_pts(center, timestamp, best_kps, feature)
                else:
                    # Crop bounding box image for feature extraction
                    x1, y1, x2, y2 = [int(i) for i in box]
                    h, w = frame.shape[:2]
                    x1 = max(0, x1);  y1 = max(0, y1)
                    x2 = min(w, x2);  y2 = min(h, y2)

                    #feature = []
                    if x2 > x1 and y2 > y1:
                        crop_img = frame[y1:y2, x1:x2].copy()
                        crops.append(crop_img)
                        feature = reid_extractor(crop_img) #returns numpy array
                        features.append(feature)

                    # Compare this feature to existing tracked objects using cosine similarity
                    matched_obj = None
                    min_distance = float('inf')
                    if feature is not None:
                        for t_obj in tracked_objects.values():
                            if not t_obj.features:
                                continue
                            prev_feat = torch.tensor(t_obj.features[-1][1]) #takes last extracted gallery feature & converts to tensor
                            t_feat = torch.tensor(feature)
                            dist = 1 - (torch.nn.functional.cosine_similarity(t_feat, prev_feat, dim=0).item())#cosine distance
                            if dist < min_distance:
                                min_distance = dist
                                matched_obj = t_obj

                    if matched_obj is not None and min_distance < thres: 
                        # Update matched_obj instead of creating new instance
                        matched_obj.update_pts(center, timestamp, best_kps, feature)
                        matched_id = matched_obj.id
                        
                    else:
                        # New tracked_object, default bounding box display
                        tracked_objects[obj_id] = TrackedObject(center, obj_id, timestamp, best_kps, feature)
                        matched_id = obj_id

        
        # Update YOLO detection id so the bounding box displays the matched object's ID
        r = results[0]
        if r.boxes is not None: 
            for xyxy, conf in zip(r.boxes.xyxy, r.boxes.conf):
                frame_vis = r.plot(labels=False)   # suppress ID labels
                x1, y1 = (int(xyxy[0])), (int(xyxy[1]))
                disp = remap_id((int(matched_id)), conf)
                cv.putText(frame_vis, str(disp),
                    (int(x1), int(y1) - 5),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv.LINE_AA)
            cv.imshow("Detection Frame", frame_vis)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    return tracked_objects

'''RUN IN TERMINAL WITH python3 -m mobreid.processing '''
if __name__ == '__main__':
    #vpath = r"mobreid/4min12fps_300clemantis.mp4"
    #rtsp1 = r"rtsp://admin:Zarpoolo01!@192.168.50.79:554/"
    #rtsp2 = r"rtsp://admin:Zarpoolo01!@192.168.50.163:554/"
    rtmp1 = r"rtmp://192.168.50.79/bcs/channel0_sub.bcs?channel=0&stream=0&user=admin&password=Zarpoolo01!"
    rtmp2 = r"rtmp://192.168.50.163/bcs/channel0_sub.bcs?channel=0&stream=0&user=admin&password=Zarpoolo0!"
    cfg_file = "logs/market1501/mgn_R50-ibn/config.yaml"  # path to model config
    weight_file = "logs/market1501/mgn_R50-ibn/model_final.pth"  # path to pretrained model
    tracked_objs = transform_real_time(rtmp1, cfg_file, weight_file)
