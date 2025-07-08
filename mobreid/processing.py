import cv2 as cv
from ultralytics import YOLO
import numpy as np

#to configure reid: "anaconda3\Lib\site-packages\ultralytics\cfg\trackers\botsort.yaml"

# Represents each object to be tracked and projected
class TrackedObject:
    def __init__(self, center_pt, id, timestamp, keypoint):
        self.center_pt = center_pt
        # Store both position and timestamp as tuples
        self.timestp_ctr_pts = [(timestamp, center_pt)]
        self.id = id
        self.timestp_keypts = [(timestamp, keypoint)]

    # Updating list of center point locations with timestamp
    def update_pts(self, new_pt, timestamp, new_keypoint):
        self.timestp_ctr_pts.append((timestamp, new_pt))
        self.timestp_keypts.append((timestamp, new_keypoint))

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
def transform_real_time(video_path):
    #Load tracking, pose estimation, and ReID models
    detect_model = YOLO("yolo11n.pt")
    pose_model = YOLO("yolo11n-pose.pt")

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_number = 0
    tracked_objects = {}
    crops = []
    cv.namedWindow("Detection + Pose Frame", cv.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        timestamp = "%.2f" % (frame_number / fps)
        frame_number += 1

        # Person detection + tracking
        #imgsz=(1088,1920)
        results = detect_model.track(source=frame, persist = True, tracker="botsort.yaml", verbose=True)

        # Pose estimation
        pose_results = pose_model(frame)

        #Combine visualization: draw pose results on top of detection frame
        annotated_frame = pose_results[0].plot()  # Pose visualized here
        #annotated_frame = results[0].plot()

        #Process each tracked detection
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

                best_kps = find_closest_keypoints(center, pose_results[0].keypoints)
                if best_kps is None:
                    continue

                if obj_id in tracked_objects:
                    tracked_objects[obj_id].update_pts(center, timestamp, best_kps)
                else:
                    '''CROP BOUNDING BOX IMAGE'''
                    x1, y1, x2, y2 = [int(i) for i in box]
                    # make sure the numbers are inside the image
                    h, w = frame.shape[:2]
                    x1 = max(0, x1);  y1 = max(0, y1)
                    x2 = min(w, x2);  y2 = min(h, y2)

                    # skip impossible boxes
                    if x2 > x1 and y2 > y1:
                        crops.append(frame[y1:y2, x1:x2].copy())      # isolate the patch
                    tracked_objects[obj_id] = TrackedObject(center, obj_id, timestamp, best_kps)

        # Show combined pose + detection overlay
        cv.imshow("Detection + Pose Frame", annotated_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    return crops

if __name__ == '__main__':
    vpath = r"4min12fps_300clemantis.mp4"
    rtsp = r"rtsp://admin:Zarpoolo01!@192.168.50.94:554/"
    rtmp = r"rtmp://192.168.50.94/bcs/channel0_sub.bcs?channel=0&stream=0&user=admin&password=Zarpoolo01!"
    crops = []
    crops = transform_real_time(rtmp)
    cv.imshow("img crop", crops[0])
    cv.waitKey(0)
    cv.destroyAllWindows()