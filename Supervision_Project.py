import numpy as np
import supervision as sv
from ultralytics import YOLO
import matplotlib.pyplot as plt

import os
HOME = os.getcwd()
print(HOME)

# image = cv2.imread(HOME+"/parking2.jpg")
model = YOLO(HOME+'/best.pt')
url= HOME+"/Mobildijalan.mp4"
# detections = Detections(...)
# result = model(image)[0]




# sv.plot_image(annotated_frame)

# cap=cv2.VideoCapture(HOME+url)
# # cap=cv2.VideoCapture(0)

# while True:
#     success, frame=cap.read()
    
#     if success:
#         print("video terbaca")
#         results=model(frame,conf=0.7,iou=0.2,imgsz=640)
#         detections = sv.Detections.from_ultralytics(results)
#         annotated_frame = annotator.annotate(scene=results[0].plot,detections=detections)
#         # annotated_frame = results[0].plot()
#         resize_annotated=cv2.resize(annotated_frame,(640,480))
#         cv2.imshow("YOLOv8 Inference", resize_annotated)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         print("Video Tidak ada")
#         break

# cap.release()
# cv2.destroyAllWindows()

# cv2.imshow('Gambar',annotated_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# extract video frame
generator = sv.get_video_frames_generator(url)
iterator = iter(generator)
frame = next(iterator)
tracker=sv.ByteTrack()
label_annotator = sv.LabelAnnotator()
START = sv.Point(400, 0)
END = sv.Point(400, 720)
line_zone = sv.LineZone(start=START, end=END)

line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_thickness=4,
    text_scale=2)


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    # detect
    results = model(frame, imgsz=600,verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)


    # labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
    labels = [
        f"#{tracker_id} {model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]


    # annotate
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)
    line_zone.trigger(detections)
    print("in count : ",line_zone.in_count,"\n")
    print("out count : ",line_zone.out_count,"\n")
    return  line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

sv.process_video(
    source_path=url,
    target_path=HOME+"result.mp4",
    callback=callback
    )

# sv.show_frame_in_notebook(frame, (16, 16))