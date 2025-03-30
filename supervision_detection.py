import argparse
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv

SOURCE = np.array([[1643,801], [2326,801], [3493, 1918], [531, 1918]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0,0],
        [TARGET_WIDTH -1, 0],
        [TARGET_WIDTH -1, TARGET_HEIGHT - 1],
        [0, TARGET_WIDTH - 1]
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)
        
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1,2)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Vehicle Speed Detection'
    )
    
    parser.add_argument(
        '--source_video_path',
        required=False, 
        default='/Users/shaki/Documents/School/CS7367/Final Paper Stuff/Highway No Audio.mp4',
        help='path to the video file',
        type=str,
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    model = YOLO('/Users/shaki/Documents/GitHub/2025-YOLO-Vehicle-Tracking/Car Tracking/yolov8s/weights/best.pt')
    byte_track = sv.ByteTrack(frame_rate = video_info.fps)
    model.to('mps')
    
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)
    
    frame_generator = sv.get_video_frames_generator(args.source_video_path)
    
    polygon_zone = sv.PolygonZone(SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    
    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)
        
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)
        
        labels = [
            f"x: {x}, y:{y}"
            for [x,y]
            in points
        ]
        
        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels) 
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections)
        
        cv2.imshow("annotated_frame", annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
