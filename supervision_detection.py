import argparse
import numpy as np
import cv2
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv

SOURCE = np.array([[1643,801], [2326,801], [3493, 1918], [531, 1918]])

TARGET_WIDTH = 18.7275
TARGET_HEIGHT = 110.56

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1]
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
    model = YOLO('/Users/shaki/Documents/GitHub/2025-YOLO-Vehicle-Tracking/Car Tracking/yolov11m/weights/best.pt')
    byte_track = sv.ByteTrack(frame_rate = video_info.fps)
    model.to('mps')
    
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER)
    #trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=video_info.fps * 2, position=sv.Position.BOTTOM_CENTER)
    
    frame_generator = sv.get_video_frames_generator(args.source_video_path)
    
    polygon_zone = sv.PolygonZone(SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    
    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)
        
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)
        
        labels = []
        for tracker_id, [_,y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance/time * 2.23694
                labels.append(f'#{tracker_id} {int(speed)} mph')
        
        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE)
        #annotated_frame = trace_annotator.annotate(
            #scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels) 
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections)
        
        cv2.imshow("annotated_frame", annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
import argparse
import numpy as np
import cv2
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv

SOURCE = np.array([[1643, 801], [2326, 801], [3493, 1918], [531, 1918]])

TARGET_WIDTH = 18.7275
TARGET_HEIGHT = 110.56

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1]
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
        return transformed_points.reshape(-1, 2)


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
    
    parser.add_argument(
        '--speed_threshold',
        required=False,
        default=45.0,
        help='Minimum speed (in mph) for detection to be annotated',
        type=float,
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    model = YOLO('/Users/shaki/Documents/GitHub/2025-YOLO-Vehicle-Tracking/Car Tracking/yolov11s/weights/best.pt')
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)
    model.to('mps')
    
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale, 
        text_thickness=thickness, 
        text_position=sv.Position.BOTTOM_CENTER, 
        color_lookup=sv.ColorLookup.TRACK
    )
    
    frame_generator = sv.get_video_frames_generator(args.source_video_path)
    
    polygon_zone = sv.PolygonZone(SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    
    # Dictionary to store a deque of y-coordinates per tracker_id
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    
    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)
        
        # Get the bottom center anchor coordinates and transform them
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)
        
        indices_to_keep = []
        final_labels = []
        
        # Process each detection
        for idx, (tracker_id, point) in enumerate(zip(detections.tracker_id, points)):
            y = point[1]
            coordinates[tracker_id].append(y)
            
            # Require a minimum number of frames to compute speed reliably
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                continue
            
            # Compute speed based on change in y-coordinate over time
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_start - coordinate_end)
            time_elapsed = len(coordinates[tracker_id]) / video_info.fps
            speed = distance / time_elapsed * 2.23694  # conversion to mph
            
            # Only keep detections above the speed threshold
            if speed >= args.speed_threshold:
                indices_to_keep.append(idx)
                final_labels.append(f'#{tracker_id} {int(speed)} mph')
        
        # Annotate only if any detections pass the speed threshold
        if len(indices_to_keep) > 0:
            indices_to_keep = np.array(indices_to_keep)
            detections = detections[indices_to_keep]
            annotated_frame = frame.copy()
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=final_labels
            )
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            cv2.imshow("annotated_frame", annotated_frame)
        else:
            cv2.imshow("annotated_frame", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
