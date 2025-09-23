import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# Add GETI support imports
from model_api.models import Model
from model_api.models.result.detection import DetectionResult
from supervision import (
    BoxCornerAnnotator,
    ByteTrack,
    Color,
    ColorLookup,
    Detections,
    DetectionsSmoother,
    LabelAnnotator,
    LineZone,
    LineZoneAnnotator,
    Point,
    TraceAnnotator,
)
from supervision.annotators.base import BaseAnnotator
from ultralytics import YOLO, YOLOE, YOLOWorld
from ultralytics.engine.results import Results

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

MODEL_DIR = Path("model")
DATA_DIR = Path("data")

# the following are "null" classes to improve detection of the main classes
NULL_CLASSES = ["person", "hand", "finger", "fabric"]

PROBABILITY_COEFFICIENT = 15


def load_yolo_model(model_name: str, main_class: str, aux_classes: list[str]) -> YOLO:
    # set classes to detect
    classes = [main_class] + aux_classes + NULL_CLASSES

    model_path = MODEL_DIR / f"{model_name}.pt"

    if "world" in model_name:
        model = YOLOWorld(model_path)
        model.set_classes(classes)
    else:
        model = YOLOE(model_path)
        model.set_classes(classes, model.get_text_pe(classes))

    model_path = model.export(format="openvino", dynamic=False, half=True)

    return YOLO(model_path)


def load_geti_model(model_path: str) -> Model:
    """
    Load GETI model via Model API.

    Args:
        model_path: Path to the GETI model XML file

    Returns:
        Model API model object
    """
    model = Model.create_model(model_path)
    return model


def model_api_to_supervision(model_api_result: DetectionResult) -> Detections:
    """
    Convert Model API result directly to supervision Detections format.

    Args:
        model_api_result: Model API DetectionResult object

    Returns:
        supervision.Detections: Converted detections
    """
    # Extract detection information from Model API result
    if (
        hasattr(model_api_result, "bboxes")
        and hasattr(model_api_result, "scores")
        and hasattr(model_api_result, "labels")
    ):
        bboxes = model_api_result.bboxes
        scores = model_api_result.scores
        labels = model_api_result.labels

        # Convert to numpy arrays and ensure correct format
        if len(bboxes) > 0:
            xyxy = np.array(bboxes, dtype=np.float32)
            confidence = np.array(scores, dtype=np.float32)
            class_id = np.array(labels, dtype=int)
        else:
            # Empty detection
            xyxy = np.empty((0, 4), dtype=np.float32)
            confidence = np.empty(0, dtype=np.float32)
            class_id = np.empty(0, dtype=int)

        # Create supervision Detections object
        detections = Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        # Add class names if available
        if hasattr(model_api_result, "label_names") and model_api_result.label_names:
            detections.data["class_name"] = model_api_result.label_names

        return detections
    else:
        # No detections or wrong format
        return Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            class_id=np.empty(0, dtype=int),
        )


def load_annotators(
    size: tuple[int, int],
) -> tuple[list[BaseAnnotator], LineZone, ByteTrack, DetectionsSmoother]:
    line_x = int(0.3 * size[0])
    start = Point(line_x, 0)
    end = Point(line_x, size[1])

    line_zone = LineZone(start=start, end=end)
    box_annotator = BoxCornerAnnotator(
        corner_length=int(0.03 * size[0]), color_lookup=ColorLookup.TRACK
    )
    label_annotator = LabelAnnotator(text_scale=0.7, color_lookup=ColorLookup.TRACK)
    trace_annotator = TraceAnnotator(
        thickness=box_annotator.thickness, color_lookup=ColorLookup.TRACK
    )
    line_zone_annotator = LineZoneAnnotator(
        thickness=box_annotator.thickness,
        color=Color.RED,
        text_scale=label_annotator.text_scale,
        display_in_count=False,
        custom_out_text="Count",
    )

    tracker = ByteTrack()
    smoother = DetectionsSmoother(length=3)

    return (
        [box_annotator, label_annotator, trace_annotator, line_zone_annotator],
        line_zone,
        tracker,
        smoother,
    )


def add_box_margin(
    box: tuple[int], frame_size: tuple[int], margin_ratio: float = 0.15
) -> tuple[int]:
    frame_w, frame_h = frame_size
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    border_x = width * margin_ratio
    border_y = height * margin_ratio

    new_x1 = max(0, x1 - border_x)
    new_y1 = max(0, y1 - border_y)
    new_x2 = min(frame_w, x2 + border_x)
    new_y2 = min(frame_h, y2 + border_y)

    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)


def filter_and_process_results(
    main_class: str,
    aux_classes: list[str],
    detections: Detections,
    tracker: ByteTrack,
    smoother: DetectionsSmoother,
    line_zone: LineZone,
) -> Detections:
    """
    Process supervision Detections with filtering, tracking, and smoothing.

    Args:
        main_class: The main class name to detect
        aux_classes: Auxiliary classes supporting the detection
        detections: supervision Detections object (from either ultralytics or GETI)
        tracker: ByteTrack tracker
        smoother: DetectionsSmoother
        line_zone: LineZone for counting

    Returns:
        Processed supervision Detections
    """
    # Apply probability coefficient for tracking (especially important for YOLOWorld)
    detections.confidence *= PROBABILITY_COEFFICIENT

    # Filter out the null classes (only for YOLO models that have them)
    if len(aux_classes) > 0:  # YOLO models have aux_classes
        detections = detections[
            np.isin(detections.class_id, np.arange(len(aux_classes) + 1))
        ]

    # Set the class_id to 0 (MAIN_CLASS) for all detections and update class names
    detections.data["class_name"] = [main_class] * len(detections)

    detections = detections.with_nmm(class_agnostic=True)
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    line_zone.trigger(detections)

    return detections


def get_patches(frame: np.ndarray, results: Detections) -> np.ndarray:
    patches = []

    for box, _, _, _, _, _ in results:
        # add border to the bounding box to fit the training data
        x1, y1, x2, y2 = add_box_margin(box, frame.shape[:2][::-1])
        patch = frame[y1:y2, x1:x2]
        patch = cv2.resize(patch, (256, 256))
        patches.append(patch)

    return np.array(patches)


def draw_results(
    frame: np.ndarray,
    annotators: list[BaseAnnotator],
    line_zone: LineZone,
    detections: Detections,
) -> None:
    for annotator in annotators:
        if isinstance(annotator, LineZoneAnnotator):
            annotator.annotate(frame, line_counter=line_zone)
        else:
            annotator.annotate(frame, detections)


def run(
    video_path: str,
    det_model_name: str,
    device: str,
    main_class: str,
    aux_classes: list[str],
    flip: bool,
    backend: str = "ultralytics",
):
    # Load model based on backend
    if backend.lower() == "geti":
        # For GETI backend, det_model_name is treated as a path to the model
        det_model = load_geti_model(det_model_name)
    else:
        # For ultralytics backend, det_model_name is a model name from choices
        det_model = load_yolo_model(det_model_name, main_class, aux_classes)

    qr_code = utils.get_qr_code(
        "https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/demos/spot_the_object_demo",
        with_embedded_image=True,
    )

    video_size = (1920, 1080)
    # initialize video player to deliver frames
    if isinstance(video_path, str) and video_path.isnumeric():
        video_path = int(video_path)
    player = utils.VideoPlayer(video_path, size=video_size, fps=60, flip=flip)
    annotators, line_zone, tracker, smoother = load_annotators(video_size)

    processing_times = deque(maxlen=100)

    title = "Press ESC to Exit"
    cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)

    # cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(title, 1920, 1080)

    # start a video stream
    player.start()
    while True:
        # Grab the frame.
        frame = player.next()
        if frame is None:
            print("Source ended")
            break

        f_height, f_width = frame.shape[:2]

        start_time = time.time()

        # Handle prediction based on backend
        if backend.lower() == "geti":
            # GETI Model API prediction - convert directly to supervision
            model_api_result = det_model(frame)
            det_results = model_api_to_supervision(model_api_result)
        else:
            # YOLO prediction - convert to supervision
            ultralytics_results = det_model.predict(
                frame, conf=0.01, verbose=False, device=f"intel:{device.lower()}"
            )[0]
            det_results = Detections.from_ultralytics(ultralytics_results)

        # Apply unified processing to supervision detections
        det_results = filter_and_process_results(
            main_class, aux_classes, det_results, tracker, smoother, line_zone
        )

        end_time = time.time()

        draw_results(frame, annotators, line_zone, det_results)

        processing_times.append(end_time - start_time)
        # mean processing time [ms]
        processing_time = np.mean(processing_times) * 1000

        fps = 1000 / processing_time
        utils.draw_text(
            frame,
            text=f"Inference time: {processing_time:.0f}ms ({fps:.1f} FPS)",
            point=(10, 10),
        )

        utils.draw_ov_watermark(frame)
        utils.draw_qr_code(frame, qr_code)

        # show the output live
        cv2.imshow(title, frame)
        key = cv2.waitKey(1)
        # escape = 27 or 'q' to close the app
        if key == 27 or key == ord("q"):
            break

    player.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream",
        default="./geti_sdk-deployment/sample_video.mp4",
        type=str,
        help="Path to a video file or the webcam number",
    )
    # parser.add_argument('--stream', default="0", type=str, help="Path to a video file or the webcam number")
    parser.add_argument(
        "--class_name", default="hazelnut", type=str, help="The class name to detect"
    )
    parser.add_argument(
        "--aux_classes",
        nargs="+",
        default=["nut", "brown ball"],
        type=str,
        help="Auxiliary classes supporting the detection of the main class",
    )
    parser.add_argument(
        "--device", default="AUTO", type=str, help="Device to run inference on"
    )
    parser.add_argument(
        "--detection_model",
        type=str,
        default="yolov8m-worldv2",
        help="Model for object detection. For 'ultralytics' backend: choose from available YOLO models (yolov8s-world, yolov8m-worldv2, etc.). For 'geti' backend: provide path to model XML file.",
    )
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")
    parser.add_argument(
        "--backend",
        type=str,
        default="ultralytics",
        choices=["ultralytics", "geti"],
        help="Backend to use: 'ultralytics' for YOLO models or 'geti' for GETI models via Model API",
    )

    args = parser.parse_args()
    run(
        args.stream,
        args.detection_model,
        args.device,
        args.class_name,
        args.aux_classes,
        args.flip,
        args.backend,
    )
