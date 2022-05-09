import argparse
import collections
import datetime
import json
import time
from pathlib import Path
from typing import Tuple, Union, Any

import cv2
import ml_collections
import numpy as np
import pandas as pd
import pytz
import six
import tensorflow as tf
from PIL import Image
from streamlink import Streamlink

from pix2seq.data import data_utils
from pix2seq.models import ar_model as model_lib
from pix2seq.tasks.object_detection import TaskObjectDetection
from pix2seq.tasks.visualization.vis_utils import STANDARD_COLORS
from pix2seq.tasks.visualization.vis_utils import _get_multiplier_for_color_randomness
from pix2seq.tasks.visualization.vis_utils import draw_bounding_box_on_image_array
from pix2seq.tasks.visualization.vis_utils import draw_keypoints_on_image_array
from pix2seq.tasks.visualization.vis_utils import draw_mask_on_image_array

# Default model
model_dir = Path(__file__).parents[1] / "coco_det_finetune" / "resnet_1024x1024"  # Restore checkpoint from local

# Default CSV logbook
logbook_path = Path(__file__).parents[1] / "logbook.csv"

# Max number of detections in the image
# WARNING: Increasing this number may lead to tf.InvalidArgumentError: Graph execution error
num_instances_to_generate = 75


def stream_to_url(url: str, quality: str = 'best') -> str:
    """
    Get URL, and return streamlink URL

    References
    ----------
    - https://stackoverflow.com/a/69501647
    """
    session = Streamlink()
    streams = session.streams(url)

    if streams:
        return streams[quality].to_url()
    else:
        raise ValueError('Could not locate your stream.')


def load_model(model_dir: Union[str, Path], from_gcloud: bool = False):
    """
    Loads checkpoint from local or GCloud
    """
    model_dir = Path(model_dir)

    if from_gcloud:
        # Downloads checkpoint from GCloud
        with tf.io.gfile.GFile(model_dir / "config.json", "r") as f:
            config = ml_collections.ConfigDict(json.loads(f.read()))
    else:
        # Local (much faster)
        with open(model_dir / "config.json", "r") as f:
            config = ml_collections.ConfigDict(json.loads(f.read()))

    # Set batch size to 1.
    config.eval.batch_size = 1

    # Remove the annotation filepaths.
    config.dataset.coco_annotations_dir = None

    # Update config fields.
    config.task.vocab_id = 10  # object_detection task vocab id.
    config.training = False
    config.dataset.val_filename = "instances_val2017.json"

    assert config.task.name == "object_detection"
    task = TaskObjectDetection(config)

    # Restore checkpoint
    model = model_lib.Model(config)
    checkpoint = tf.train.Checkpoint(model=model, global_step=tf.Variable(0, dtype=tf.int64))
    ckpt = tf.train.latest_checkpoint(model_dir)
    checkpoint.restore(ckpt).expect_partial()
    global_step = checkpoint.global_step

    return task, model, config


def load_category_names() -> dict:
    """
    Category names for COCO

    Returns
    -------
    dict
    """
    categories_str = '{"categories": [{"supercategory": "person","id": 1,"name": "person"}]}'
    categories_dict = json.loads(categories_str)
    categories_dict = {c["id"]: c for c in categories_dict["categories"]}

    return categories_dict


@tf.function
def infer(model, preprocessed_outputs: tuple):
    return task.infer(model, preprocessed_outputs)


def _predict_image(img: np.ndarray, model, task, config) -> Tuple[tf.Tensor, ...]:
    features = {
        "image": tf.image.convert_image_dtype(img, tf.float32),
        "image/id": 0,  # dummy image id.
        "orig_image_size": tf.shape(img)[:2],
    }
    labels = {
        "label": tf.zeros([1], tf.int32),
        "bbox": tf.zeros([1, 4]),
        "area": tf.zeros([1]),
        "is_crowd": tf.zeros([1]),
    }

    features, labels = data_utils.preprocess_eval(
        features,
        labels,
        max_image_size=config.model.image_size,
        max_instances_per_image=1,
    )

    # Batch features and labels.
    features = {k: tf.expand_dims(v, 0) for k, v in features.items()}
    labels = {k: tf.expand_dims(v, 0) for k, v in labels.items()}

    # Inference.
    preprocessed_outputs = (features['image'], None, (features, labels))
    infer_outputs = infer(model, preprocessed_outputs)
    _, pred_seq, _ = infer_outputs
    results = task.postprocess_tpu(*infer_outputs)

    return results


def visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        instance_boundaries=None,
        keypoints=None,
        keypoint_edges=None,
        track_ids=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        agnostic_mode=False,
        line_thickness=4,
        groundtruth_box_visualization_color='black',
        skip_boxes=False,
        skip_scores=False,
        skip_labels=False,
        skip_track_ids=False):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then this
        function assumes that the boxes to be plotted are groundtruth boxes and
        plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      instance_masks: a numpy array of shape [N, image_height, image_width] with
        values ranging between 0 and 1, can be None.
      instance_boundaries: a numpy array of shape [N, image_height, image_width]
        with values ranging between 0 and 1, can be None.
      keypoints: a numpy array of shape [N, num_keypoints, 2], can be None
      keypoint_edges: A list of tuples with keypoint indices that specify which
        keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
        edges from keypoint 0 to 1 and from keypoint 2 to 4.
      track_ids: a numpy array of shape [N] with unique track ids. If provided,
        color-coding of boxes will be determined by these ids, and not the class
        indices.
      use_normalized_coordinates: whether boxes is to be interpreted as normalized
        coordinates or not.
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw all
        boxes.
      min_score_thresh: minimum score threshold for a box to be visualized
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
      line_thickness: integer (default: 4) controlling line width of the boxes.
      groundtruth_box_visualization_color: box color for visualizing groundtruth
        boxes
      skip_boxes: whether to skip the drawing of bounding boxes.
      skip_scores: whether to skip score when drawing a single detection
      skip_labels: whether to skip label when drawing a single detection
      skip_track_ids: whether to skip track id when drawing a single detection

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    box_to_track_ids_map = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if track_ids is not None:
                box_to_track_ids_map[box] = track_ids[i]
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in six.viewkeys(category_index):
                            class_name = category_index[classes[i]]['name']
                        else:
                            # Skip N/A
                            # class_name = 'N/A'
                            continue
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = '{}%'.format(int(100 * scores[i]))
                    else:
                        display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
                if not skip_track_ids and track_ids is not None:
                    if not display_str:
                        display_str = 'ID {}'.format(track_ids[i])
                    else:
                        display_str = '{}: ID {}'.format(display_str, track_ids[i])
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                elif track_ids is not None:
                    prime_multipler = _get_multiplier_for_color_randomness()
                    box_to_color_map[box] = STANDARD_COLORS[(prime_multipler *
                                                             track_ids[i]) %
                                                            len(STANDARD_COLORS)]
                else:
                    box_to_color_map[box] = STANDARD_COLORS[classes[i] %
                                                            len(STANDARD_COLORS)]

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        if instance_masks is not None:
            draw_mask_on_image_array(
                image, box_to_instance_masks_map[box], color=color)
        if instance_boundaries is not None:
            draw_mask_on_image_array(
                image, box_to_instance_boundaries_map[box], color='red', alpha=1.0)
        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=0 if skip_boxes else line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)
        if keypoints is not None:
            draw_keypoints_on_image_array(
                image,
                box_to_keypoints_map[box],
                color=color,
                radius=line_thickness / 2,
                use_normalized_coordinates=use_normalized_coordinates,
                keypoint_edges=keypoint_edges,
                keypoint_edge_color=color,
                keypoint_edge_width=line_thickness // 2)

    return image


def predict(frame: np.ndarray,
            now: datetime.datetime,
            save_image: bool,
            min_score_thresh: float = 0.75) -> tuple[int, Any | None]:
    print(f"[{now:%Y-%m-%d %H:%M:%S}] Processing frame with min_score_thresh={min_score_thresh}", end="")
    # Build inference graph.
    task.config.task.max_instances_per_image_test = num_instances_to_generate
    images, _, pred_bboxes, _, pred_classes, scores, _, _, _, _, _ = _predict_image(frame, model, task, config)
    scores = scores.numpy()
    classes = pred_classes.numpy()
    pred_bboxes = pred_bboxes[0].numpy() * frame.shape[1] + np.array([-10, 0, 0, 0])

    # Compute number of people
    people_detected = int(np.sum(scores[0, pred_classes[0] == 1] > min_score_thresh))
    print(f": ~{people_detected} people detected")

    vis = None
    if save_image:
        # Visualization.
        vis = visualize_boxes_and_labels_on_image_array(
            image=tf.image.convert_image_dtype(
                # images[0],
                frame,
                tf.uint8).numpy(),
            boxes=pred_bboxes,
            classes=classes[0],
            scores=scores[0],
            category_index=categories_dict,
            use_normalized_coordinates=False,
            min_score_thresh=min_score_thresh,
            max_boxes_to_draw=100,
            # agnostic_mode=True,
            skip_boxes=True,
            skip_scores=True,
        )

        # plt.imshow(vis); plt.axis("off"); plt.tight_layout(); plt.show()

    return people_detected, vis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Wondering how crowded is your favourite beach of Gipuzkoa "
                                                 f"(Basque Country)?")
    parser.add_argument("-b", "--beach", help="Beach to analyze", choices=["Zurriola", "Kontxa", "Ondarreta"], required=True)
    parser.add_argument("-t", "--threshold", help="Minimum score threshold", type=float, default=0.75)
    parser.add_argument("-s", "--save-prediction", help="Saves prediction JPG to last.jpg", action="store_true",
                        default=False)
    parser.add_argument("-d", "--delay", help="Delay between frame captures. 0 disables (predict and exit)", default=0)
    parser.add_argument("-m", "--model-dir", help=f"Path to object detection model (default: {model_dir})",
                        default=model_dir)
    parser.add_argument("--from-gcloud", help=f"Uses GCloud stored model", action="store_true", default=False)

    args = parser.parse_args()

    task, model, config = load_model(args.model_dir, from_gcloud=False)
    categories_dict = load_category_names()

    # https://stackoverflow.com/a/69501647
    url = f'https://58f14c0895a20.streamlock.net:443/camaramar/GIP_{args.beach.lower()}_169.stream/playlist.m3u8'
    stream_url = stream_to_url(url)

    finished = False
    while not finished:
        now = datetime.datetime.now(tz=pytz.timezone("Europe/Madrid"))
        cap = cv2.VideoCapture(stream_url)

        try:
            # Read new frame
            success, frame = cap.read()
            people_detected, vis = predict(frame,
                                           now,
                                           save_image=args.save_prediction,
                                           min_score_thresh=args.threshold)
            # Save images
            if vis is not None:
                vis_output = Path(args.beach)
                vis_output.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame).save(vis_output / f"{now:%Y%m%d%H%M%S}_original.jpg")
                Image.fromarray(vis).save(vis_output / f"{now:%Y%m%d%H%M%S}_prediction.jpg")

            # Save logs to CSV
            log = pd.DataFrame({
                "beach": [args.beach],
                "people": [people_detected],
                "timestamp": [now.strftime("%Y-%m-%d %H:%M:%S%z")]
            })
            if not logbook_path.is_file():
                logbook = pd.DataFrame(columns=["timestamp", "beach", "people"])
            else:
                logbook = pd.read_csv(logbook_path)
            logbook = pd.concat([logbook, log]).drop_duplicates().sort_values(by="timestamp")
            logbook.to_csv(logbook_path, index=False)

            # Wait to process other image
            if int(args.delay) > 0:
                time.sleep(int(args.delay))
            else:
                finished = True

        except KeyboardInterrupt:
            finished = True
        finally:
            cap.release()

    exit(0)
