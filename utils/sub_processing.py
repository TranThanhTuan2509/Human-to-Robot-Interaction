from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import cv2
import torch
from mmaction2.mmaction.apis import inference_recognizer, init_recognizer
from moviepy.editor import VideoFileClip
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import os
import shutil

def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.
    Returns:
        dict: The label map (int -> label name).
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = {i: x.strip() for i, x in enumerate(lines)}
    return lines

def overlap(interval_1, interval_2):
    x1, x2 = interval_1
    x3, x4 = interval_2
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def compute_iou(box1, box2):
    """box1 stands for object, box2 stands for hand"""
    """Compute IOU between box1 and box2"""
    x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
    x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]

    ## if box2 is inside box1
    if (x1 < x2) and (y1 < y2) and (w1 > w2) and (h1 > h2):
        return 1

    area1, area2 = w1 * h1, w2 * h2
    intersect_w = overlap((x1, x1 + w1), (x2, x2 + w2))
    intersect_h = overlap((y1, y1 + h1), (y2, y2 + w2))
    intersect_area = intersect_w * intersect_h
    iou = intersect_area / (area1 + area2 - intersect_area)
    return iou

def clearest_frames(obj_dets, hand_dets):
    obj_bbox = list(int(np.round(x)) for x in obj_dets[0, :4])
    hand_bbox = list(int(np.round(x)) for x in hand_dets[0, :4])
    score = compute_iou(obj_bbox, hand_bbox)
    return score

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def setup_directories(args):
    for directory in [args.cropped_pick_dir, args.cropped_place_dir, args.image_dir, args.save_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

def load_faster_rcnn_model(args, pascal_classes):
    print("Loading Faster R-CNN model...")

    model_dir = os.path.join(args.load_dir, f"{args.net}_handobj_100K", args.dataset)

    if not os.path.exists(model_dir):
        raise Exception(f"No input directory found for loading network from {model_dir}")

    load_name = os.path.join(model_dir, f'faster_rcnn_{args.checksession}_{args.checkepoch}_{args.checkpoint}.pth')

    # Load model only once
    if args.net == 'vgg16':
        model = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        model = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        model = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        model = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        raise ValueError("Invalid network type")

    model.create_architecture()
    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    model.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    # model.load_state_dict(torch.load(load_name, map_location='cuda' if args.cuda else 'cpu')['model'])
    model.eval()

    return model

def action_recognition(args):
    model = init_recognizer(args.config, args.action_checkpoint, device=args.device)  # device can be 'cuda:0'
    # test a single image
    result = inference_recognizer(model, args.reduced_frame_video)

    predict_value = torch.argmax(result.pred_score, dim=0).item()
    truth_label = load_label_map(args.label_map)
    print(predict_value)
    action = list(truth_label.values())[predict_value]
    print('\n', action)
    return action

def crop_image(image, bbox, expand_ratio):
    """
    Expands the bounding box by a certain percentage while ensuring it stays within image boundaries."""
    h_img, w_img = image.shape[:2]  # Get image dimensions
    x, y, w, h = map(int, bbox)

    # Calculate expansion size
    expand_w = int(w * expand_ratio / 2)
    expand_h = int(h * expand_ratio / 2)

    # Expand the bounding box
    x1 = max(0, x - expand_w)  # Ensure x1 is not < 0
    y1 = max(0, y - expand_h)  # Ensure y1 is not < 0
    x2 = min(w_img, x + w + expand_w)  # Ensure x2 is not > image width
    y2 = min(h_img, y + h + expand_h)  # Ensure y2 is not > image height

    return image[y1:y2, x1:x2]  # Cropped image with safe expansion


def reduce_frame(args):
    clip = VideoFileClip(args.video)

    # Set the new frame rate
    new_fps = 25

    # Write the video file with the new frame rate
    clip.set_duration(clip.duration).set_fps(new_fps).write_videofile(args.reduced_frame_video, codec='libx264')

    print('\nSuccessfully reduce the frame of the input video.')

def filter_abnormal_bboxes(bboxes, base_bbox=None, shape_threshold=0.3):
    """
    Lọc bỏ bounding boxes có kích thước hoặc hình dạng bất thường dựa trên bbox chuẩn.

    Args:
        bboxes (list): Danh sách bbox [(w, h, x, y)]
        base_bbox (tuple): BBox gốc để tính aspect ratio (w, h)
        shape_threshold (float): Ngưỡng sai lệch kích thước cho phép (mặc định: 30%)

    Returns:
        List: Danh sách bbox hợp lệ.
    """

    if not bboxes:
        return []  # Không có bbox nào, trả về list rỗng

    # Nếu không có base_bbox, chọn bbox đầu tiên làm chuẩn
    if base_bbox is None:
        base_bbox = bboxes[0]  # Lấy bbox đầu tiên làm chuẩn

    base_w, base_h = base_bbox[:2]
    base_aspect_ratio = base_w / base_h  # Tính tỷ lệ w/h gốc

    # Xác định khoảng hợp lệ cho aspect ratio
    aspect_ratio_min = base_aspect_ratio * (1 - shape_threshold)
    aspect_ratio_max = base_aspect_ratio * (1 + shape_threshold)

    # Tính trung vị của w, h
    widths = [bbox[0] for bbox in bboxes]
    heights = [bbox[1] for bbox in bboxes]

    median_w, median_h = np.median(widths), np.median(heights)

    # Ngưỡng trên & dưới để loại bỏ bbox quá to hoặc quá nhỏ
    min_w, max_w = median_w * (1 - shape_threshold), median_w * (1 + shape_threshold)
    min_h, max_h = median_h * (1 - shape_threshold), median_h * (1 + shape_threshold)

    # Lọc bounding boxes hợp lệ
    filtered_bboxes = []
    for bbox in bboxes:
        w, h, x, y = bbox
        aspect_ratio = w / h  # Tỷ lệ w/h

        if (min_w <= w <= max_w and min_h <= h <= max_h and
            aspect_ratio_min <= aspect_ratio <= aspect_ratio_max):
            filtered_bboxes.append(bbox)

    return filtered_bboxes

def description(action, categories):
    action = action.split(' ')
    cate = 0
    sth = 0
    for i, a in enumerate(action):
        if action[i] == 'something':
            sth += 1
            # print(cate, categories)
            if sth > len(categories):
                break
            action[i] = categories[cate] #For using LLMs
            cate += 1
    return ' '.join(action)
