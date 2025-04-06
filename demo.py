# python3 demo.py --checkepoch=8 --checkpoint=132028 --video /home/tuan/Downloads/picking_cube.mp4
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
'''Label definitions:

    L: left hand
    R: right hand
    N: no contact
    S: self contact
    O: other person contact
    P: portable object contact
    F: stationary object contact (e.g.furniture)
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from torchvision.models import resnet50
import os
import sys
import numpy as np
import argparse
import time
import cv2
import torch
import torch.nn as nn
from Video_keyframe_detector.KeyFrameDetector.key_frame_detector import keyframeDetection
import shutil
import logging
# from multitask.main import *
from LLaVA.inference_LLaVA import caption_image

from utils.sub_processing import (clearest_frames, variance_of_laplacian
, _get_image_blob, action_recognition, reduce_frame, description, crop_image, load_faster_rcnn_model, setup_directories)
from utils.multivideo import split_video
try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')
# from scipy.misc import imread
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections_filtered_objects_PIL  # (1) here add a function to viz

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

ROOT_PATH = os.path.abspath('.')
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Action recognition and hand object detection')
  parser.add_argument('--video', default=' mmaction2/demo'
                                         '/practical_video/picking_reg_cube.mp4', help='video file/url')
  parser.add_argument('--reduced_frame_video', default=f'{ROOT_PATH}/reduced_frame_video/reduced_frame_video.mp4', help='reduced frame video')
  parser.add_argument(
      '--config',
      default=(f'{ROOT_PATH}/mmaction2/pretrained_file_and_checkpoint/'
               'tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb.py'),
      help='action recognition model config file path')
  parser.add_argument(
      '--action_checkpoint',
      default=(f'{ROOT_PATH}/mmaction2/pretrained_file_and_checkpoint/'
               'tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb_20230317-ec6696ad.pth'),
      help='action recognition model checkpoint file/url')
  # parser.add_argument(
  #     '--robotu_checkpoint',
  #     default=(f'{ROOT_PATH}/multitask/checkpoint/checkpoint_40000/content/checkpoint_40000'),
  #     help='robot understanding module checkpoint file/url')
  parser.add_argument(
      '--label_map',
      default=f'{ROOT_PATH}/mmaction2/label_map.txt',
      help='label map file')
  parser.add_argument('--kf_thres', type=float,
                      default=0.1,
                      help='key frame threshold')
  parser.add_argument(
      '--device', type=str, default='cpu', help='CPU/CUDA device option')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--motion_saver', dest='motion_saver',
                      help='directory to save the motion description',
                      default=f'{ROOT_PATH}/motion_saver')
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default=f'{ROOT_PATH}')
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default=f'{ROOT_PATH}/images')
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save results',
                      default=f'{ROOT_PATH}/output')
  parser.add_argument('--cropped_pick_dir', dest='cropped_pick_dir',
                      help='directory to save cropped picked object',
                      default=f'{ROOT_PATH}/cropped_pick')
  parser.add_argument('--cropped_place_dir', dest='cropped_place_dir',
                      help='directory to save cropped placed object',
                      default=f'{ROOT_PATH}/cropped_place')
  parser.add_argument('--cuda', dest='cuda', 
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=8, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=132028, type=int, required=True)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      default=True)
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--thresh_hand',
                      type=float, default=0.5,
                      required=False)
  parser.add_argument('--thresh_obj', default=0.5,
                      type=float,
                      required=False)
  parser.add_argument("--image_size", "-s", type=int, default=224, help='fixed image size')
  parser.add_argument("--checkpoint_path", "-m", type=str,
                      default=f'{ROOT_PATH}/classification_branch/checkpoints/no_best.pt',
                      help='classification checkpoint')

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
# START RUNNING

def main(args):
    video_dir = "Dataset/Dataset"
    video_list = sorted(
       [f for f in os.listdir(video_dir) if f.endswith(".MOV") or f.endswith(".mp4")],
       key=lambda x: int(x.split(".")[0])
   )
    # video_list = ["IMG_2577.MOV","IMG_2589.MOV"]
    # for i in range(len(video_list)):
    #     folder_path = os.path.join("Dataset/Best", str(i + 1))
    #
    #     if os.path.isdir(folder_path):
    #         shutil.rmtree(folder_path)
    #
    #     os.makedirs(folder_path)
    # folder_list = sorted(
    #     [f for f in os.listdir("Dataset/Best")],
    #     key=lambda x: int(x.split(".")[0])
    # )

    # set multitask flag here
    # Load Detection Model
    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
    fasterRCNN = load_faster_rcnn_model(args, pascal_classes)
    # for _, (video, folder) in enumerate(zip(video_list, folder_list)):  # Fixed unpacking issue
    for _, video in enumerate(video_list):  # Fixed unpacking issue
        # folder_path = os.path.join("Dataset/Best", folder)
        args.video = os.path.join(video_dir, video)
        setup_directories(args)

        # Reduce the video to take out the appropriate input for the action recognition
        reduce_frame(args)

        # Action recognition
        action = action_recognition(args)

        # Keyframe extraction
        keyframeDetection(args.reduced_frame_video, args.image_dir, args.kf_thres)

        if args.cfg_file is not None:
            cfg_from_file(args.cfg_file)

        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs)

        cfg.USE_GPU_NMS = args.cuda
        np.random.seed(cfg.RNG_SEED)

        # initilize the tensor holder here.
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)
        box_info = torch.FloatTensor(1)

        # ship to cuda
        if args.cuda > 0:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_boxes = num_boxes.cuda()
            gt_boxes = gt_boxes.cuda()
            fasterRCNN.cuda()
            cfg.CUDA = True

        with torch.no_grad():

            start = time.time()
            max_per_image = 100
            thresh_hand = args.thresh_hand
            thresh_obj = args.thresh_obj
            vis = args.vis

            webcam_num = args.webcam_num
            # Set up webcam or get image directories
            if webcam_num >= 0 :
                cap = cv2.VideoCapture(webcam_num)
                num_images = 0
            else:
                logging.info(f'image dir = {os.path.join(args.image_dir, "keyFrames")}')
                logging.info(f'save dir = {args.save_dir}')
                imglist = sorted([f for f in os.listdir(os.path.join(args.image_dir, "keyFrames"))
                                  ], reverse=True, key=lambda x: int(x.split(".")[0]))
                # reverse the images to arrange the right order
                num_images = len(imglist)
                num_images_for_index = len(imglist)

            logging.info('Loaded Photo: {} images.'.format(num_images))
            index = 0
            # for picked object
            list_for_cropped_picked_object_link = dict()
            list_for_picking_blur = dict()
            # for placed object
            list_for_cropped_placed_object_link = dict()
            list_for_placing_blur = dict()

            # initializing a variable to avoid detecting bowl object only
            previous_area_of_cuboid = float('inf')
            previous_y_coor_of_obj2 = float('inf')
            while (num_images >= 1 and index <= num_images_for_index):
                total_tic = time.time()
                index += 1
                if webcam_num == -1:
                    num_images -= 1
                logging.info(f"Loading image {num_images_for_index-num_images}/{num_images_for_index}\n")
                # Get image from the webcam
                if webcam_num >= 0:
                    if not cap.isOpened():
                        raise RuntimeError("Webcam could not open. Please check connection.")
                    ret, frame = cap.read()
                    im_in = np.array(frame)
                # Load the demo image
                else:
                    im_file = os.path.join(os.path.join(args.image_dir, "keyFrames"), imglist[num_images])
                    im_in = cv2.imread(im_file)
                    # fix
                # bgr
                im = im_in
                blobs, im_scales = _get_image_blob(im)
                assert len(im_scales) == 1, "Only single-image batch implemented"
                im_blob = blobs
                im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

                im_data_pt = torch.from_numpy(im_blob)
                im_data_pt = im_data_pt.permute(0, 3, 1, 2)
                im_info_pt = torch.from_numpy(im_info_np)

                with torch.no_grad():
                    im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                    im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                    gt_boxes.resize_(1, 1, 5).zero_()
                    num_boxes.resize_(1).zero_()
                    box_info.resize_(1, 1, 5).zero_()

                    # pdb.set_trace()
                det_tic = time.time()
                logging.info("Detecting....")
                # boxes
                rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

                logging.info("Detect successfully!")
                scores = cls_prob.data
                boxes = rois.data[:, :, 1:5]

                # extact predicted params
                contact_vector = loss_list[0][0] # hand contact state info
                offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
                lr_vector = loss_list[2][0].detach() # hand side info (left/right)

                # get hand contact
                _, contact_indices = torch.max(contact_vector, 2)
                contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

                # get hand side
                lr = torch.sigmoid(lr_vector) > 0.5
                lr = lr.squeeze(0).float()

                if cfg.TEST.BBOX_REG:
                    # Apply bounding-box regression deltas
                    box_deltas = bbox_pred.data
                    # print(f"\nbox_deltas shape: {box_deltas.shape}")
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        # Optionally normalize targets by a precomputed mean and stdev
                        if args.class_agnostic:
                            if args.cuda > 0:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            else:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                            box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            if args.cuda > 0:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            else:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                            box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))
                    # Predict box
                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    # print(f"\n pred_boxes bbox_transform_inv shape: {pred_boxes.shape}")
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                    # print(f"\npred_boxes clip_boxes shape: {pred_boxes.shape}")
                else:
                    # Simply repeat the boxes, once for each class
                    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                pred_boxes /= im_scales[0]
                scores = scores.squeeze()
                pred_boxes = pred_boxes.squeeze()
                det_toc = time.time()
                detect_time = det_toc - det_tic
                misc_tic = time.time()
                if vis:
                    im2show = np.copy(im)
                # initialize checking variables
                obj_dets, hand_dets, num_detected_object = None, None, 0
                obj_checking, hand_checking = None, None
                for j in xrange(1, len(pascal_classes)):
                    # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
                    if pascal_classes[j] == 'hand':
                        inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
                    elif pascal_classes[j] == 'targetobject':
                        inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)

                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = scores[:,j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if args.class_agnostic:
                            cls_boxes = pred_boxes[inds, :]
                        else:
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                        'second is hand and interacted-object'
                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        "object"
                        if pascal_classes[j] == 'targetobject':
                            # check1
                            obj_checking = True
                            # check number of object
                            num_detected_object = len(cls_dets.tolist())
                            obj_dets = cls_dets.cpu().numpy()

                        "hand"
                        if pascal_classes[j] == 'hand':
                            # check2
                            hand_checking = True
                            hand_dets = cls_dets.cpu().numpy()
                            # print(f"\nhand_dets: {hand_dets}")
                ''' Initialize checking variable to avoid detecting hand or object only '''
                if type(obj_dets) == type(hand_dets) and (obj_checking == True) and (hand_checking == True):
                    obj_checking, hand_checking = None, None
                    area_comparison = []
                    for i in range(num_detected_object):
                        # fix here
                        overlap_score = clearest_frames(obj_dets, hand_dets)
                        bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
                        x_top_left, y_top_left, x_bottom_right, y_bottom_right = list(map(int, bbox))
                        x = x_top_left
                        y = y_top_left
                        w = x_bottom_right - x_top_left
                        h = y_bottom_right - y_top_left

                        # object bottom left coordinate
                        y_bottom_left = y_bottom_right - h
                        # area_comparison.append([w*h, x, y, w, h])
                        area_comparison.append([w*h, y_bottom_left, x, y, w, h])
                    area_comparison = sorted(area_comparison, reverse=False, key = lambda x : int(x[1]))
                    # print(area_comparison)
                    """PLACE"""
                    if len(area_comparison) == 2:
                        # fix
                        previous_area_of_cuboid = area_comparison[0][0]
                        previous_y_coor_of_obj2 = area_comparison[1][1]

                        # Crop image
                        # crop_place_img = im[y:y + h, x:x + w]
                        crop_place_img = crop_image(im, area_comparison[1][2:], expand_ratio=0.4)
                        # Write image
                        cv2.imwrite(os.path.join(args.cropped_place_dir, f"cropped_{index}.jpg"), crop_place_img)
                        rate = (4000 - variance_of_laplacian(
                            cv2.cvtColor(crop_place_img, cv2.COLOR_BGR2GRAY))) + overlap_score + area_comparison[1][0]
                        # resize cropped images
                        list_for_cropped_placed_object_link.update(
                            {f'image_{index}': os.path.join(args.cropped_place_dir, f"cropped_{index}.jpg")})
                        list_for_placing_blur.update({f'image_{index}': rate})

                    """PICK"""
                    if ((area_comparison[0][0] < 1.3 * previous_area_of_cuboid) #In case, the only object that model catches is the placable object
                            and (previous_y_coor_of_obj2 > area_comparison[0][1])):
                        # Crop image
                        crop_pick_img = crop_image(im, area_comparison[0][2:], expand_ratio=0.8)
                        # Write image
                        cv2.imwrite(os.path.join(args.cropped_pick_dir, f"cropped_{index}.jpg"), crop_pick_img)
                        # rate = (4000 - variance_of_laplacian(
                        #     cv2.cvtColor(crop_pick_img, cv2.COLOR_BGR2GRAY))) + overlap_score + area_comparison[0][0]
                        rate = area_comparison[0][1]
                        # resize cropped images
                        list_for_cropped_picked_object_link.update(
                            {f'image_{index}': os.path.join(args.cropped_pick_dir, f"cropped_{index}.jpg")})
                        list_for_picking_blur.update({f'image_{index}': rate})

                    logging.info(f'Pickable object is detected!')
                    if num_detected_object == 2:
                        logging.info(f'Placable object is detected!')
                    """ Remove high occluded image by using iou """


                    ''' Image Processing '''
                    if vis:
                        # visualization
                        im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)

                    misc_toc = time.time()
                    nms_time = misc_toc - misc_tic

                    if webcam_num == -1:
                        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                                         .format(num_images + 1, len(imglist), detect_time, nms_time))
                        sys.stdout.flush()

                    if vis and webcam_num == -1:

                        folder_name = args.save_dir
                        os.makedirs(folder_name, exist_ok=True)
                        result_path = os.path.join(folder_name, imglist[num_images][:-4] + "_det.png")
                        im2show.save(result_path)
                    else:
                        im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
                        cv2.imshow("frame", im2showRGB)
                        total_toc = time.time()
                        total_time = total_toc - total_tic
                        frame_rate = 1 / total_time
                        logging.info('Frame rate:', frame_rate)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                else:
                    pass

            if webcam_num >= 0:
                cap.release()
                cv2.destroyAllWindows()
            image_links = []
            image_lists = []
            if list_for_placing_blur != dict():
                The_lowest_blured_and_occluded_place_image = dict(
                    [sorted(list_for_placing_blur.items(), reverse=False, key=lambda x: int(x[1]))[0]])
                intersect = set(list_for_cropped_placed_object_link).intersection(
                    set(The_lowest_blured_and_occluded_place_image))
                place = [list_for_cropped_placed_object_link[i] for i in intersect][0]
                print(place)
                image_links.append(place)

            The_lowest_blured_and_occluded_pick_image = dict(
                [sorted(list_for_picking_blur.items(), reverse=False, key=lambda x:int(x[1]))[0]])
            intersect = set(list_for_cropped_picked_object_link).intersection(
                set(The_lowest_blured_and_occluded_pick_image))
            pick = [list_for_cropped_picked_object_link[i] for i in intersect][0]
            print(pick)
            image_links.append(pick)
            logging.info("Generating....")
            '''For LLMs'''
            prompt = (
                "Identify the main object in the image that is directly interacted with by a human hand. "
                "Describe it using exactly two words: first its color, then its object type. "
                "Format: 'the [color] [object]', e.g., 'the blue block'. "
            )
            responses = [caption_image(image_path, prompt) for image_path in image_links[::-1]]
            # Save action result and object result
            with open(os.path.join(args.motion_saver, 'motion_saver.txt'), 'a') as file:
                file.write(description(action, responses) + "\n")

if __name__ == '__main__':
    args = parse_args()
    main(args)
