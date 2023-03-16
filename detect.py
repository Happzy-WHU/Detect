import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from PIL import Image 
import imagehash
import numpy as np
import copy

# 相似返回True，否则返回False
def is_similar_images(image1, image2):
    if image1 is None or image2 is None:
        return False
    
    # Load images
    image1 = Image.fromarray(cv2.cvtColor(image1,cv2.COLOR_BGR2RGB))  
    image2 = Image.fromarray(cv2.cvtColor(image2,cv2.COLOR_BGR2RGB))  
    
    # Convert images to dhash strings
    image1_dhash = str(imagehash.dhash(image1))
    image2_dhash = str(imagehash.dhash(image2))
    
    # Calculate Hamming distance
    hamming_distance = sum(c1 != c2 for c1, c2 in zip(image1_dhash, image2_dhash))
    
    # Set threshold
    threshold = 5
    
    # Compare Hamming distance to threshold
    if hamming_distance < threshold:
        return True
    else:
        return False

def calculate_iou(box1, box2):
    # 计算两个框的交集面积
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    
    # 计算两个框的并集面积
    area_box_1 = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    area_box_2 = abs((box2[2] - box2[0]) * (box2[3] - box2[1]))
    
    union_area = area_box_1 + area_box_2 - inter_area
    
    # 计算IoU
    iou = inter_area / union_area
    
    return iou

def match_boxes(list1, list2):
    b1,b2 = [],[]
    for box1 in list1:
        for box2 in list2:
            if box1[5] == box2[5]: # 检查类别是否相同
                iou = calculate_iou(box1[:4], box2[:4]) # 计算两个框的IoU
                if iou > 0.2: # 如果IoU大于阈值，则认为两个框匹配
                    b1.append(box1)
                    b2.append(box2)
    return b1,b2

# 插值实现平滑
# torch.stack(list(leftDet[0]), dim=0)
# torch.stack(b1, dim=0)
def insert_det(proQueue, n):
    leftDet = proQueue[0]["pred"]
    rightDet = proQueue[len(proQueue)-1]["pred"]
    if len(leftDet)*len(rightDet) == 0:
        return proQueue
    
    b1, b2 = match_boxes(list(leftDet[0]), list(rightDet[0]))
    for i in range(1,n+1):
        if i == 1:
            proQueue[i]["pred"].clear()
        alpha = i/(n+1)
        if len(b1) * len(b2):

            proQueue[i]["pred"].append(torch.mul(torch.stack(b1, dim=0), 1-alpha)+torch.mul(torch.stack(b2, dim=0), alpha))

    return proQueue

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    
    frame_count = 3
    save_path = "" 
    im_last = None
    im0 = None
    state = 0
    proQueue = []
    staticCount = 0
    
    for path, img, im0s, vid_cap in dataset:
        
        im0 = im0s
        
        frame_count += 1

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        if not is_similar_images(im_last, im0s) or frame_count == 4:
            im_last = im0s
            frame_count = 0
            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            
            proQueue.append({"pred":pred, "im0s":im0s, "img":img})
            if state != 0:
                proQueue = insert_det(proQueue, state)
            
        else:
            state+=1
            proQueue.append({"pred":[], "im0s":im0s, "img":img})
            continue
        
        if len(proQueue) * state:
            proQueue.pop(0)
        state = 0
        for item in proQueue:
            # Process detections
            for i, det in enumerate(item["pred"]):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, item["im0s"][i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', item["im0s"], getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    itemPred = []
                    if item is proQueue[len(proQueue)-1]:
                        itemPred = copy.deepcopy(item["pred"])
                    det[:, :4] = scale_coords(item["img"].shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                staticCount+=1
            
            if item is proQueue[len(proQueue)-1]:
                proQueue.clear()
                proQueue.append({"pred":itemPred, "im0s":im0s, "img":img})
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''

    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='detect.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0,1,2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7-tiny.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
