# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import os
import argparse
import time
import platform
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadWebcam
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2, apply_classifier,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, set_logging, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, time_sync

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QFrame
from PyQt5.QtCore import QSize, QRect, Qt, QThread, QTimer
from PyQt5.QtGui import QFont, QPixmap, QImage
from PIL import ImageFont, ImageDraw, Image
from collections import OrderedDict
from queue import Queue

continuous = {'안녕하세요 ':["hello1","hello2"], '하나 둘 김치 ':["one","two"], '콧물 ':["runnynose1","runnynose2"],
              '쓰러지다 ':["fall1","fall2"], '설사 ':["diarrhea1","diarrhea2"], '입원 ':["hospitalization1","hospitalization2","hospitalization3"],
              '퇴원 ':["hospitalization3","hospitalization2","hospitalization1"],
              '완쾌 ':["recovery1","recovery2","recovery3"], '소화불량 ' :["digestion1","digestion2","poor"], '변비 ':["constipation1","constipation2","constipation3"],
              '소변 ':["urine1","urine2"], '수술 ':["surgery1","surgery2"],  '낫다 ':["","recovery3"]}
#핵심 이미지가 여러개인 수화 동작 저장

one = {'Glasses':'안경 ', 'yes':'네 ', 'head':'머리 ', 'stomach':'배 ', 'sick':'아프다 ','reset':'','medicine':'약 '}
#핵심 이미지가 하나인 수화 동작 저장

list_of_key = list(continuous.keys())
list_of_value = list(continuous.values())
#핵심 이미지가 여러개인 단어인 경우,
#단어 별 핵심 이미지들은 value에 저장, 한국어 단어는 key에 저장

b,g,r,a = 255,255,255,0
fontpath = "fonts/gulim.ttc"
font = ImageFont.truetype(fontpath, 20)
#한글을 출력할 폰트 설정

class appGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        #self.video = video_stream()

        self.paddingX = 100
        self.paddingY = 130
        self.vWidth = 800
        self.vHeight = 600

        self.mw_height = 900
        self.mw_width = 1100

        self.scale = 1.6
        self.midX = (self.vWidth * self.scale) / 2
        self.midY = (self.vHeight * self.scale) / 2

        self.trackX = 0
        self.trackY = 0
        self.predList = []  # Save the detected coordinates for each frame
        self.TRACKFLAG = False
        self.tracked_X = 0
        self.tracked_Y = 0

        self.init_gui()

    def detect(self):
        source, weights, view_img, save_txt = opt.source, opt.weights, opt.view_img, opt.save_txt
        data = ROOT / 'data/coco128.yaml',
        imgsz=(640,640)
        conf_thres=0.25, #여기서부터 추가
        iou_thres=0.45,
        max_det=1000,
        device='',
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False

        random.seed(3)
        label = ""
        word = ""  # 최종으로 출력할 단어
        sentence = []  # 출력할 문장

        result = ""  # 현재 결과
        before_result = ""  # 이전 결과
        result_que = Queue(3)  # result들을 저장하는 큐 생성. 현재 결과까지 최대 3개 저장

        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) #추가
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) #추가
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file: #추가
            source = check_file(source)  # download

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
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        #classify = False
        #if classify:
            #modelc = load_classifier(name='resnet101', n=2)  # initialize
            #modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            save_img = True
            view_img = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap, s in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            #if classify:
                #pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop #crop추가
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        label = names[int(c)]
                        # UI에 label 출력
                        #self.text.setText(str(names[int(c)])) # label 출력
                        #self.text.setText(str(one.get(names[int(c)])))
                        # 문장 출력
                        sentence = list(OrderedDict.fromkeys(list(sentence)))
                        self.text.setText(''.join(sentence))

                        # result가 null이 아닌 경우에만 before_result에 저장
                        if result != "":
                            before_result = result

                        # 디텍션 결과가 null이 아닌 경우에만 result에 저장
                        if label != "":
                            result = label

                            # 이전 결과와 현재 결과가 다른 경우에만 결과 큐에 저장
                            if (before_result != result and result not in list(one.keys())):
                                if (not result_que.full()):
                                    result_que.put(result)
                                else:
                                    # 큐가 가득 차있으면 원소 제거 후 삽입
                                    result_que.get()
                                    result_que.put(result)

                            # 핵심동작 1개인 수화 출력
                            if label in list(one.keys()):
                                if label == 'reset':
                                    # 인식한 이미지가 리셋일 경우 문장 초기화
                                    sentence = []
                                else:
                                    # 리셋이 아닐경우 문장에 추가
                                    sentence.append(one.get(label))
                                result_que = Queue(3)

                            list_of_result = list(result_que.queue)
                            # 큐를 리스트로 변환

                            # 핵심동작 2개,3개인 수화 출력
                            for i in range(len(list_of_key)):
                                if list_of_result == list_of_value[i] or list_of_result[1:] == list_of_value[i]:
                                    # 현재까지 저장된 result들을 토대로 단어 생성
                                    word = list_of_key[i]
                                    sentence.append(word)
                                    # 출력할 문장에 최종 단어 추가
                                    break

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img or save_crop:  # Add bbox to image
                            c = int(cls)
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            label_name = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label_name, color=colors[c], line_thickness=3)

                        if save_crop: #crop 추가
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                # UI 화면에 출력
                cv2.cvtColor(im0, cv2.COLOR_BGR2RGB, im0)
                self.show_frame(im0)

                # Stream results
                if view_img:
                    #cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
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

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')

    def init_gui(self):
        # Height, Width for QMainWindow

        # mw_height = 900
        # mw_width = 1200

        # Resize for QMainWindow
        self.resize(self.mw_width, self.mw_height)
        #self.resize(800, 600)
        # Fixed Size for QMainWindow
        self.setMinimumSize(QSize(self.mw_width, self.mw_height))
        self.setMaximumSize(QSize(self.mw_width, self.mw_height))
        # Title for QMainWindow
        self.setWindowTitle("sign language interpreter")

        # Define QWidget
        self.centralWidget = QWidget(self)
        # Resize for QWidget
        self.centralWidget.resize(self.mw_width, self.mw_height)

        # Define QFont
        self.font = QFont()
        self.font.setPointSize(12)
        self.font.setBold(True)
        self.font.setWeight(75)

        # Define QPushButtons
        self.pushButton_open_camera = QPushButton("Open Camera", self.centralWidget)
        self.pushButton_open_camera.setGeometry(QRect(260, 50, 151, 50))
        self.pushButton_open_camera.setFont(self.font)

        self.pushButton_close_camera = QPushButton("Close Camera", self.centralWidget)
        self.pushButton_close_camera.setGeometry(QRect(80, 50, 161, 51))
        self.pushButton_close_camera.setFont(self.font)

        # Define QLABEL for VIDEO frame
        self.label_img = QLabel(self.centralWidget)
        # starting point (50，130) draw（640 * 480）the box
        # self.label_img.setGeometry(QRect(50, 130, 640, 480))  Horizontal first, vertical later
        self.label_img.setGeometry(QRect(self.paddingX, self.paddingY, self.vWidth, self.vHeight))
        self.label_img.setFrameShape(QFrame.Box)
        #self.label_img.setText("")
        
        # 수화 번역 창
        self.text = QLabel(self.centralWidget)
        self.text.setGeometry(QRect(self.paddingX, self.paddingY+650, self.vWidth, self.vHeight-550))
        self.text.setFrameShape(QFrame.Box)
        self.text.setText("hello world")
        self.text.setFont(QFont("궁서",20))

        # SIGNALS
        self.pushButton_open_camera.clicked.connect(self.on_pushButton_open_camera_clicked)
        self.pushButton_close_camera.clicked.connect(self.on_pushButton_close_camera_clicked)

        # Create a thread for running detect()
        self.acquisition_timer = QTimer()
        self.acquisition_timer.timeout.connect(self.update_frame)

    def on_pushButton_open_camera_clicked(self):
        # self.video.acquisition()
        self.acquisition_timer.start(1)

    def on_pushButton_close_camera_clicked(self):
        self.video.close()
        self.acquisition_timer.stop(1)

    def update_frame(self):
        self.detect()
        cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB, self.frame)

    def show_frame(self, frame):  # qt display function
        QImg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixMap = QPixmap.fromImage(QImg)
        pixMap = pixMap.scaled(self.vWidth, self.vHeight, Qt.KeepAspectRatio)
        self.label_img.setPixmap(pixMap)

    def mousePressEvent(self, event):
        self.setMouseTracking(True)
        s = event.windowPos()
        x = s.x() - self.paddingX;
        y = s.y() - self.paddingY;

        # Since the video pixels are 1920*1080, the ratio corresponding to 1200*900 of the qt interface is 1.6
        # print('origin',x,y)
        # mouse coordinates
        x *= self.scale
        y *= self.scale
        print(x, y)

        if (x >= 0 and y >= 0) and (x <= self.vWidth * self.scale and y <= self.vHeight * self.scale):

            for pred in self.predList:  # Traverse the detection box list
                left_top = pred[0]
                right_bottom = pred[1]

                if (self.TRACKFLAG == False and x >= int(left_top[0]) and x <= int(right_bottom[0])) and (
                        y >= int(left_top[1]) and y <= int(right_bottom[1])):
                    self.TRACKFLAG = True
                    self.tracked_X = (left_top[0] + right_bottom[0]) / 2
                    self.tracked_Y = (left_top[1] + right_bottom[1]) / 2
                    print('tracked_X', self.tracked_X)
                    print('tracked_Y', self.tracked_Y)
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()

    app = QApplication(sys.argv)
    ui = appGUI()
    ui.show()
    sys.exit(app.exec_())
