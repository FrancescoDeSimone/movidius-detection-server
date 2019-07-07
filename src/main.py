#! /usr/bin/env python3

from mvnc import mvncapi as mvnc
from utils.ssd_mobilenet_processor import SsdMobileNetProcessor
from utils.age_processor import AgeNetProcessor
from utils.gender_processor import GenderNetProcessor
from utils import visualize_output
import cv2
import numpy
import os
import paho.mqtt.publish as publish
import imutils
import sys
import json
from utils import clean_close
from PIL import Image
import base64
import io 

object_classifications_mask = [1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1]

def main():
    devices = mvnc.enumerate_devices()
    if len(devices) < 1:
        print('No NCS device detected.')
        print('Insert device and try again!')
        return 1

    obj_detect_list = list()
    obj_age_list = list()
    obj_gender_list = list()

    device_number = 0

    net = cv2.dnn.readNetFromCaffe("./deploy.prototxt.txt","./face_det.caffemodel")
    for one_device in devices:
        try:
            obj_detect_dev = mvnc.Device(one_device)
            obj_detect_dev.open()
            print("opened device " + str(device_number))
            obj_detector_proc = SsdMobileNetProcessor("./graphs/ssd_mobilenet", obj_detect_dev,
                                                      inital_box_prob_thresh=60 / 100.0,
                                                      classification_mask=object_classifications_mask,
                                                      name="object detector " + str(device_number))
            obj_detect_list.append(obj_detector_proc)

            obj_age_proc = AgeNetProcessor("./graphs/age_net",obj_detect_dev,name="age detector "+str(device_number))

            obj_age_list.append(obj_age_proc)

            obj_gender_proc = GenderNetProcessor("./graphs/gender_net",obj_detect_dev,name="gender detector "+str(device_number))

            obj_gender_list.append(obj_gender_proc)

            device_number += 1

        except:
            print("Could not open device " + str(device_number) + ", trying next device")
            pass


    if len(obj_detect_list) < 1 or len(obj_gender_list) < 1:
        print('Could not open any NCS devices.')
        print('Reinsert devices and try again!')
        return 1

    print("Using " + str(len(obj_detect_list)) + " devices for object detection")

    camera = cv2.VideoCapture(0)
    cc = clean_close.clean_close();
    while(True):
        if cc.close:
            for one_obj_detect_proc in obj_detect_list:
                one_obj_detect_proc.cleanup(True)
            for one_obj_age_proc in obj_age_list:
                one_obj_age_proc.cleanup(True)
            for one_obj_gender_proc in obj_gender_list:
                one_obj_gender_proc.cleanup(True)
            camera.release();
            break;

        for one_obj_detect_proc, one_obj_age_proc, one_obj_gender_proc in zip(obj_detect_list,obj_age_list,obj_gender_list):
            ret, frame = camera.read()
            send_frame = frame
            one_obj_detect_proc.start_aysnc_inference(frame)
            (detections, display_image) = one_obj_detect_proc.get_async_inference_result()
            if len(detections) > 0:
                for object in detections:
                    payload = {}
                    payload['class'] = str(object[0])
                    payload['box'] = str(object[1:5])
                    payload['score'] = str(object[5])
                    send_frame = visualize_output.draw_bounding_box(object[1],object[2],object[3],object[3],send_frame,display_str=str(object[0]))
                    if(payload['class'] == "person"):
                        person = frame
                        (h, w) = person.shape[:2]
                        person = imutils.resize(person,width=400)
                        blob = cv2.dnn.blobFromImage(cv2.resize(person, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))
                        net.setInput(blob)
                        detections = net.forward()
                        for i in range(0,detections.shape[2]):
                            confidence = detections[0,0,i,2];
                            if confidence < 80/100:
                                continue
                            box_face = ((detections[0,0,i,3:7] * numpy.array([w,h,w,h])).astype("int") + [-50,-50,+50,+50])
                            one_obj_age_proc.start_aysnc_inference(person,box_face)
                            one_obj_gender_proc.start_aysnc_inference(person,box_face)
                            (age_prediction, age_detection) = one_obj_age_proc.get_async_inference_result()
                            (gender_prediction, gender_detection) = one_obj_gender_proc.get_async_inference_result()
                            send_frame = visualize_output.draw_bounding_box(box_face[1],box_face[0],box_face[3],box_face[2],send_frame,display_str=(str(gender_detection)+" - "+str(age_detection)))
                            gender = {}
                            age = {}
                            age['score'] = age_prediction
                            age['detection'] = age_detection
                            gender['score'] = gender_prediction
                            gender['detection'] = gender_detection
                            payload['face_box'] = str((detections[0,0,i,3:7] * numpy.array([w,h,w,h])).astype("int"))
                            payload['age'] = json.dumps(age) 
                            payload['gender'] = json.dumps(gender) 
                    print(payload)
                    publish.single("data/"+payload['class'],payload=json.dumps(payload),hostname="localhost")
                    #byte_frame = io.BytesIO()
                    #Image.fromarray(cv2.cvtColor(send_frame,cv2.COLOR_BGR2RGB)).save(byte_frame,format="png")
                    #publish.single("frame",payload=str(base64.b64encode(byte_frame.getvalue())),hostname="localhost")
    
main()

