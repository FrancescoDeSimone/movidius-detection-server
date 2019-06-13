import paho.mqtt.publish as publish
import os
import cv2
import sys
import numpy
import ntpath
import argparse
import socket
import mvnc.mvncapi as mvnc
import json
import base64 

from utils import visualize_output
from utils import deserialize_output
from utils import clean_close


CONFIDANCE_THRESHOLD = 0.60
ARGS                 = None
camera               = None

def open_ncs_device():
    devices = mvnc.EnumerateDevices()
    if len( devices ) == 0:
        print( "No devices found" )
        quit()
    device = mvnc.Device( devices[0] )
    device.OpenDevice()
    return device

def load_graph( device ):
    with open( ARGS.graph, mode='rb' ) as f:
        blob = f.read()
    graph = device.AllocateGraph( blob )
    return graph

def pre_process_image( frame ):
    img = cv2.resize( frame, tuple( ARGS.dim ) )
    if( ARGS.colormode == "rgb" ):
        img = img[:, :, ::-1]
    img = img.astype( numpy.float16 )
    img = ( img - numpy.float16( ARGS.mean ) ) * ARGS.scale
    return img

def send_to_topic( graph, img, frame ):
    graph.LoadTensor( img, 'user object' )
    output, userobj = graph.GetResult()
    output_dict = deserialize_output.ssd( 
                      output, 
                      CONFIDANCE_THRESHOLD, 
                      frame.shape )
    for i in range( 0, output_dict['num_detections'] ):
        (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
        (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]
        display_str = ( 
                labels[output_dict.get('detection_classes_' + str(i))]
                + ": "
                + str( output_dict.get('detection_scores_' + str(i) ) )
                + "%" )
        frame = visualize_output.draw_bounding_box(y1,x1,y2,x2,frame,thickness = 4, color=(255,0,0), display_str=display_str)
        payload = {}
        object_class = labels[output_dict.get('detection_classes_' + str(i))].split(":")[1].strip()
        payload['class'] = object_class 
        payload['confidence'] = output_dict.get('detection_scores_' + str(i))
        ret, frame_buff = cv2.imencode('.jpg',frame)
        payload['frame'] = str(base64.b64encode(frame_buff))
        publish.single("detection/"+object_class,payload=json.dumps(payload),hostname="localhost")

    #cv2.imshow('NCS live inference', frame )

def close_ncs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()
    camera.release()
    cv2.destroyAllWindows()

def main():

    device = open_ncs_device()
    graph = load_graph( device )
    cc = clean_close.clean_close();
    while( True ):
        ret, frame = camera.read()
        img = pre_process_image( frame )
        send_to_topic( graph, img, frame )
        if cc.close:
            break;
        #if( cv2.waitKey( 5 ) & 0xFF == ord( 'q' ) ):
        #    break

    close_ncs_device( device, graph )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument( '-g', '--graph', type=str,
                         default='./caffe/SSD_MobileNet/graph',
                         help="Absolute path to the neural network graph file." )

    parser.add_argument( '-v', '--video', type=int,
                         default=0,
                         help="Index of your computer's V4L2 video device. \
                               ex. 0 for /dev/video0" )

    parser.add_argument( '-l', '--labels', type=str,
                         default='./caffe/SSD_MobileNet/labels.txt',
                         help="Absolute path to labels file." )

    parser.add_argument( '-M', '--mean', type=float,
                         nargs='+',
                         default=[127.5, 127.5, 127.5],
                         help="',' delimited floating point values for image mean." )

    parser.add_argument( '-S', '--scale', type=float,
                         default=0.00789,
                         help="Absolute path to labels file." )

    parser.add_argument( '-D', '--dim', type=int,
                         nargs='+',
                         default=[300, 300],
                         help="Image dimensions. ex. -D 224 224" )

    parser.add_argument( '-c', '--colormode', type=str,
                         default="bgr",
                         help="RGB vs BGR color sequence. This is network dependent." )

    ARGS = parser.parse_args()

    camera = cv2.VideoCapture( ARGS.video )

    camera.set( cv2.CAP_PROP_FRAME_WIDTH, 620 )
    camera.set( cv2.CAP_PROP_FRAME_HEIGHT, 480 )

    labels =[ line.rstrip('\n') for line in
              open( ARGS.labels ) if line != 'classes\n']

    print("Hello!\nTopic at ip:"+"\npossible topic: \n")
    for label in labels:
        print(label)


    main()
