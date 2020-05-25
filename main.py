"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2


import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from collections import deque
from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def model_out(frame, result):
    """
    Parse model output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse model
    :return: person count and frame
    """
    current_count = 0
    for obj in result[0][0]:
        #if obj['class_id'] == 1:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        #print(obj)
            if obj[2] > prob_threshold:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
                current_count = current_count + 1
    return frame, current_count


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    #f = open("inference.txt","x")
    # Flag for the input image
    track_len = 12 # for ssd_mobilenet_v2_coco
    #track_len = 3 # for person-detection-retail-0013
    buffer = deque(maxlen=track_len)
    
    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    #count = 0
    #previous_frame = False
    #previous_previous_frame = False
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    global initial_w, initial_h, prob_threshold
    
    prob_threshold = args.prob_threshold
    
    start_time = time.time()

    ### Load the model through `infer_network` ###
    n, c, h, w = infer_network.load_model(args.model, args.device, 
                                          1, 1, cur_request_id,
                                          args.cpu_extension)[1]
    load_time = time.time() - start_time
    #f = open("mn_v1_inference.txt", "a")  # append mode 
    #f = open("mn_v2_inference.txt", "a")  # append mode 
    #f = open("i_v2_inference.txt", "a")  # append mode 
    f = open("inference.txt","a")
    
    f.write("Load time: {}".format(str(load_time)))
    
    


    ### Handle the input stream ###
    # Checks for camera (live) feed
    single_image_mode = False
    if args.input == 'CAM':
        input_stream = 0
    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_stream = args.input
    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    
    if input_stream:
        cap.open(args.input)
        
    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
    
    
    initial_w = cap.get(3)
    initial_h = cap.get(4)
    ### Loop until stream is over ###
    start_time = time.time()
    while cap.isOpened():

        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        
        key_pressed = cv2.waitKey(60)

        ### Pre-process the image as needed ###
        image = cv2.resize(frame, (w,h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))

        ### Start asynchronous inference for specified request ###
        inf_start = time.time()
        infer_network.exec_net(cur_request_id, image)

        ### Wait for the result ###
        #if infer_network.wait() == 0:
        if infer_network.wait(cur_request_id) == 0:
            det_time = time.time() - inf_start
            ### Get the results of the inference request ###
            result = infer_network.get_output(cur_request_id)
            ### Extract any desired stats from the results ###
            frame, current_count = model_out(frame, result)
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)            
            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            buffer.append(current_count)
            if np.sum(buffer) > 0:
                current_count = 1
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))
                
            if current_count < last_count:
                duration = int(time.time() - start_time - det_time*track_len)
                client.publish("person/duration", json.dumps({"duration": duration}))
                
            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count
            
            if key_pressed == 27:
                break
                
                

        ### end the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### Write an output image if `single_image_mode` ###
        #infer_time = time.time() - start_time
        
        f.write('\n')
        f.write(str(det_time * 1000))
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
            
        #cv2.imwrite("mobilenet_v1/frame%d.jpg" % count,frame)
        #cv2.imwrite("mobilenet_v2/frame%d.jpg" % count,frame)
        #cv2.imwrite("inception_v2/frame%d.jpg" % count,frame)
        #print(count)
        #count+=1
        
        
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()
    f.close()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream

    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
