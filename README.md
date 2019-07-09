# Movidius detection server

The project provides classification and detection by using caffe mobilenet, agenet and gendernet models. This program is meant to be deployed on a Raspberry Pi (or other compatible boards) equipped with one or more intel movidius compute sticks and a usb webcam. Thanks to the ncsdk2 it is able to scale on all the sticks (tested up to 2 sticks). The Raspberry Pi works also as a MQTT broker (data are send via MQTT protocol and organized in Topics).

## Installation
It was test only on arm devices (in particular a raspberry pi 3B and raspberry pi 3 B+).

For easly deploy the project just build the docker image with the Dockerfile:

```docker build -t movidius-detection-server .```

## Usage

Just run the docker image in privilege mode:

```docker run --net=host --privileged -v /dev:/dev -t movidius-detection-server```

It will start to detect 15 classes:

```
background, aeroplane, bicycle, bird, boat, bottle, bus, 
car, cat, chair, cow, dining table, dog, horse, motorbike, 
person, potted plant, sheep, sofa, train, tvmonitor
```
It publishes information about the class, the recognition score and the box position plus some additional information in the case a person is detected (like age and gender) is send like a json file in a topic like

```
/data/[class_category]
```

If the user needs to filter only topic with person he/she can subscribe to `/data/person` topic.

## Future Improvement

- For the gender and the age detection I need to get only the face's frame, this is done without the intel compute stick, a future improvement could be convert the model to be load in the compute stick for improve performance. 
- This is just the server side on raspberry, a client side for visualize the data visualization is required.
