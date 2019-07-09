# Movidius detection server

Classification and detection software made using caffe mobilenet, agenet and gendernet models.
This program is meant to be deployed on a raspberry pi (or other compatible board) with one or more intel movidius compute stick and a usb webcam. Thanks to the ncsdk2 it should be able to scale on all the sticks (tested upper 2 sticks).
The raspberry pi work also as a broker, the data will send into topics with mqtt protocol. 

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
The class, score and box position plus some additional information if a person is detected (like age and gender) is send like a json file in a topic like

```
/data/[class_category]
```

So if we want filter only topic with person we subscribe to `/data/person` topic.

## Future Improvement

- For the gender and the age detection I need to get only the face's frame, this is done without the intel compute stick, a future improvement could be convert the model to be load in the compute stick for improve performance. 
- This is just the server side on raspberry, a client side for visualize the data will be nice
