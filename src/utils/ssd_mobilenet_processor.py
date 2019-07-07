#! /usr/bin/env python3
from mvnc import mvncapi as mvnc
import numpy as numpy
import cv2
import time
import threading


class SsdMobileNetProcessor:

    # Neural network assumes input images are these dimensions.
    SSDMN_NETWORK_IMAGE_WIDTH = 300
    SSDMN_NETWORK_IMAGE_HEIGHT = 300

    def __init__(self, network_graph_filename: str, ncs_device: mvnc.Device,
                 inital_box_prob_thresh: float, classification_mask:list=None,
                 name = None):
        self._device = ncs_device
        self._network_graph_filename = network_graph_filename
        # Load graph from disk and allocate graph.
        try:
            with open(self._network_graph_filename, mode='rb') as graph_file:
                graph_in_memory = graph_file.read()
            self._graph = mvnc.Graph("SSD MobileNet Graph")
            self._fifo_in, self._fifo_out = self._graph.allocate_with_fifos(self._device, graph_in_memory)

            self._input_fifo_capacity = self._fifo_in.get_option(mvnc.FifoOption.RO_CAPACITY)
            self._output_fifo_capacity = self._fifo_out.get_option(mvnc.FifoOption.RO_CAPACITY)

        except:
            print('\n\n')
            print('Error - could not load neural network graph file: ' + network_graph_filename)
            print('\n\n')
            raise

        self._classification_labels=SsdMobileNetProcessor.get_classification_labels()

        self._box_probability_threshold = inital_box_prob_thresh
        self._classification_mask=classification_mask
        if (self._classification_mask is None):
            # if no mask passed then create one to accept all classifications
            self._classification_mask = [1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1]

        self._end_flag = True
        self._name = name
        if (self._name is None):
            self._name = "no name"

        # lock to let us count calls to asynchronus inferences and results
        self._async_count_lock = threading.Lock()
        self._async_inference_count = 0

    def cleanup(self, destroy_device=False):
        self._drain_queues()
        self._fifo_in.destroy()
        self._fifo_out.destroy()
        self._graph.destroy()

        if (destroy_device):
            self._device.close()
            self._device.destroy()


    def get_device(self):
       return self._device

    def get_name(self):
        return self._name

    def drain_queues(self):
       self._drain_queues()

    @staticmethod
    def get_classification_labels():
        ret_labels = list(['background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'dining table', 'dog', 'horse',
          'motorbike', 'person', 'potted plant',
          'sheep', 'sofa', 'train', 'tvmonitor'])
        return ret_labels


    def start_aysnc_inference(self, input_image:numpy.ndarray):
        inference_image = cv2.resize(input_image,
                                 (SsdMobileNetProcessor.SSDMN_NETWORK_IMAGE_WIDTH,
                                  SsdMobileNetProcessor.SSDMN_NETWORK_IMAGE_HEIGHT),
                                 cv2.INTER_LINEAR)

        # modify inference_image for network input
        inference_image = inference_image - 127.5
        inference_image = inference_image * 0.007843

        self._inc_async_count()

        # Load tensor and get result.  This executes the inference on the NCS
        self._graph.queue_inference_with_fifo_elem(self._fifo_in, self._fifo_out, inference_image.astype(numpy.float32), input_image)

        return

    def _inc_async_count(self):
        self._async_count_lock.acquire()
        self._async_inference_count += 1
        self._async_count_lock.release()

    def _dec_async_count(self):
        self._async_count_lock.acquire()
        self._async_inference_count -= 1
        self._async_count_lock.release()

    def _get_async_count(self):
        self._async_count_lock.acquire()
        ret_val = self._async_inference_count
        self._async_count_lock.release()
        return ret_val


    def get_async_inference_result(self):
        self._dec_async_count()

        # get the result from the queue
        output, input_image = self._fifo_out.read_elem()

        # save original width and height
        input_image_width = input_image.shape[1]
        input_image_height = input_image.shape[0]

        # filter out all the objects/boxes that don't meet thresholds
        return self._filter_objects(output, input_image_width, input_image_height), input_image


    def is_input_queue_empty(self):
        count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        return (count == 0)


    def is_input_queue_full(self):
        count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        return ((self._input_fifo_capacity - count) == 0)


    def _drain_queues(self):
        in_count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        out_count = self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)
        count = 0

        while (self._get_async_count() != 0):
            count += 1
            if (out_count > 0):
                self.get_async_inference_result()
                out_count = self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)
            else:
                time.sleep(0.1)

            in_count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
            out_count = self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)
            if (count > 3):
                blank_image = numpy.zeros((self.SSDMN_NETWORK_IMAGE_HEIGHT, self.SSDMN_NETWORK_IMAGE_WIDTH, 3),
                                          numpy.float32)
                self.do_sync_inference(blank_image)

            if (count == 30):
                # should really not be nearly this high of a number but working around an issue in the
                # ncapi where an inferece can get stuck in process
                raise Exception("Could not drain FIFO queues for '" + self._name + "'")

        in_count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        out_count = self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)
        return


    def do_sync_inference(self, input_image:numpy.ndarray):
        self.start_aysnc_inference(input_image)
        filtered_objects, original_image = self.get_async_inference_result()

        return filtered_objects


    def get_box_probability_threshold(self):
        return self._box_probability_threshold


    def set_box_probability_threshold(self, value):
        self._box_probability_threshold = value


    def _filter_objects(self, inference_result:numpy.ndarray, input_image_width:int, input_image_height:int):
        num_valid_boxes = int(inference_result[0])

        classes_boxes_and_probs = []
        for box_index in range(num_valid_boxes):
                base_index = 7+ box_index * 7
                if (inference_result[base_index + 2] < self._box_probability_threshold):
                    # probability/confidence is too low for this box, omit it.
                    continue
                if (self._classification_mask[int(inference_result[base_index + 1])] != 1):
                    # masking off these types of objects
                    continue

                if (not numpy.isfinite(inference_result[base_index]) or
                        not numpy.isfinite(inference_result[base_index + 1]) or
                        not numpy.isfinite(inference_result[base_index + 2]) or
                        not numpy.isfinite(inference_result[base_index + 3]) or
                        not numpy.isfinite(inference_result[base_index + 4]) or
                        not numpy.isfinite(inference_result[base_index + 5]) or
                        not numpy.isfinite(inference_result[base_index + 6])):
                    # boxes with non finite (inf, nan, etc) numbers must be ignored
                    continue

                x1 = max(int(inference_result[base_index + 3] * input_image_width), 0)
                y1 = max(int(inference_result[base_index + 4] * input_image_height), 0)
                x2 = min(int(inference_result[base_index + 5] * input_image_width), input_image_width-1)
                y2 = min(int(inference_result[base_index + 6] * input_image_height), input_image_height-1)

                classes_boxes_and_probs.append([self._classification_labels[int(inference_result[base_index + 1])], # label
                                                x1, y1, # upper left in source image
                                                x2, y2, # lower right in source image
                                                inference_result[base_index + 2] # confidence
                                                ])
        return classes_boxes_and_probs





