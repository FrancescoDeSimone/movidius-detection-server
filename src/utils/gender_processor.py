#! /usr/bin/env python3
from mvnc import mvncapi as mvnc
import numpy as numpy
import cv2
import time
import threading


class GenderNetProcessor:

    def __init__(self, network_graph_filename: str, ncs_device: mvnc.Device,
                 name = None):
        self._device = ncs_device
        self._network_graph_filename = network_graph_filename
        try:
            with open(self._network_graph_filename, mode='rb') as graph_file:
                graph_in_memory = graph_file.read()
            self._graph = mvnc.Graph("GenderNet Graph")
            self._fifo_in, self._fifo_out = self._graph.allocate_with_fifos(self._device, graph_in_memory)

            self._input_fifo_capacity = self._fifo_in.get_option(mvnc.FifoOption.RO_CAPACITY)
            self._output_fifo_capacity = self._fifo_out.get_option(mvnc.FifoOption.RO_CAPACITY)

        except:
            print('\n\n')
            print('Error - could not load neural network graph file: ' + network_graph_filename)
            print('\n\n')
            raise

        self._end_flag = True
        self._name = name
        if (self._name is None):
            self._name = "no name"

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

    def start_aysnc_inference(self, input_image:numpy.ndarray,bound):
        inference_image, frame = self._pre_process_image(input_image,bound) 

        self._inc_async_count()

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
        output, _ = self._fifo_out.read_elem()
        return self._filter_objects(output)


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


    def _filter_objects(self, inference_result:numpy.ndarray):
        top_prediction = inference_result.argmax()
        label = ["male","female"]
        return (100.0 * inference_result[top_prediction], label[top_prediction] )

    def _pre_process_image(self,frame,bound):
        height, width, channels = frame.shape
        x1,y1,x2,y2 = bound
        img = frame[ y1 : y2, x1 : x2 ]
        img = cv2.resize( img, tuple( [227, 227] ) )
        img = ( img - [78.42633776, 87.76891437, 114.89584775] )
        return img, frame


