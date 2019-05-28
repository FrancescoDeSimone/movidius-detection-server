import mvnc.mvncapi as mvnc
import sys
import os

class movidius:
    devices = []
    def __init__(self):
        devices_connected = mvnc.EnumerateDevices();
        if len(devices_connected) == 0:
            print( "No devices found" )
            quit()
        for d in devices_connected:
            devices.append([d,[]])

    def load_graph(graph, devices_number=0):
        with open( graph, mode='rb' ) as f:
            blob = f.read()
        devices[devices_number][1].append(devices[devices_number][0].AllocateGraph(blob))

    def close_all_device():
        for d in devices:
            for graph in d[1]:
                graph.DeallocateGraph()
            d[0].CloseDevice()

    def close_devices(devices_number = 0):
        for graph in devices[devices_number]:
            graph.DeallocateGraph()
        devices[devices_number][0].CloseDevice()



