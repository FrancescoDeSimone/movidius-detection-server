from mvnc import mvncapi as mvnc

class movidious:
    def __init__(self):
        self._devices = mvnc.enumerate_devices()
        if len(devices < 1):
            print("No NCS devices found\nInsert at least one")
            return 1
        self._dev_list = list()
        device_number = 0
        for one_device in self._devices:
            try:
                dev = mvnc.Device(one_device)
                dev.open()
                self._dev_list.append(dev)
                device_number += 1
            except:
                print("Error with devices "+str(device_number))
                pass
        if(len(obj_detect_list) < 1)
            print("Could not open any NCS devices!")
            return 1
        print("Using "+str(len(obj_detect_list))+" devices")

    def load_graph(self, processor):
        for d in self._dev:


    
