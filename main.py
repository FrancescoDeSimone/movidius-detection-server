from utils import movidius

sticks = movidius.movidius()

sticks.load_graph("./caffe/SSD_MobileNet/graph")

sticks.close_all_device();
