import signal
import time

class clean_close:
    close = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit)
        signal.signal(signal.SIGTERM, self.exit)

    def exit(self,signum, frame):
        self.close = True
