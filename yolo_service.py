import darknet
import cv2
import asyncio
import time
import struct
import sys
from queue import Queue
from PIL import Image

def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = DarknetService.darknet_height
    _width      = DarknetService.darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


class SuperbFrame:
    def __init__(self):
        self.image = None
        self.darknet_image = darknet.make_image(DarknetService.darknet_width,DarknetService.darknet_height,3)
        self.results = None
        self.recv_timestamp = 0
        self.send_timestamp = 0
        self.inference_time = 0

class DarknetService:
    network,class_names,class_colors = darknet.load_network(
            "./cfg/yolov4.cfg",
            "./cfg/coco.data",
            "./yolov4.weights",
            batch_size=1
            )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    def __init__(self):
        self.input_queue = Queue(maxsize=3)
        self.result_queue = Queue(maxsize=3)

        self.keep_alive = True
        
    async def get_image_online(self,input_address):
        with pynng.Pair1(recv_timeout=100,send_timeout=100) as sock:
            sock.listen(input_address)
            while self.keep_alive:
                try:
                    msg = sock.recv()
                except pynng.Timeout:
                    continue
                recv_time = time.time()
                #print("get one image")
                newFrame = SuperbFrame()
                newFrame.recv_timestamp = int(recv_time*1000.0) # in ms

                # msg handling
                header = msg[0:24]
                hh,ww,cc,tt = struct.unpack('iiid',header)
                newFrame.send_timestamp = int(tt*1000.0)
                hh,ww,cc,tt,ss = struct.unpack('iiid'+str(hh*ww*cc)+'s',msg)

                newFrame.image = cv2.cvtColor((np.frombuffer(ss,dtype=np.uint8)).reshape(hh,ww,cc), cv2.COLOR_BGR2RGB)
                darknet.copy_image_from_bytes(newFrame.darknet_image,cv2.resize(newFrame.image,(DarknetService.darknet_width,DarknetService.darknet_height),interpolation=cv2.INTER_LINEAR).tobytes())
                try:
                    self.input_queue.put(newFrame,block=False,timeout=1)
                except:
                    print("input_queue is full, discard current msg")
                    continue

    def get_image_from_file(self,file_address):
        if self.keep_alive:
            image = cv2.cvtColor(cv2.imread(file_address),cv2.COLOR_BGR2RGB)
            recv_time = time.time()
            #print("get one image")
            newFrame = SuperbFrame()
            newFrame.image = image
            newFrame.recv_timestamp = 0 # fill sth
            newFrame.send_timestamp = 0 # fill sth
            
            self.input_queue.put(newFrame)
            return

    def inference(self):
            newFrame = self.input_queue.get()
            prev_time = time.time()
            newFrame.results = darknet.detect_image(DarknetService.network, DarknetService.class_names, newFrame.darknet_image, thresh=0.2)
            newFrame.inference_time = int((time.time()-prev_time)*1000.0) # s -> ms
            darknet.free_image(newFrame.darknet_image)
            self.result_queue.put(newFrame)
            return


    async def keep_inference(self):
        while self.keep_alive:
            try:
                newFrame = self.input_queue.get(block=False,timeout=1)
            except:
                print("input_queue empty")
                continue
            prev_time = time.time()
            newFrame.results = darknet.detect_image(DarknetService.network, DarknetService.class_names, newFrame.image, thresh=0.2)
            newFrame.inference_time = int((time.time()-prev_time)*1000.0) # s -> ms
            try:
                self.result_queue.put(newFrame,block=False,timeout=1)
            except:
                print("result_queue is full, discard current msg")
                continue

    def generate_output(self):
        while self.keep_alive:
            try:
                newFrame = self.result_queue.get(block=False,timeout=1)
            except:
                continue
            detections_adjusted = []
            if newFrame is not None:
                for label, confidence, bbox in newFrame.results:
                    bbox_adjusted = convert2original(newFrame.image, bbox)
                    detections_adjusted.append((str(label), confidence, bbox_adjusted))
                image = darknet.draw_boxes(detections_adjusted, newFrame.image, DarknetService.class_colors)
                return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    ds = DarknetService()
    ds.get_image_from_file(str(sys.argv[1]))
    ds.inference()
    final_image = ds.generate_output()
    cv2.imwrite("dr.png",final_image)
    #cv2.imshow("Image",final_image)
    #cv2.waitKey (0)  
    #cv2.destroyAllWindows()
