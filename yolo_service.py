import darknet
import cv2
import time
import struct
import sys
import base64
import json
from jtracer.tracing import init_tracer
from queue import Queue
import pynng
from PIL import Image
from opentracing.propagation import Format

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
        self.final_image = None
        self.bytes = None
        self.span = None

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
        self.tracer = init_tracer("image-process")

        self.keep_alive = True
        self.sock = None
    
    def _parse_data(self,data):
        head_length, msg_length = struct.unpack("ii", data[0:8])
        head_length, msg_length, msg_head, msg = struct.unpack("ii"+ str(head_length) + "s" + str(msg_length) + "s", data)
        
        if head_length > 2: 
            span_dict = json.loads(msg_head)
            span_ctx = self.tracer.extract(Format.TEXT_MAP, span_dict)
            return span_ctx, msg
        else:
            return None, msg
            
    def get_image_online(self,input_address):
        self.sock = pynng.Pair1(recv_timeout=100,send_timeout=100) 
        self.sock.listen(input_address)
        while self.keep_alive:
            try:
                msg = self.sock.recv()
            except pynng.Timeout:
                continue
            recv_time = time.time()
            #print("get one image")
            newFrame = SuperbFrame()
            newFrame.recv_timestamp = int(recv_time*1000.0) # in ms

            # msg handling
            span_ctx, msg_content = self._parse_data(msg)
            if span_ctx is not None:
                newFrame.span = self.tracer.start_span('image_procss',child_of=span_ctx)
            header = msg_content[0:24]
            hh,ww,cc,tt = struct.unpack('iiid',header)
            newFrame.send_timestamp = int(tt*1000.0)
            hh,ww,cc,tt,ss = struct.unpack('iiid'+str(hh*ww*cc)+'s',msg_content)

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
            darknet.copy_image_from_bytes(newFrame.darknet_image,cv2.resize(image,(DarknetService.darknet_width,DarknetService.darknet_height),interpolation=cv2.INTER_LINEAR).tobytes())
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


    def keep_inference(self):
        while self.keep_alive:
            try:
                newFrame = self.input_queue.get(block=False,timeout=1)
            except:
                print("input_queue empty")
                continue
            prev_time = time.time()
            newFrame.results = darknet.detect_image(DarknetService.network, DarknetService.class_names, newFrame.darknet_image, thresh=0.2)
            newFrame.inference_time = int((time.time()-prev_time)*1000.0) # s -> ms
            if newFrame.span is not None:
                index = newFrame.span.get_baggage_item('index')
                try:
                    self.sock.send(index.encode())
                except pynng.Timeout:
                    print("Error: span reply fail")
            darknet.free_image(newFrame.darknet_image)
            try:
                self.result_queue.put(newFrame,block=False,timeout=1)
            except:
                print("result_queue is full, discard current msg")
                continue

    def generate_output(self,need_bytes,resizew=960,resizeh=480):
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
                cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                newFrame.final_image = image
                if need_bytes:
                    Image.fromarray(image).resize((resizew,resizeh))
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    newFrame.bytes = base64.b64encode(img_byte_arr.read()).decode()
                return newFrame
            else:
                continue

if __name__ == "__main__":
    ds = DarknetService()
    ds.get_image_from_file(str(sys.argv[1]))
    ds.inference()
    final_image = ds.generate_output(False)
    cv2.imwrite("dr.png",final_image.final_image)
