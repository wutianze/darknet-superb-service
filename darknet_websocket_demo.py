from ctypes import *
#from multiprocessing import Process, Queue
import queue
import time
from threading import Lock,Thread
from fastapi import FastAPI
from fastapi import Request
from fastapi import WebSocket, WebSocketDisconnect
import uvicorn
#from yolo_service import *
import socket
import random
from typing import List
import darknet
import cv2
import time
import io
import struct
import numpy as np
import base64
import json
from jtracer.tracing import init_tracer
import pynng
from PIL import Image
from opentracing.propagation import Format


def convert2relative(bbox,darknet_height,darknet_width):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox,darknet_height,darknet_width):
    x, y, w, h = convert2relative(bbox,darknet_height,darknet_width)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


class SuperbFrame:
    def __init__(self,darknet_height,darknet_width):
        self.image = None
        self.results = None
        self.darknet_image = darknet.make_image(darknet_width,darknet_height,3)
        self.recv_timestamp = 0
        self.send_timestamp = 0
        self.inference_time = 0
        self.final_image = None
        self.bytes = None
        self.span = None


def port_is_used(port,ip="0.0.0.0"):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        s.connect((ip,port))
        s.shutdown(2)
        return True
    except Exception as e:
        return False

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        # 存放激活的ws连接对象
        self.active_connections: List[WebSocket] = []
        self.ports = set()
        self.port_lock = Lock()

    async def connect(self, ws: WebSocket):
        # 等待连接
        await ws.accept()
        # 存储ws连接对象
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        # 关闭时 移除ws对象
        self.active_connections.remove(ws)

manager = ConnectionManager()
@app.get("/get_port")
def get_port(request:Request):
    while True:
        manager.port_lock.acquire()
        port_tmp = random.randint(10000,20000)
        #port_tmp = 10778
        if port_tmp in manager.ports or port_is_used(port_tmp):
            manager.port_lock.release()
            continue
        else:
            manager.ports.add(port_tmp)
            manager.port_lock.release()
            return port_tmp # port_tmp is the key for a client

def parse_data(data,tracer):
    head_length, msg_length = struct.unpack("ii", data[0:8])
    head_length, msg_length, msg_head, msg = struct.unpack("ii"+ str(head_length) + "s" + str(msg_length) + "s", data)
    
    if head_length > 2: 
        span_dict = json.loads(msg_head)
        span_ctx = tracer.extract(Format.TEXT_MAP, span_dict)
        return span_ctx, msg
    else:
        return None, msg
def send_index(send_queue, sock,keep_alive):
    while keep_alive:
        try:
            span_reply = send_queue.get(block=False,timeout=20)
            sock.send(span_reply)
        except pynng.Timeout:
            print("sock.send timeout")
        except:
            pass # no msg to send


def send_then_recv(input_address,send_queue,input_queue,tracer,darknet_width,darknet_height,sock,keep_alive):
    #sock = pynng.Pair1(recv_timeout=100,send_timeout=100) 
    #sock.listen(input_address)
    while keep_alive:
        try:
            span_reply = send_queue.get(block=False,timeout=20)
            sock.send(span_reply)
        except pynng.Timeout:
            print("sock.send timeout")
        except:
            pass # no msg to send

        try:
            msg = sock.recv()
        except pynng.Timeout:
            continue
        recv_time = time.time()
        newFrame = SuperbFrame(darknet_height,darknet_width)
        newFrame.recv_timestamp = int(recv_time*1000.0) # in ms

        # msg handling
        span_ctx, msg_content = parse_data(msg,tracer)
        if span_ctx is not None:
            newFrame.span = tracer.start_span('image_procss',child_of=span_ctx)
        
        header = msg_content[0:24]
        hh,ww,cc,tt = struct.unpack('iiid',header)
        newFrame.send_timestamp = int(tt*1000.0)
        hh,ww,cc,tt,ss = struct.unpack('iiid'+str(hh*ww*cc)+'s',msg_content)

        newFrame.image = cv2.cvtColor((np.frombuffer(ss,dtype=np.uint8)).reshape(hh,ww,cc), cv2.COLOR_BGR2RGB)
        darknet.copy_image_from_bytes(newFrame.darknet_image,cv2.resize(newFrame.image,(darknet_width,darknet_height),interpolation=cv2.INTER_LINEAR).tobytes())
        #if span_ctx is not None:
        #    newFrame.span.finish()
        try:
            input_queue.put(newFrame,block=False,timeout=100)
        except:
            print("input_queue is full, discard current msg")
            continue

def keep_inference(send_queue,input_queue,result_queue,network,class_names,keep_alive):
    while keep_alive:
        try:
            #print("get newFrame")
            newFrame = input_queue.get(block=False,timeout=100)
        except:
            #print("inference get fail")
            continue

        prev_time = time.time()
        newFrame.results = darknet.detect_image(network, class_names, newFrame.darknet_image, thresh=0.2)
        newFrame.inference_time = int((time.time()-prev_time)*1000.0) # s -> ms
        darknet.free_image(newFrame.darknet_image)
        if newFrame.span is not None:
            index = newFrame.span.get_baggage_item('index')
            newFrame.span.finish()
            try:
                send_queue.put(index.encode())
                sock.send(index.encode())
            except:
                print("send_queue is full, discard current msg")
        try:
            result_queue.put(newFrame,block=False,timeout=10)
        except:
            print("result_queue is full, discard current msg")
            continue

def generate_output(result_queue,need_bytes,keep_alive,class_colors,darknet_height,darknet_width,resizew=960,resizeh=480):
    while keep_alive:
        try:
            newFrame = result_queue.get(block=False,timeout=30)
        except:
            continue
        detections_adjusted = []
        if newFrame is not None:
            for label, confidence, bbox in newFrame.results:
                bbox_adjusted = convert2original(newFrame.image, bbox,darknet_height,darknet_width)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            image = darknet.draw_boxes(detections_adjusted, newFrame.image, class_colors)
            cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            newFrame.final_image = image
            if need_bytes:
                img = Image.fromarray(image).resize((resizew,resizeh))
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                newFrame.bytes = base64.b64encode(img_byte_arr.read()).decode()
            return newFrame
        else:
            continue

@app.websocket("/ws/{port}")# user is the received port_tmp
async def stream_handler(websocket: WebSocket, port: str):
    await manager.connect(websocket)
    network,class_names,class_colors = darknet.load_network(
            "./cfg/yolov4.cfg",
            "./cfg/coco.data",
            "./yolov4.weights",
            batch_size=1
            )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    tracer = init_tracer("image-process")
    input_queue = queue.Queue(maxsize=5)
    result_queue = queue.Queue(maxsize=5)
    send_queue = queue.Queue(maxsize=5)
    input_address = "tcp://0.0.0.0:"+port
    sock = pynng.Pair1(recv_timeout=100,send_timeout=100) 
    sock.listen(input_address)

    keep_alive = True
    p0 = Thread(target=send_then_recv,args=(input_address,send_queue,input_queue,tracer,darknet_width,darknet_height,sock,keep_alive))
    p1 = Thread(target=keep_inference,args=(send_queue,input_queue,result_queue,network,class_names,keep_alive))
    p2 = Thread(target=send_index,args=(send_queue,sock,keep_alive))
    p0.start()
    p1.start()
    p2.start()
    try:
        while keep_alive:
            superbFrame = generate_output(result_queue,True,keep_alive,class_colors,darknet_width,darknet_height)
            send1_time = int(time.time()*1000.0)
            payload = {"img": "data:image/png;base64,%s"%(superbFrame.bytes),"send0_time":superbFrame.send_timestamp,"recv_time":superbFrame.recv_timestamp,"send1_time":send1_time}
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        keep_alive = False
        p0.join()
        p1.join()
        p2.join()
        sock.close()
        manager.disconnect(websocket)
        manager.ports.discard(port)

            
if __name__ == "__main__":
    uvicorn.run("darknet_websocket_demo:app",host="0.0.0.0",port=11935,log_level="info")
