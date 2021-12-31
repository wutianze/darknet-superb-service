from ctypes import *
from multiprocessing import Process
import time
from threading import Lock
import time
from fastapi import FastAPI
from fastapi import Request
from fastapi import WebSocket, WebSocketDisconnect
import uvicorn
from yolo_service import *
import socket
import random
from typing import List


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
        port_lock.acquire()
        port_tmp = random.randint(10000,20000)
        if port_tmp in manager.ports or port_is_used(port_tmp):
            port_lock.release()
            continue
        else:
            manager.ports.add(port_tmp)
            return port_tmp # port_tmp is the key for a client

@app.websocket("/ws/{port}")# user is the received port_tmp
async def stream_handler(websocket: WebSocket, port: str):
    await manager.connect(websocket)
    ds = DarknetService()
    p0 = Process(target=ds.get_image_online,args=("tcp://0.0.0.0:"+port))
    p1 = Process(target=ds.keep_inference)
    p0.start()
    p1.start()
    try:
        while ds.keep_alive:
            superbFrame = ds.generate_output(True)
            send1_time = int(time.time()*1000.0)
            payload = {"img": "data:image/png;base64,%s"%(superbFrame.bytes),"send0_time":superbFrame.send_timestamp,"recv_time":superbFrame.recv_timestamp,"send1_time":send1_time}
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        ds.keep_alive = False
        p0.join()
        p1.join()
        manager.disconnect(websocket)
        manager.ports.discard(port)

            
if __name__ == "__main__":
        port_lock = Lock()
        uvicorn.run("darknet_websocket_demo:app",host="0.0.0.0",port=1935,log_level="info")