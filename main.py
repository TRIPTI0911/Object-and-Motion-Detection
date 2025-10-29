from fastapi import FastAPI
from threading import Thread
import uvicorn
import motion_detector
import time
import os
port = int(os.environ.get("PORT", 10000))


app = FastAPI(title="Motion Detector API")

detector_thread = None
is_running = False

def run_motion_detector():
    """Runs your motion detection script."""
    global is_running
    is_running = True
    motion_detector.run_motion_detector()
    is_running = False

@app.get("/")
def root():
    return {"message": "Motion Detection API is running"}

@app.post("/start")
def start_detection():
    global detector_thread, is_running
    if is_running:
        return {"status": "already running"}
    detector_thread = Thread(target=run_motion_detector, daemon=True)
    detector_thread.start()
    time.sleep(1)
    return {"status": "started"}

@app.post("/stop")
def stop_detection():
    global is_running
    if not is_running:
        return {"status": "not running"}
    motion_detector.stop_motion_detector()
    is_running = False
    return {"status": "stopped"}

@app.get("/status")
def status():
    return {"running": is_running}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
