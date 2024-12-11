import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frame_processors import process_frame
from camera import RealSenseStreamClient

def run_client():
    client = RealSenseStreamClient('10.100.14.69')  # Replace with your server IP
    
    if client.connect():
        client.stream_frames(callback=process_frame, show_preview=True)

if __name__ == "__main__":
    run_client()