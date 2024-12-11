import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from camera import RealSenseStreamServer

def run_server():
    server = RealSenseStreamServer()
    server.start_streaming()
    
    try:
        input("Press Enter to stop streaming...")
    finally:
        server.stop_streaming()

if __name__ == "__main__":
    run_server()