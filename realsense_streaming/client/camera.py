import numpy as np
import cv2
import socket
import struct
import threading
import json
import os
import base64


class RealSenseStreamClient:
    def __init__(self, server_host, port=65432):
        """
        Initialize RealSense Stream Client
        
        :param server_host: IP address of the streaming server
        :param port: Port of the streaming server
        """
        self.server_host = server_host
        self.port = port
        self.client_socket = None
        self.is_connected = False
    
    def connect(self, timeout=10):
        """
        Connect to the RealSense stream server
        
        :param timeout: Connection timeout in seconds
        :return: Connection status
        """
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(timeout)
            self.client_socket.connect((self.server_host, self.port))
            self.is_connected = True
            print(f"Connected to {self.server_host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def receive_frame(self):
        """
        Receive a single frame from the server
        
        :return: Dictionary containing color and optional depth frames
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to server")
        
        try:
            # Receive payload size
            payload_size_bytes = self.client_socket.recv(4)
            payload_size = struct.unpack("!I", payload_size_bytes)[0]
            
            # Receive payload
            payload_data = b''
            while len(payload_data) < payload_size:
                chunk = self.client_socket.recv(payload_size - len(payload_data))
                if not chunk:
                    return None
                payload_data += chunk
            
            # Parse JSON payload
            payload = json.loads(payload_data.decode('utf-8'))
            
            # Decode frames
            result = {}
            
            # Decode color frame
            color_bytes = base64.b64decode(payload['frame'])
            color_array = np.frombuffer(color_bytes, dtype=np.uint8)
            result['color_frame'] = cv2.imdecode(color_array, cv2.IMREAD_COLOR)
            
            # Decode depth frame if available
            if 'depth' in payload:
                depth_bytes = base64.b64decode(payload['depth'])
                result['depth_frame'] = np.frombuffer(depth_bytes, dtype=np.uint16)
            
            return result
        
        except Exception as e:
            print(f"Frame receive error: {e}")
            return None
    
    def stream_frames(self, callback=None, show_preview=True):
        """
        Continuously receive and process frames
        
        :param callback: Optional function to process each frame
        :param show_preview: Whether to show a preview window
        """
        try:
            while True:
                frame_data = self.receive_frame()
                if frame_data is None:
                    break
                
                color_frame = frame_data['color_frame']
                
                # Optional callback for custom processing
                if callback:
                    try:
                        callback(frame_data)
                    except Exception as e:
                        print(f"Callback error: {e}")
                
                # Show preview if enabled
                if show_preview:
                    cv2.imshow('RealSense Stream', color_frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cv2.destroyAllWindows()
            self.client_socket.close()
            self.is_connected = False
