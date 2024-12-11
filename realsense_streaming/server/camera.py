import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import struct
import threading
import json
import os
import base64

class RealSenseStreamServer:
    def __init__(self, host='0.0.0.0', port=65432, config_path='stream_config.json'):
        """
        Initialize RealSense Streaming Server
        
        :param host: IP address to bind the server (default: all interfaces)
        :param port: Port to listen on (default: 65432)
        :param config_path: Path to configuration file
        """
        # Load configuration
        self.load_config(config_path)
        
        # RealSense camera setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure RGB stream
        self.config.enable_stream(
            rs.stream.color, 
            self.stream_width, 
            self.stream_height, 
            rs.format.bgr8, 
            self.stream_fps
        )
        
        # Optional: Configure depth stream if needed
        if self.enable_depth:
            self.config.enable_stream(
                rs.stream.depth, 
                self.stream_width, 
                self.stream_height, 
                rs.format.z16, 
                self.stream_fps
            )
        
        # Networking setup
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []
        
        # Streaming control
        self.is_streaming = False
        
        # Logging
        self.logger = self._setup_logger()
    
    def load_config(self, config_path):
        """
        Load streaming configuration from JSON file
        
        :param config_path: Path to configuration file
        """
        # Default configuration
        default_config = {
            "stream_width": 640,
            "stream_height": 480,
            "stream_fps": 30,
            "enable_depth": False,
            "compression_quality": 80,
            "max_clients": 5
        }
        
        # Try to load from file, use defaults if not found
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                config = {**default_config, **user_config}
        except FileNotFoundError:
            # Create default config file if not exists
            config = default_config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        
        # Set configuration attributes
        self.stream_width = config['stream_width']
        self.stream_height = config['stream_height']
        self.stream_fps = config['stream_fps']
        self.enable_depth = config['enable_depth']
        self.compression_quality = config['compression_quality']
        self.max_clients = config['max_clients']
    
    def _setup_logger(self):
        """
        Setup basic logging
        
        :return: Logging function
        """
        def logger(message, level='INFO'):
            print(f"[{level}] {message}")
        return logger
    
    def start_camera(self):
        """Start the RealSense camera pipeline"""
        try:
            self.pipeline.start(self.config)
            self.logger("Camera started successfully")
        except Exception as e:
            self.logger(f"Failed to start camera: {e}", 'ERROR')
            raise
    
    def stop_camera(self):
        """Stop the RealSense camera pipeline"""
        try:
            self.pipeline.stop()
            self.logger("Camera stopped")
        except Exception as e:
            self.logger(f"Error stopping camera: {e}", 'WARNING')
    
    def setup_server(self):
        """Set up TCP server for streaming"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.max_clients)
            self.logger(f"Server listening on {self.host}:{self.port}")
        except Exception as e:
            self.logger(f"Failed to setup server: {e}", 'ERROR')
            raise
    
    def handle_client(self, client_socket):
        """
        Handle individual client connection
        
        :param client_socket: Socket for the connected client
        """
        try:
            while self.is_streaming:
                # Capture frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to numpy array
                frame = np.asanyarray(color_frame.get_data())
                
                # Optional depth frame
                depth_frame = None
                if self.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_frame = np.asanyarray(depth_frame.get_data())
                
                # Encode frame with compression
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.compression_quality])
                frame_bytes = buffer.tobytes()
                
                # Prepare payload (with optional depth)
                payload = {
                    'frame': base64.b64encode(frame_bytes).decode('utf-8')
                }
                if depth_frame is not None:
                    depth_bytes = depth_frame.tobytes()
                    payload['depth'] = base64.b64encode(depth_bytes).decode('utf-8')
                
                # Convert payload to JSON
                json_payload = json.dumps(payload).encode('utf-8')
                
                # Send payload size and then payload
                client_socket.sendall(struct.pack("!I", len(json_payload)) + json_payload)
        
        except Exception as e:
            self.logger(f"Client handling error: {e}", 'ERROR')
        finally:
            client_socket.close()
            if client_socket in self.clients:
                self.clients.remove(client_socket)
    
    def accept_clients(self):
        """Accept incoming client connections"""
        while self.is_streaming:
            try:
                client_socket, addr = self.server_socket.accept()
                self.logger(f"Connection from {addr}")
                
                # Check client limit
                if len(self.clients) >= self.max_clients:
                    self.logger("Max clients reached. Rejecting connection.", 'WARNING')
                    client_socket.close()
                    continue
                
                self.clients.append(client_socket)
                
                # Start a thread to handle this client
                client_thread = threading.Thread(
                    target=self.handle_client, 
                    args=(client_socket,), 
                    daemon=True
                )
                client_thread.start()
            
            except Exception as e:
                if self.is_streaming:
                    self.logger(f"Error accepting client: {e}", 'ERROR')
    
    def start_streaming(self):
        """Start camera and streaming server"""
        try:
            # Start camera
            self.start_camera()
            
            # Setup server
            self.setup_server()
            
            # Set streaming flag
            self.is_streaming = True
            
            # Start accepting clients in a separate thread
            accept_thread = threading.Thread(
                target=self.accept_clients, 
                daemon=True
            )
            accept_thread.start()
            
            self.logger("Streaming started successfully")
        
        except Exception as e:
            self.logger(f"Failed to start streaming: {e}", 'ERROR')
            self.stop_streaming()
    
    def stop_streaming(self):
        """Stop streaming and close connections"""
        self.is_streaming = False
        
        # Close all client connections
        for client in self.clients:
            client.close()
        
        # Clear client list
        self.clients.clear()
        
        # Stop camera
        self.stop_camera()
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
        
        self.logger("Streaming stopped")
