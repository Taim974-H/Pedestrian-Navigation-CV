def process_frame(frame_data):
    """
    Custom frame processing callback
    
    :param frame_data: Dictionary containing color and optional depth frames
    """
    color_frame = frame_data['color_frame']
    
    # Example processing:
    print(f"Frame received. Shape: {color_frame.shape}")
    
    # Add your custom processing logic here
    # For example:
    # - Object detection
    # - Color analysis
    # - Motion tracking
    # etc.