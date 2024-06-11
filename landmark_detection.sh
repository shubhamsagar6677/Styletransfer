#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <content_image_path> <style_image_path> <content_pts_path> <style_pts_path>"
    exit 1
fi

# Assigning arguments to variables
content_path='data/content/barack_obama.jpg'
style_path='data/style/Comics/1.png'
content_pts_path='data/content/pts/barack_obama.txt'
style_pts_path='data/style/pts/1.txt'

# Run the Python script
python face_landmark_detection.py ${content_path} ${style_path} ${content_pts_path} ${style_pts_path}