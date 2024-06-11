#!/bin/bash

# Check if correct number of arguments are provided
#pts_path='data/style/pts/1.txt'
#image_path='data/style/Comics/1.png'


pts_path='data/content/pts/barack_obama.txt'
image_path='data/content/barack_obama.jpg'

# Run the Python script with provided arguments
python show_landmarks.py ${pts_path} ${image_path}
