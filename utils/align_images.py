import os
import bz2
from keras.utils import get_file
from utils.face_alignment import image_align
from utils.landmarks_detector import LandmarksDetector

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    """

    # Direct variable assignments instead of command line arguments
    RAW_IMAGES_DIR = '../data/content/raw_images'
    ALIGNED_IMAGES_DIR = '../data/content/allign_images'
    output_size = 1024
    x_scale = 1
    y_scale = 1
    em_scale = 0.1
    use_alpha = False

    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in os.listdir(RAW_IMAGES_DIR):
        print('Aligning %s ...' % img_name)
        try:
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            fn = face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], 1)
            if os.path.isfile(fn):
                continue
            print('Getting landmarks...')
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                try:
                    print('Starting face alignment...')
                    face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
                    print("hello1")
                    image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=output_size,
                                x_scale=x_scale, y_scale=y_scale, em_scale=em_scale, alpha=use_alpha)
                    print('Wrote result %s' % aligned_face_path)
                except:
                    print("Exception in face alignment!")
        except:
            print("Exception in landmark detection!")
