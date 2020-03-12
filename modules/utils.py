import tensorflow as tf
import numpy as np
import PIL.Image
import os

def tensor_to_image(tensor):
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
    
def load_img(path_to_img, max_dim=None, resize=True):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    if resize:
        new_shape = tf.cast([256, 256], tf.int32)
        img = tf.image.resize(img, new_shape)

    if max_dim:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        
    img = img[tf.newaxis, :]

    return img

def load_img_path(path_to_img, batch_size):
    content_targets = [os.path.join(path_to_img, fname) for fname in os.listdir(path_to_img)]
    mod = len(content_targets) % batch_size
    if mod > 0:
        print('Train set has been trimmed slightly..')
        content_targets = content_targets[:-mod]

    return content_targets

def create_folder(diirname):
    if not os.path.exists(diirname):
        os.mkdir(diirname)
        print('Directory ', diirname, ' createrd')
    else:
        print('Directory ', diirname, ' already exists')       

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)