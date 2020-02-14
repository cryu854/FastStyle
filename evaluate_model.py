# Only for size = 256x256

import tensorflow as tf
from utils import tensor_to_image, load_img, clip_0_1

network = tf.keras.models.load_model('model')

content_image = load_img('Source\\chicago.jpg', resize=True)



import time
start = time.time()


image = network(content_image)

end = time.time()
print("Total time: {:.1f}".format(end-start))

image = clip_0_1(image)


file_name = 'Results\\512.jpg'
tensor_to_image(image).save(file_name)