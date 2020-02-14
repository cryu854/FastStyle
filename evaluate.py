from utils import tensor_to_image, load_img, clip_0_1
from forward import feed_forward

network = feed_forward()
network.load_weights('weights/weights')

content_image = load_img('Source/chicago.jpg')



import time
start = time.time()


image = network(content_image)

end = time.time()
print("Total time: {:.1f}".format(end-start))

image = clip_0_1(image)


file_name = 'Results/1011.jpg'
tensor_to_image(image).save(file_name)