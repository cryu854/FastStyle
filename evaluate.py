from modules.utils import tensor_to_image, load_img, clip_0_1
from modules.forward import feed_forward

def transfer(image, weights, result):

    image = load_img(image, resize=False)
    
    network = feed_forward()
    network.load_weights(weights)

    import time
    start = time.time()

    image = network(image)

    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    image = clip_0_1(image)

    tensor_to_image(image).save(result)