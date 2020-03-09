from modules.utils import tensor_to_image, load_img, clip_0_1
from modules.forward import feed_forward

def transfer(image, weights, result):

    # Load content image.
    image = load_img(image, resize=False)
    
    # Build the feed-forward network and load the weights.
    network = feed_forward()
    network.load_weights(weights)

    # Geneerate the style imagee
    image = network(image)

    # Clip pixel values to 0-255
    image = clip_0_1(image)

    # Save the style image
    tensor_to_image(image).save(result)