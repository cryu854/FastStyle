import tensorflow as tf
import numpy as np
import os

from utils import tensor_to_image, load_img, load_img_path, create_folder
from vgg19 import preprocess_input, VGG19
from forward import feed_forward



def vgg_layers(layer_names):
    vgg = VGG19(include_top = False, weights = 'imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
    return model

def gram_matrix(features, normalize = True):
    batch_size , height, width, filters = features.shape
    features = tf.reshape(features, (batch_size, height*width, filters))

    tran_f = tf.transpose(features, perm=[0,2,1])
    gram = tf.matmul(tran_f, features)
    if normalize:
        gram /= tf.cast(height*width*filters, tf.float32)

    return gram

def style_loss(style_outputs, style_target):
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_target[name])**2)
                        for name in style_outputs.keys()])

    return style_loss

def content_loss(content_outputs, content_target):
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_target[name])**2)
                            for name in content_outputs.keys()])

    return content_loss

def total_variation_loss(img):
    x_var = img[:,:,1:,:] - img[:,:,:-1,:]
    y_var = img[:,1:,:,:] - img[:,:-1,:,:]

    return tf.reduce_mean(tf.square(x_var)) + tf.reduce_mean(tf.square(y_var))

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        preprocessed_input = preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        # Compute the gram_matrix
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        # Features that extracted by VGG
        style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
        content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}

        return {'content':content_dict, 'style':style_dict}




def trainer(style_file, train_path, content_weight, style_weight,
                   tv_weight, batch_size, epoch):
    """
    Inputs:
    - style_file: filename of style image
    - train_path: directory of coco dataset image
    - content_weight: weighting on content loss
    - style_weight: weighting on style loss
    - tv_weight: weight of total variation regularization term
    - batch_size: number of content images in each batch
    - epoch: maximum number of training epochs
    """
    # Create a folder for weights
    create_folder('weights')

    # Setup the given layers
    content_layers = ['block4_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

 
    # Build Feed-forward transformer
    network = feed_forward()

    # Build VGG-19 Loss network
    extractor = StyleContentModel(style_layers, content_layers)

    # Load style target image
    style_image = load_img(style_file, resize=True)

    # Load content target images
    batch_shape = (batch_size, 256, 256, 3)
    content_targets = load_img_path(train_path, batch_size)
    X_batch = np.zeros(batch_shape, dtype=np.float32)

    # Extract style target 
    style_target = extractor(style_image*255.0)['style']

    # Build optimizer
    loss_metric = tf.keras.metrics.Mean()
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)


    # tf.function decorator
    @tf.function()
    def train_step(X_batch):
        with tf.GradientTape() as tape:
            content_target = extractor(X_batch*255.0)['content']
            image = network(X_batch)
            outputs = extractor(image)
            
            s_loss = style_weight * style_loss(outputs['style'], style_target)
            c_loss = content_weight * content_loss(outputs['content'], content_target)
            t_loss = tv_weight * total_variation_loss(image)
            loss = s_loss + c_loss + t_loss

        grad = tape.gradient(loss, network.trainable_variables)
        opt.apply_gradients(zip(grad, network.trainable_variables))
        loss_metric(loss)




    def load_img1(path_to_img, resize=True):
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        
        if resize:
            new_shape = tf.cast([256, 256], tf.int32)
            img = tf.image.resize(img, new_shape)

        img = img[tf.newaxis, :]

        return img



    import time
    start = time.time()

    train_dataset = tf.data.Dataset.list_files('dataset/train2014/*.jpg')
    train_dataset = train_dataset.map(load_img1,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(2048)
    train_dataset = train_dataset.batch(4, drop_remainder=True)

    for e in range(1):
        print('Epoch {}'.format(e+1))
        iteration = 0

        for img in train_dataset:
   

            for j, img_p in enumerate(img):
                X_batch[j] = img_p

            iteration += 1

        print(img)

    print(iteration)
    end = time.time()
    print("Total time: {:.1f}".format(end-start))


  







""" 

    import time
    start = time.time()


    for e in range(epoch):
        print('Epoch {}'.format(e+1))
        num = len(content_targets)
        iteration = 0

        while (iteration*batch_size < num):
            curr = iteration * batch_size
            step = curr + batch_size

            for j, img_p in enumerate(content_targets[curr:step]):
                X_batch[j] = load_img(img_p, resize=True)

            iteration += 1
            train_step(X_batch)

            if iteration % 1000 == 0:
                print('step %s: loss = %s' % (iteration, loss_metric.result()))

                # Save checkpoints
                network.save_weights('weights/weights', save_format='tf')
                print('=====================================')
                print('            Weights saved!           ')
                print('=====================================\n')


    end = time.time()
    print("Total time: {:.1f}".format(end-start))


    # Training is done !
    network.save_weights('weights/weights', save_format='tf')
    print('=====================================')
    print('             All saved!              ')
    print('=====================================\n')



 """








def main():

    parameters = {
        'style_file' : 'Source/starry_night.jpg',
        'train_path' : 'dataset/train2014',
        'content_weight' : 8e0,
        'style_weight' : 1e2,
        'tv_weight' : 2.5e2,
        'batch_size' : 4,
        'epoch' : 2
    }

    trainer(**parameters)


if __name__ == '__main__':
    main()