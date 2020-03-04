import os
import argparse
from train import trainer
from evaluate import transfer

# USAGE
# python main.py train
# python main.py evaluate 

CONTENT_WEIGHT = 6e0
STYLE_WEIGHT = 2e-3
TV_WEIGHT = 6e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
BATCH_SIZE = 2

STYLE_IMAGE = './Source/udnie.jpg'
CONTENT_IMAGE = './Source/101.jpg'
DATASET_PATH = '../datasets/train2014'
WEIGHTS_PATH = './weights/weights'
RESULT_NAME = './Results/Result.jpg'


def main():
    # Paese command line arguments
    parser = argparse.ArgumentParser(
        description='Train Fast Style Transfer.')
    parser.add_argument('command',
                         metavar='<command>',
                         help="'train' or 'evaluate'")
    parser.add_argument('--debug', required=False,
                         metavar=False,
                         help='Whether to print the loss',
                         default=False)
    parser.add_argument('--dataset', required=False,
                         metavar=DATASET_PATH,
                         default=DATASET_PATH)
    parser.add_argument('--style', required=False,
                         metavar=STYLE_IMAGE,
                         help='Style image to train the specific style',
                         default=STYLE_IMAGE) 
    parser.add_argument('--image', required=False,
                         metavar=CONTENT_IMAGE,
                         help='Content image to evaluate with',
                         default=CONTENT_IMAGE)  
    parser.add_argument('--weights', required=False,
                         metavar=WEIGHTS_PATH,
                         help='Checkpoints directory',
                         default=WEIGHTS_PATH)
    parser.add_argument('--result_name', required=False,
                         metavar=RESULT_NAME,
                         help='Path to the transfer results',
                         default=RESULT_NAME)
    args = parser.parse_args()



    # Validate arguments
    if args.command == "train":
        assert os.path.exists(args.dataset), 'dataset path not found !'
        assert os.path.exists(args.style), 'style image not found !'
        assert BATCH_SIZE > 0
        assert NUM_EPOCHS > 0
        assert CONTENT_WEIGHT >= 0
        assert STYLE_WEIGHT >= 0
        assert TV_WEIGHT >= 0
        assert LEARNING_RATE >= 0

        parameters = {
                'style_file' : args.style,
                'train_path' : args.dataset,
                'weights_path' : args.weights,
                'content_weight' : CONTENT_WEIGHT,
                'style_weight' : STYLE_WEIGHT,
                'tv_weight' : TV_WEIGHT,
                'learning_rate' : LEARNING_RATE,
                'batch_size' : BATCH_SIZE,
                'epochs' : NUM_EPOCHS,
                'debug' : args.debug
            }

        trainer(**parameters)


    elif args.command == "evaluate":
        assert args.image, 'content image not found !'
        assert args.weights, 'weights path not found !'

        parameters = {
                'image' : args.image,
                'weights' : args.weights,
                'result' : args.result_name,
            }

        transfer(**parameters)

if __name__ == '__main__':
    main()