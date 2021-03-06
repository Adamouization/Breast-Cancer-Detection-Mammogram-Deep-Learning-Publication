import argparse
import time

from data_operations.dataset_feed import create_dataset_masks
from data_operations.data_preprocessing import import_cbisddsm_training_dataset, \
    generate_image_transforms, import_cbisddsm_segmentation_training_dataset
from data_visualisation.output import evaluate, evaluate_segmentation, visualise_examples
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from utils import create_label_encoder, print_error_message, print_num_gpus_available, print_runtime
from model.train_model_segmentation import make_predictions, train_segmentation_network
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def main() -> None:
    """
    Script to check ground truth masks as one was found to be most of the image and not a tumour
    """
    parse_command_line_arguments()
    print_num_gpus_available()

    # Start recording time.
    start_time = time.time()

#     df = pd.read_csv("../data/CBIS-DDSM-mask/final_mask_training.csv")
    
    df = pd.read_csv("../data/CBIS-DDSM-mask/shortened_mask_testing.csv")

    images = df['img_path'].values
    image_masks = df['mask_img_path'].values
    
    i = 0

    for image_mask_name in image_masks:
        image_bytes_mask = tf.io.read_file(image_mask_name)
        image_mask = tfio.image.decode_dicom_image(image_bytes_mask, color_dim = True)
        image_mask = tf.image.resize_with_pad(image_mask, config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
        current_min_mask = tf.reduce_min(image_mask)
        current_max_mask = tf.reduce_max(image_mask)
        image_mask = (image_mask - current_min_mask) / (current_max_mask - current_min_mask)
        array = np.array(image_mask)
        sum_arr = sum(sum(sum(array)))
        if sum_arr>200000:
            print (image_mask_name)



def parse_command_line_arguments() -> None:
    """
    Parse command line arguments and save their value in config.py.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        default="CBIS-DDSM",
                        help="The dataset to use. Must be either 'mini-MIAS' or 'CBIS-DDMS'."
                        )
    parser.add_argument("-m", "--model",
                        default="basic",
                        help="The model to use. Must be either 'basic' or 'advanced'."
                        )
    parser.add_argument("-r", "--runmode",
                        default="train",
                        help="Running mode: train model from scratch and make predictions, otherwise load pre-trained "
                             "model for predictions. Must be either 'train' or 'test'."
                        )
    parser.add_argument("-i", "--imagesize",
                        default="small",
                        help="small: use resized images to 512x512, otherwise use 'large' to use 2048x2048 size image with model with extra convolutions for downsizing."
                        )
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Verbose mode: include this flag additional print statements for debugging purposes."
                        )
    parser.add_argument("-s", "--segmodel",
                        default="RS50",
                        help="Segmentation model to be used."
                        )
    parser.add_argument("-p", "--prep",
                        default="N",
                        help="Preprocessing of images"
                        )

    args = parser.parse_args()
    config.dataset = args.dataset
    config.model = args.model
    config.run_mode = args.runmode
    config.imagesize = args.imagesize
    config.verbose_mode = args.verbose
    config.segmodel = args.segmodel
    config.prep = args.prep
    


if __name__ == '__main__':
    main()
