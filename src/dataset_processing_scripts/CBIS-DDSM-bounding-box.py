import os

import pandas as pd
from pathlib import Path
import pydicom
from skimage.measure import label, regionprops
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np


def main() -> None:
    """
    Initial dataset pre-processing for the CBIS-DDSM dataset to generate the bounding box ground truth for images
    :return: None
    """
    csv_path = "../data/CBIS-DDSM-mask/final_mask_training.csv"
    as_df = pd.read_csv(csv_path)
    
    f = open("../data/CBIS-DDSM-mask/bbox_groud_truth.txt", "a")
    for i in range(as_df.shape[0]):
        string = as_df["img_path"][i]
        # Get bounding box
        minr, minc, maxr, maxc = get_bbox_of_mask(as_df["mask_img_path"][i])
        string += "," + str(minr)
        string += "," + str(minc)
        string += "," + str(maxr)
        string += "," + str(maxc)
        string += ",0"
        f.write("\n")
        f.write(string)

    f.close()


def get_bbox_of_mask(mask_path):
    # Process input ground truth mask and generate bounding box dimensions of tumours
    image_bytes = tf.io.read_file(mask_path)
    image = tfio.image.decode_dicom_image(image_bytes, color_dim = True,  dtype=tf.uint16)
    image = tf.image.resize_with_pad(image, 1024, 640)
    array = np.array(image)
    array = array[0,:,:,0].astype(int)
    regions = regionprops(array)
    region = regions[0]
    minr, minc, maxr, maxc = region.bbox
    return minr, minc, maxr, maxc


if __name__ == '__main__':
    main()
