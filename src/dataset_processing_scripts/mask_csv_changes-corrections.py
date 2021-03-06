import os

import pandas as pd


def main() -> None:

    corrections = pd.read_csv("../data/CBIS-DDSM-mask/corrections_required.csv")
    img_names = corrections["img"].values
    new_img_names = []
    for i in img_names:
        new_img_names.append(i[:-2])
    original_data = pd.read_csv("../data/CBIS-DDSM-mask/shortened_mask_testing.csv")

    for i in range(original_data.shape[0]):
        if original_data["img"][i] in new_img_names:
            original_data["mask_img_path"][i] = original_data["mask_img_path"][i][:-7] + "1-1.dcm"
    
    original_data.to_csv("../data/CBIS-DDSM-mask/final_mask_testing.csv", index=False)
    
                      


if __name__ == '__main__':
    main()

