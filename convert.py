import pandas as pd
import numpy as np

def process_df(ori_csv, ims_dir, ims_ext, nw, nh, output_fn):
    '''
    Convert detection csv into the csv used in training RetinaNet.

    Args:
        ori_csv (str): path to the detection csv
        ims_dir (str): directory of the images
        ims_ext (str): extension of the images
        nw, nh (int, int): new image dimensions
        output_fn: path and filename of the output csv
    '''

    # Load detection csv
    df = pd.read_csv(ori_csv)

    # Create an empty dataframe
    output = pd.DataFrame()

    # Process bbox coordinates
    output['x1'] = df['x_min'] / df['width'] * nw
    output['y1'] = df['y_min'] / df['height'] * nh
    output['x2'] = df['x_max'] / df['width'] * nw + 1 # +1 for no 0 side
    output['y2'] = df['y_max'] / df['height'] * nh + 1 # same

    # Convert NaN into empty string (because cannot convert NaN to integer)
    output = output.fillna(-1).astype(int).astype('str').replace('-1', np.nan)

    # Process images path
    output['path'] = df['image_id'].apply(
        lambda x: f'{ims_dir}/{x}{ims_ext}'
    )

    # Process class labels
    output['class_name'] = df['class_name'].apply(lambda x: '' if x == 'No finding' else x)

    # Write file
    output = output[['path', 'x1', 'y1', 'x2', 'y2', 'class_name']]
    output.to_csv(output_fn, header=False, index=False)

root = 'data'

# Prepare train/val csv files
for fid in range(5):
    for phase in ['train', 'val']:
        process_df(f'{root}/{fid}_{phase}.csv',
                   f'{root}/vinbigdata-512/train', '.png',
                   512, 512,
                   f'{root}/vinbigdata-512/{fid}_{phase}_retina.csv')

# Prepare class_mapping file
class_mapping = pd.DataFrame({
    'class_name': ['Aortic enlargement',
                    'Atelectasis',
                    'Calcification',
                    'Cardiomegaly',
                    'Consolidation',
                    'ILD',
                    'Infiltration',
                    'Lung Opacity',
                    'Nodule/Mass',
                    'Other lesion',
                    'Pleural effusion',
                    'Pleural thickening',
                    'Pneumothorax',
                    'Pulmonary fibrosis',],
    'id': list(range(14))
})
class_mapping.to_csv(root + '/' + 'class_mapping.csv', index=False, header=False)