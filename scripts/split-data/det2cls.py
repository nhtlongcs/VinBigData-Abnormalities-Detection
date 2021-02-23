import os
import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp',
                        type=str,
                        default='../data/train_fold.csv',
                        help='path to input directory')
    return parser.parse_args()


def det2cls(csv_path, output_fn):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Find all classes appearing in each image
    class_by_im = df.groupby('image_id')['class_id'].unique()

    # Generate output dataframe
    output = pd.DataFrame({
        'image_id': df['image_id'].unique()
    })

    # For each image, check if the class exists
    for i in sorted(df['class_id'].unique()):
        output[f'class_{i}'] = \
            output['image_id'].apply(lambda x: i in class_by_im[x]).astype(int)

    # Save to file
    output.to_csv(output_fn, index=False)


# Parse arguments
args = parse_args()

for fold_id in os.listdir(args.inp):
    if not os.path.isdir(args.inp):
        continue

    for phase in ['train', 'val']:
        input_csv = f'{args.inp}/{fold_id}/{fold_id}_{phase}.csv'
        output_csv = f'{args.inp}/{fold_id}/{fold_id}_{phase}_cls.csv'
        det2cls(input_csv, output_csv)
