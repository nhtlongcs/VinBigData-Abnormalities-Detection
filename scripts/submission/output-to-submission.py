import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--df',
                    type=str,
                    default='output.csv',
                    help='path to model output csv file')
parser.add_argument('--test_info',
                    type=str,
                    default='data/test_info.csv',
                    help='path to test info csv file')
parser.add_argument('--filename',
                    type=str,
                    default='submission.csv',
                    help='output csv file name')
parser.add_argument('--round_to',
                    type=int,
                    default=3,
                    help='number of decimal places to round confidence values to. \
                        Round to 3 decimal places by default.')
parser.add_argument('--use_trick',
                    action='store_true',
                    default=False,
                    help='add class 14 to all images')
parser.add_argument('--keep_ratio',
                    action='store_true',
                    default=False,
                    help='set this flag if your model central-pads the images \
                        and you want to subtract the padding from your outputs')
args = vars(parser.parse_args())


def to_submission(df, test_info, filename, round_to, use_trick, keep_ratio):
    '''
    df: model output filename
    test_info: sample_submission file (to get image_id)
    filename: output csv file name
    round_to: number of decimal places to round confidence values to
    use_trick: add class 14 to all images
    keep_ratio: undo central pad
    
    return a dataframe
    '''

    df = pd.read_csv(df)
    if keep_ratio:
        print("Subtracting outputs to take account of central padding")
    if use_trick:
        print("Adding class 14")
    
    ss = pd.read_csv(test_info)
    image_dict = {i: [] for i in ss['image_id']}
    sizemap = {i: (w, h) for i, w, h in zip(ss['image_id'], ss['width'], ss['height'])}
    trick_string = {i: f'14 {prob} 0 0 1 1' for i, prob in zip(ss['image_id'], ss['class14prob'])}

    if df['class_id'].min() == 1:
        sub_one = True
        print("Subtracting one from class_ids")
    else:
        sub_one = False

    for row in df.itertuples(index=False):
        image_id = row.image_id
        w, h = sizemap[image_id]

        x_min = row.x_min * w
        y_min = row.y_min * h
        x_max = row.x_max * w
        y_max = row.y_max * h

        if keep_ratio:
            if w > h:
                y_min -= (w - h) / 2
                y_max -= (w - h) / 2
            else:
                x_min -= (h - w) / 2
                x_max -= (h - w) / 2

        x_min, y_min, x_max, y_max = (round(x) for x in (x_min, y_min, x_max, y_max))

        assert 0 <= x_min <= w
        assert 0 <= x_max <= w
        assert 0 <= y_min <= h
        assert 0 <= y_max <= h

        class_id = int(row.class_id)
        if sub_one:
            class_id -= 1

        confidence = round(row.score, round_to)

        image_dict[image_id].append(f"{class_id} {confidence} {x_min} {y_min} {x_max} {y_max}")

    def convert_entry_to_one_prediction_string(img_id, preds):
        # model does not predict anything for this image
        if len(preds) == 0:
            return trick_string[img_id]
        # model output contains only 14 x 0 0 1 1
        elif len(preds) == 1 and preds[0].startswith("14 "):
            conf = preds[0].split()[1]
            return f"14 {conf} 0 0 1 1"
        else:
            if use_trick:
                preds.append(trick_string[img_id])
            return " ".join(preds)

    result = pd.DataFrame().from_dict({
        'image_id': image_dict.keys(),
        'PredictionString': [convert_entry_to_one_prediction_string(k, v) for k, v in image_dict.items()]
    })

    result.sort_values('image_id', ignore_index=True, inplace=True)
    
    if filename:
        result.to_csv(filename, index=False)
    return result


if __name__ == '__main__':
    to_submission(**args)
