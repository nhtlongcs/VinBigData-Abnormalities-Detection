import argparse
import pandas as pd
import warnings

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
parser.add_argument('--trick',
                    type=bool,
                    default=False,
                    help='add class 14 to all images')
parser.add_argument('--keep_ratio',
                    type=bool,
                    default=False,
                    help='Your output is keep ratio')
args = vars(parser.parse_args())

def to_submission(df='output.csv', test_info='data/test_info.csv', filename='submission.csv', trick=True, keep_ratio=False):
    '''
    df: model output filename
    test_info: sample_submission file (to get image_id)
    filename: output csv file name
    trick: add class 14 to all images
    keep_ratio: Your output is keep ratio
    
    return a dataframe
    '''
    
    df = pd.read_csv(df)
    
    ss = pd.read_csv(test_info)
    d = {i:[] for i in ss['image_id'].values.tolist()}
    sizemap = {i:(w,h) for i,w,h in zip(ss['image_id'].values.tolist(), ss['width'].values.tolist(), ss['height'].values.tolist())}
    trick_string = {i:'14 '+str(s)+' 0 0 1 1' for i,s in zip(ss['image_id'].values.tolist(), ss['class14prob'].values.tolist())}
    
    image_id = []
    PredictionString = []
    for i in range(len(df['image_id'].values.tolist())):
        w,h = sizemap[df['image_id'][i]]

        x_min = round(float(df['x_min'][i])*w)
        y_min = round(float(df['y_min'][i])*h)
        x_max = round(float(df['x_max'][i])*w)
        y_max = round(float(df['y_max'][i])*h)

        if (keep_ratio):
            if (w > h):
                y_min -= (w-h)/2
                y_max -= (w-h)/2
            else:
                x_min -= (h-w)/2
                x_max -= (h-w)/2
                
            if ((x_min < 0) or (y_min < 0)):
                warnings.warn("Check your keep_ratio flag!!!")
            

        d[df['image_id'][i]].append(" ".join(map(str, [int(df['class_id'][i]), df['score'][i], x_min, y_min, x_max, y_max] )))
        
    for (k,v) in d.items():
        image_id.append(k)
        if (len(v) == 0): #case 14 1 0 0 1 1 not in df
            PredictionString.append(trick_string[k])
        elif (len(v) == 1 and v[0].startswith("14 ")): #case 14 x 0 0 1 1 in df
            PredictionString.append(" ".join(v[0].split()[:2])+' 0 0 1 1')
        else:
            if (trick):
                v.append(trick_string[k])
            PredictionString.append(" ".join(v))
    result = pd.DataFrame()
    result['image_id'] = image_id
    result['PredictionString'] = PredictionString
    
    result.sort_values('image_id', ignore_index=True,  inplace=True)
    
    if (filename):
        result.to_csv(filename, index = False)
    return result
    
if __name__ =='__main__':
    to_submission(**args)
