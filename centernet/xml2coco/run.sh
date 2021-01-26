mkdir ./sample/imgs
mkdir ./sample/xmls
mv ./sample/*.jpg ./sample/imgs/
mv ./sample/*.xml ./sample/xmls/
python3 script.py
python3 voc2coco.py  --ann_dir ./sample/xmls --ann_ids ./sample/list.txt --labels ./labels.txt --output output.json  --ext xml
