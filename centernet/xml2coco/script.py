from os import listdir
from os.path import isfile, join
path = "./sample/xmls"
onlyfiles = [f.split('.')[0] for f in listdir(path) if isfile(join(path, f))]
onlyfiles.sort()
with open('./sample/list.txt', 'w') as the_file:
    for f in onlyfiles:
      the_file.write(f + '\n')
