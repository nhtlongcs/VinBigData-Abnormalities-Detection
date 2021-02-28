import yaml

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key) + ':')
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

class Config():
    def __init__(self, yaml_path):
        yaml_file = open(yaml_path)
        self._attr = yaml.load(yaml_file, Loader=yaml.FullLoader)['settings']

    def __getattr__(self, attr):
        try:
            return self._attr[attr]
        except KeyError:
            return None

    def __str__(self):
        print("##########   CONFIGURATION INFO   ##########")
        pretty(self._attr)
        