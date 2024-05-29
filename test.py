from glob import glob
import os 
path = './'

dir = glob(os.path.join(path, '*'))

print(sorted(dir))