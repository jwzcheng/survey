# %%
import os
import shutil
#%%
def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)


makedirs('tmp/ori')
makedirs('tmp/tar')


with open('tmp/ori/test.txt', 'a') as f:
    f.write('hello python!')

ori = 'tmp/ori/'
tar = 'tmp/tar/'

shutil.move(ori+'test.txt', tar+'test.txt')

# os.remove('tmp/tar/test.txt')
# %%
