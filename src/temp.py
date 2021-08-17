import georinex as gr
import pickle
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm

R = Path(__file__).parent / "data"

print(R)
flist = list(glob("data/**/*.*o",recursive=True)) + list(glob("data/**/*.obs",recursive=True))
assert len(flist) > 0

def myLoadRinex(path):
    if os.path.exists(path+'.pkl'):
        with open(path+'.pkl', 'rb') as f:
            return pickle.load(f)

    d = gr.load(path)
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    
    return d

d1 = myLoadRinex('data/train/2020-05-14-US-MTV-1/Pixel4/supplemental/Pixel4_GnssLog.20o')
dt = d1["C1C"]
print(dt[0][0])
print(dt[0][0].sv)
print(dt[0][0].time)
exit()

count = 0
for o in d1["C1C"]:
    print(o)
    count += 1
    if(count > 20):
        break


for f in flist:
    print("Loading ", f)
    myLoadRinex(f)

d1 = myLoadRinex('data/train/2020-05-14-US-MTV-1/Pixel4/supplemental/Pixel4_GnssLog.20o')
print(d1)
d2 = myLoadRinex('data/google-sdc-corrections/osr/rinex/2020-05-14-US-MTV-1.obs')
print(d2)

'''
for t in d1.time:
    print(t)


from matplotlib.pyplot import figure, show

ax = figure().gca()
ax.plot(d2.time, d2['L1C'])
show()
'''