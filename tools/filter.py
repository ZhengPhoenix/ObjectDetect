import os
import shutil

target = 'target'

IDClean = 'IDCardClean_1196'
IDSample = 'IDCardSample_1541'

count = 0
for fn in os.listdir(IDSample):
    if fn[0] == '.':
        continue

    if fn.endswith('csv'):
        continue

    if os.path.isfile(os.path.join(IDClean, fn)):
        continue

    count += 1
    shutil.copy(os.path.join(IDSample, fn), target)

print('{} files copied'.format(count))