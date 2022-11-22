from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


try:
    from urllib.request import urlopen  # python 3
except ImportError:
    from urllib2 import urlopen  # python 2
import sys
import tarfile
import tempfile
import shutil

'''------------------该横线范围内被我修改------------------'''
# dataset = sys.argv[1]  # 原版
dataset = 'facades'  # 选择要下载的datase
'''------------------------终止线------------------------'''

url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/%s.tar.gz" % dataset
with tempfile.TemporaryFile() as tmp:
    print("downloading", url)
    shutil.copyfileobj(urlopen(url), tmp)
    print("extracting")
    tmp.seek(0)
    tar = tarfile.open(fileobj=tmp)
    tar.extractall()
    tar.close()
    print("done")
