import pysftp
import pathlib

sHostName = ''
sUserName = ''
sPassWord = ''

cnopts = pysftp.CnOpts(knownhosts='my_macbook')
cnopts.hostkeys = None

with pysftp.Connection(sHostName, username=sUserName, password=sPassWord, cnopts=cnopts) as sftp:
    # 移動目錄
    sftp.cwd('./tmp/original/')

    # 取得目錄內容
    directory = sftp.listdir_attr()

    # 印出結果
    for attr in directory:
        print(attr.filename, attr)

# TEST copy files from /tmp/original
original = '/Users/jaycheng/tmp/original'

