import os
import glob
import argparse
from io import BytesIO
import pandas as pd
from tqdm import tqdm
from minio import Minio

### minio config set up
# export MINIO_ACCESS_KEY='dsml_key'
# export MINIO_SECRET_KEY='ThalesPass!'

MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY')
domain = '192.168.96.95:9000'

def upload_local_directory_to_minio(client, local_path, bucket_name, minio_path):
    # assert os.path.isdir(local_path)
    for local_file in tqdm(glob.glob(local_path + '/**')):
        local_file = local_file.replace(os.sep, "/")

        if not os.path.isfile(local_file):
            upload_local_directory_to_minio(
                client, local_file, bucket_name, minio_path + "/" + os.path.basename(local_file))
        else:
            remote_path = os.path.join(
                opt.prefix, local_file[1 + len(opt.origin):])
            remote_path = remote_path.replace(os.sep, "/") 
            client.fput_object(opt.bucket, remote_path, local_file)

def main(opt):
    minioClient = Minio(
        domain, 
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False)
    upload_local_directory_to_minio(minioClient, opt.origin, opt.bucket, opt.prefix)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin', type=str, default='./data', help='local path')
    parser.add_argument('--bucket', type=str, default='test', help='bucket in minio')
    parser.add_argument('--prefix', type=str, default='./landing/', help='path in minio')
    return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

# python publish.py --origin "../data/landing" --bucket "smart-passage-logic" --prefix "/data/landing"
