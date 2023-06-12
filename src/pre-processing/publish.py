import os
import argparse
from io import BytesIO
import pandas as pd
from minio import Minio

### minio config set up
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY')
domain = '192.168.96.95:9000'

def main(opt):
    minioClient = Minio(
        domain, 
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin', type=str, default='./data', help='local path')
    parser.add_argument('--bucket', type=str, default='test', help='bucket in minio')
    parser.add_argument('--prefix', type=str, default='./landing/', help='path in minio')
    return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

# python publish.py --origin "../data/landing" --bucket "test" --prefix "landing"
