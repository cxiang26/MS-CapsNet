import logging
import mxnet as mx
from mxboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
resized_images = mx.nd.load('resize_image.ndarray')[0]
convnet_codes = mx.nd.load('convet.ndarray')[0]
labels = mx.nd.load('label.ndarray')[0].asnumpy()

label_strs = ['0','1','2','3','4','5','6','7','8','9']

with SummaryWriter(logdir='./logs4') as sw:
    sw.add_image(tag='cifar10', image=resized_images)
    sw.add_embedding(tag='capsule_codes', embedding=convnet_codes, images=resized_images,labels=[label_strs[idx] for idx in labels])