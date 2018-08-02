
# coding: utf-8



import sys
import mxnet as mx
from mxnet import gluon, nd, autograd, image, init
from mxnet.gluon import nn
from msCapsuleLayer import msPrimaryCap, msCapsuleLayer, length
sys.path.append('../')
import utils


# In[2]:


def createnet(batch_size=2, ctx=mx.cpu()):
    msCapNet = nn.Sequential()
    with msCapNet.name_scope():
        msCapNet.add(nn.Conv2D(channels=256, kernel_size=13, strides=1, padding=(0,0), activation='relu'))
        msCapNet.add(msPrimaryCap(k_size=9, cap_channels=30, len_vectors=[4,8,12], strides=2))
        msCapNet.add(msCapsuleLayer(len_vectors_input=[4,8,12],len_vectors_output=16,batch_size=batch_size, num_input=[1080,1080,1080], num_output=10, num_routing=3,  drop_probability=None))
        msCapNet.add(length())
    msCapNet.initialize(ctx=ctx)
    return msCapNet


# In[3]:


def loss(y_pred, y_true):
    L = y_true * nd.square(nd.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * nd.square(nd.maximum(0., y_pred - 0.1))
    return nd.mean(nd.sum(L, 1))
# def o_loss(y_pred, label, net):
#     idx = nd.arange(y_pred.shape[0], ctx=y_pred.context)
#     y_true = nd.one_hot(label, 10)
#     v = nd.expand_dims(y_pred[idx, :, label], axis=2)
#     vv = net(y_pred)
#     L = y_true*nd.square(nd.maximum(0., 0.9-vv)) + 0.5*(1-y_true) * (nd.square(nd.maximum(0., vv-0.1))+nd.maximum(0, nd.square(nd.sum(y_pred*v, axis=1)/nd.norm(v, axis=1)/nd.norm(y_pred, axis=1))-0.1))
#     return nd.mean(nd.sum(L, 1))


if __name__ == "__main__":
    ctx = mx.gpu(2)
    Train = True
    batch_size = 128
    num_epochs = 50
    train_data, test_data = utils.load_data_cifar10(batch_size)
    net = createnet(batch_size, ctx)
    # net.load_parameters('best_acc_0.757.params', ctx=ctx)

    if Train:
        print('================Train==================')
        
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001, 'wd':0.00001}) #'wd':0.00001
        utils.traincaps(train_data, test_data, net, loss, trainer, ctx, num_epochs)