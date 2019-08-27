from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data.dataset import Dataset
import mxnet as mx
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, filename='./log/capsule.log')
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(level=logging.INFO)
logger.addHandler(stream_handler)

transform_train = transforms.Compose([transforms.RandomResizedCrop(32,(0.8,1.0),ratio=(1.0,1.0)),
                                     transforms.RandomFlipLeftRight(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
transform_test = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
class getndata(Dataset):
    def __init__(self, train_data, transform, num):
        super(getndata,self).__init__()
        self._transform = transform
        self.data = train_data
        self.num = num
        self._data = None
        self._label = None
        self._get_data()
    def __getitem__(self, idx):
        if self._transform is not None:
            return self._transform(self._data[idx], self._label[idx])
        return self._data[idx], self._label[idx]
    def __len__(self):
        return len(self._label)
    def _get_data(self):
#         raise NotImplementedError
        num = np.array([0,0,0,0,0,0,0,0,0,0,0])
        for i in range(len(self.data)):
            if num[int(self.data._label[i])]<self.num:
                num[int(self.data._label[i])] += 1
                num[-1] += 1
                if self._data is None:
                    self._data = nd.expand_dims(self.data._data[i], axis=0)
                else:
                    self._data = nd.concat(*[self._data, nd.expand_dims(self.data._data[i], axis=0)], dim=0)
                if self._label is None:
                    self._label = [self.data._label[i]]
                else:
                    self._label.append(self.data._label[i])
            elif num[-1] == self.num*10:
                break

class getndata_for_cifar10(Dataset):
    def __init__(self, train_data, transform, num):
        super(getndata_for_cifar10,self).__init__()
        self._transform = transform
        self.data = train_data._data
        self.num = num
        self._data = None
        self._label = None
        self._get_data()
    def __getitem__(self, idx):
        if self._transform is not None:
            return self._transform(self._data[idx]), self._label[idx]
        return self._data[idx], self._label[idx]
    def __len__(self):
        return len(self._label)
    def _get_data(self):
#         raise NotImplementedError
        num = np.array([0,0,0,0,0,0,0,0,0,0,0])
        for i in range(len(self.data)):
            if num[int(self.data._label[i])]<self.num:
                num[int(self.data._label[i])] += 1
                num[-1] += 1
                if self._data is None:
                    self._data = nd.expand_dims(self.data._data[i], axis=0)
                else:
                    self._data = nd.concat(*[self._data, nd.expand_dims(self.data._data[i], axis=0)], dim=0)
                if self._label is None:
                    self._label = [self.data._label[i]]
                else:
                    self._label.append(self.data._label[i])
            elif num[-1] == self.num*10:
                break

def load_data_cifar10(batch_size, n=None):
    """download the fashion mnist dataest and then load into memory"""
    cifar_train = gluon.data.vision.CIFAR10(train=True).transform_first(transform_train)
    cifar_test = gluon.data.vision.CIFAR10(train=False).transform_first(transform_test)
    if n:
        cifar_train = getndata_for_cifar10(cifar_train, transform_train, n)
    train_data = gluon.data.DataLoader(
        cifar_train, batch_size, shuffle=True, last_batch='discard', num_workers=4)
    test_data = gluon.data.DataLoader(
        cifar_test, batch_size, shuffle=False, num_workers=4)
    return (train_data, test_data)

def load_data_fashion_mnist(batch_size, resize=None, n=None):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        if resize:
            # resize to resize x resize
            data = image.imresize(data, resize, resize)
        # change data's shape from height x weight x channel to channel x height x weight
        return nd.transpose(data.astype('float32'), (2,0,1))/255, label.astype('float32')
    mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform_mnist)
    mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform_mnist)
    # get different training samples
    if n:
        mnist_train = getndata(mnist_train, transform_mnist, n)
    train_data = gluon.data.DataLoader(
        mnist_train, batch_size, shuffle=True, num_workers=4)
    test_data = gluon.data.DataLoader(
        mnist_test, batch_size, shuffle=False, num_workers=4)
    return (train_data, test_data)

def load_data_mnist(batch_size, resize=None, n=None):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        if resize:
            # resize to resize x resize
            data = image.imresize(data, resize, resize)
        # change data from height x weight x channel to channel x height x weight
        return nd.transpose(data.astype('float32'), (2,0,1))/255, label.astype('float32')
    mnist_train = gluon.data.vision.MNIST(train=True, transform=transform_mnist)
    mnist_test = gluon.data.vision.MNIST(train=False, transform=transform_mnist)
    if n:
        import numpy as np
        mnist = []
        num = np.array([0,0,0,0,0,0,0,0,0,0,0])
        for data in mnist_train:
            if num[int(data[1])]<n:
                num[int(data[1])] += 1
                num[-1] += 1
                mnist.append(data)
            elif num[-1] == n*10:
                mnist_train = mnist
                break
    train_data = gluon.data.DataLoader(
        mnist_train, batch_size, shuffle=True, num_workers=4)
    test_data = gluon.data.DataLoader(
        mnist_test, batch_size, shuffle=False, num_workers=4)
    return (train_data, test_data)

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def accuracy(output, label):
    # print('accuracy',output, label)
    return nd.mean(nd.argmax(output,axis=1)==label).asscalar()

def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return data.as_in_context(ctx), label.as_in_context(ctx)

def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    acc = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        data, label = _get_batch(batch, ctx)
        output, _ = net(data)
        acc += accuracy(output, label.astype('float32'))
    return acc / (i+1)

def evaluate_acc(data_iterator, net, ctx=mx.cpu()):
    acc = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        data, label = _get_batch(batch, ctx)
        output = net(data)
        acc += accuracy(output, label.astype('float32'))
    return acc / (i+1)

def embedding(data_iterator, net, ctx=mx.cpu()):
    convnet_codes = None
    resize_images = None
    labels = None
    for i, batch in enumerate(data_iterator):
        data, label = _get_batch(batch, ctx)
        idx = nd.arange(data.shape[0])
        _, output = net(data)
        output = output[idx.as_in_context(ctx), :, label]
        output.wait_to_read()
        if convnet_codes is None:
            convnet_codes = output
        else:
            convnet_codes = nd.concat(*[convnet_codes, output], dim=0)
        if labels is None:
            labels = label
        else:
            labels = nd.concat(*[labels, label], dim=0)
        images = data.copyto(mx.cpu())
        if images.shape[1] != 1:
            images[:,0,:,:] += 0.4914
            images[:,1,:,:] += 0.4822
            images[:,2,:,:] += 0.4465
        images = nd.clip(images*255, 0, 255).astype('uint8')
        if resize_images is None:
            resize_images = images
        else:
            resize_images = nd.concat(*[resize_images, images], dim=0)
    nd.save('convet.ndarray', convnet_codes.as_in_context(mx.cpu()))
    nd.save('resize_image.ndarray', resize_images)
    nd.save('label.ndarray', labels.astype('int32').as_in_context(mx.cpu()))

def traincaps(train_data, test_data, net, loss, trainer, dnet, dloss, trainer_d, ctx, num_epochs, lr_decay=None, print_batches=None):
    best_acc = 0.
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        n = 0
        Loss = []
        net.digitcap_decode.is_train = True
        for i, batch in enumerate(train_data):
            data, label = batch
            one_hot_label = nd.one_hot(label, 10)
            with autograd.record():
                idx = nd.arange(0,data.shape[0], ctx=ctx)
                output, digitcaps = net(data.as_in_context(ctx))
                doutput = dnet(digitcaps[idx, :, label.as_in_context(ctx)])
                l = loss(output, one_hot_label.as_in_context(ctx))
                dl = dloss(doutput, data.reshape((0, -1)).as_in_context(ctx))
                L = l+dl
            L.backward()
            trainer_d.step(data.shape[0])
            trainer.step(data.shape[0])
            train_loss += nd.mean(L).asscalar()
            Loss.append(nd.mean(L).asscalar())
            train_acc += accuracy(output, label.astype('float32').as_in_context(ctx))
            n = i + 1
            if print_batches and n%print_batches == 0:
                print('Batch %d | Loss: %f | Train acc: %f'%(n, train_loss/n, train_acc/n))
        if lr_decay and epoch%lr_decay == 0:
            trainer.set_learning_rate(lr=trainer.learning_rate*0.1)
        net.digitcap_decode.is_train = False
        test_acc = evaluate_accuracy(test_data, net, ctx)
        if test_acc > best_acc:
            net.save_parameters('./params/best_acc_%.3f.params'%test_acc)
            best_acc = test_acc
        logger.info('Epoch %d | Loss: %f | Train acc: %f | Test acc: %f'%(epoch, train_loss/n, train_acc/n, test_acc))
    net.load_parameters('./params/best_acc_%.3f.params'%best_acc)
    embedding(test_data, net, ctx)

def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, lr_decay=None, print_batches=None):
    best_acc = 0.
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        n = 0
        Loss = []
        for i, batch in enumerate(train_data):
            data, label = batch
            one_hot_label = nd.one_hot(label, 10)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                L = loss(output, one_hot_label.as_in_context(ctx))
            L.backward()
            trainer.step(data.shape[0])
            train_loss += nd.mean(L).asscalar()
            Loss.append(nd.mean(L).asscalar())
            train_acc += accuracy(output, label.astype('float32').as_in_context(ctx))
            n = i + 1
            if print_batches and n%print_batches == 0:
                print('Batch %d | Loss: %f | Train acc: %f'%(n, train_loss/n, train_acc/n))
        if lr_decay and (epoch+1)%lr_decay == 0:
            trainer.set_learning_rate(lr=trainer.learning_rate*0.1)
            print('learning rate: ', trainer.learning_rate)
        test_acc = evaluate_acc(test_data, net, ctx)
        if test_acc > best_acc:
            net.save_parameters('./params/best_acc_%.3f.params'%test_acc)
            best_acc = test_acc
        logger.info('Epoch %d | Loss: %f | Train acc: %f | Test acc: %f'%(epoch, train_loss/n, train_acc/n, test_acc))
    net.load_parameters('./params/best_acc_%.3f.params'%best_acc)