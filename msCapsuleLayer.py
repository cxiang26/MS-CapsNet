import mxnet as mx
from mxnet import gluon, nd, autograd, image, init
from mxnet.gluon import nn

def caps_dropout(X, drop_probability):
    keep_probability = 1 - drop_probability
    assert 0 <= keep_probability <= 1
    # 这种情况下把全部元素都丢弃。
    # all the uints are dropouted in the case.
    if keep_probability == 0:
        return X.zeros_like()

    # 随机选择一部分该层的输出作为丢弃元素。
    # some of layers are dropouted randomly.
    mask = nd.random.uniform(
        0, 1.0, X.shape[0]*X.shape[2], ctx=X.context).reshape((X.shape[0],1,X.shape[2],1,1)) < keep_probability
    # 保证 E[dropout(X)] == X
    scale =  1 / keep_probability
    return mask * X *scale

# class msPrimaryCap(nn.Block):
#     # k_size: the size of kernal
#     # cap_channels: 通道数，每个不同长度向量的通道
#     # len_vectors: 向量长度，是一个list，分别生成不同长度的向量
#     # stride: 卷积步长
#     def __init__(self, k_size, cap_channels, len_vectors, strides, **kwargs):
#         super(msPrimaryCap, self).__init__(**kwargs)
#         self.k = k_size
#         self.c = cap_channels
#         self.l = len_vectors
#         self.s = strides
#         self.net = nn.Sequential()
#         with self.name_scope():
#             for n in self.l:
#                 for _ in range(self.c):
#                     self.net.add(nn.Conv2D(channels=n, kernel_size=self.k, strides=self.s))
#     def forward(self, x):
#         out = []
#         output = []
#         for i, net in enumerate(self.net):
#             out.append(nd.reshape(net(x),(0,0,-1,1)))
#         for i in range(len(self.l)):
#             output.append(Squash(nd.expand_dims(nd.concat(*out[i*self.c:(i+1)*self.c], dim=2),axis=4),axis=1))
#         return output
class msPrimaryCap(nn.Block):
    # k_size: the size of kernal
    # cap_channels: 通道数，每个不同长度向量的通道
    # len_vectors: 向量长度，是一个list，分别生成不同长度的向量
    # stride: 卷积步长
    def __init__(self, k_size, cap_channels, len_vectors, strides, **kwargs):
        super(msPrimaryCap, self).__init__(**kwargs)
        self.k = k_size
        self.c = cap_channels
        self.l = len_vectors
        self.s = strides
        self.net = nn.Sequential()
        with self.name_scope():
            for _ in range(self.c):
                self.net.add(self.l_net(self.l[0]))
            for _ in range(self.c):
                self.net.add(self.c_net(self.l[1]))
            for _ in range(self.c):
                self.net.add(self.h_net(self.l[2]))
    # define lower-level feature extraction
    def l_net(self, channel):
        net = nn.Sequential()
        net.add(#nn.BatchNorm(),
                nn.Conv2D(channels=channel, kernel_size=(9,9), strides=(2,2)))
        return net
    # define med-level feature extraction
    def c_net(self, channel):
        net = nn.Sequential()
        net.add(#nn.BatchNorm(), # when the training sample is large, we need use BatchNorm.
                nn.Conv2D(channels=16, kernel_size=(5,5), strides=(2,2)),
                nn.BatchNorm(),
                nn.Activation('tanh'),
                nn.Conv2D(channels=channel, kernel_size=(3,3)))
        return net
    # define high-level feature extraction
    def h_net(self, channel):
        net = nn.Sequential()
        net.add(#nn.BatchNorm(),
                nn.Conv2D(channels=32, kernel_size=(3,3), padding=(1,1), strides=(2,2), activation='relu'),
                #nn.BatchNorm(),
                nn.Conv2D(channels=16, kernel_size=(3,3)),
                nn.BatchNorm(),
                nn.Activation('tanh'),
                nn.Conv2D(channels=channel, kernel_size=(3,3)))
        return net

    def forward(self, x):
        out = []
        output = []
        for i, net in enumerate(self.net):
            out.append(nd.reshape(net(x),(0,0,-1,1)))
        for i in range(len(self.l)):
            output.append(Squash(nd.expand_dims(nd.concat(*out[i*self.c:(i+1)*self.c], dim=2),axis=4),axis=1))
        return output

# activative function
def Squash(vector, axis):
    norm = nd.sum(nd.square(vector), axis, keepdims=True)
    v_j = norm/(1+norm)/nd.sqrt(norm, keepdims=True)*vector
    return v_j

class msCapsuleLayer(nn.Block):
    # len_vectors: 输入向量的长度，为list
    # len_vectors: 输出向量的长度，输出统一
    # num_input：输入capsule个数，为list, 即每个不同长度的capsule的个数
    # num_output: 输出capsule的个数，输出统一
    # num_routing: routing次数
    def __init__(self, len_vectors_input, len_vectors_output, batch_size, num_input, num_output, num_routing, drop_probability=None, **kwargs):
        super(msCapsuleLayer, self).__init__(**kwargs)
        self.is_train = True
        self.bs = batch_size
        self.lvi = len_vectors_input  #list
        self.lvo = len_vectors_output
        self.ni = num_input  #list
        self.no = num_output
        self.nr = num_routing
        self.dp = drop_probability
        self.W = []
        [self.W.append('self.W_%d'%(lvi)) for lvi in self.lvi] #self
        with self.name_scope():
            for i, lvi in enumerate(self.lvi):
                self.W[i] = self.params.get('weight_%d'%(lvi), shape=(1, lvi, self.ni[i], self.lvo, self.no), init=init.Normal(0.5))

    def forward(self, x): #由于输入维度不一致，此处输入x为list
        #x.shape: (batchsize, 8, 1152, 1, 1)
        #routing_weight.shape: (1, 1, 1152, 1, 10)
        routing_weight = nd.softmax(nd.zeros(shape=(1, 1, sum(self.ni), 1, self.no), ctx=x[0].context),axis=1)
        u = None
        for i,data in enumerate(x):
            # required the norm of child capsule greater than 0.2
            # x[i] = (nd.norm(x[i], axis=1, keepdims=True) > 0.2) * x[i]
            if u is None:
                u = nd.sum(x[i]*self.W[i].data(), axis=1, keepdims=True)
            else:
                u = nd.concat(*[u, nd.sum(x[i]*self.W[i].data(), axis=1, keepdims=True)], dim=2)

            #out.append(nd.sum(x[i]*self.W[i].data(), axis=1, keepdims=True))
        #u.shape: (batchsize, 1, 1152, 16, 10)
        #u = nd.sum(x*self.W.data(), axis=1, keepdims=True)
        #u = nd.concat(*out, dim=2)

        # advance the capsule dropout operation
        if self.is_train == True and self.dp is not None:
            u = caps_dropout(u, self.dp)
        s = nd.sum(u*routing_weight, axis=2, keepdims=True)
        
        v = Squash(s, axis=3)
        for i in range(self.nr):

            routing_weight = routing_weight + nd.sum(u*v, axis=3, keepdims=True)
            c = nd.softmax(routing_weight, axis=2)
            # c = nd.softmax(-(routing_weight<0)*routing_weight + (routing_weight>0)*routing_weight, axis=2)
            # c = -(routing_weight<0)*c + (routing_weight>0)*c
            # c = (routing_weight > 0)*nd.exp(routing_weight)/nd.sum((routing_weight>0)*routing_weight+1e-10, axis=2, keepdims=True)
            # testing the performance of different dimensional capsules
            # if i == (self.nr-1):
            #     c[:,:,:360,:,:] = 0
            s = nd.sum(u*c, axis=2, keepdims=True)
            v = Squash(s, axis=3)
        return nd.reshape(v,shape=(-1, self.lvo, self.no))

# the probability calculation
class length(nn.Block):
    def __init__(self, axis=1, **kwargs):
        super(length, self).__init__(**kwargs)
        self.axis = axis
    def forward(self, x):
        out = nd.sqrt(nd.sum(nd.square(x), self.axis))
        return out