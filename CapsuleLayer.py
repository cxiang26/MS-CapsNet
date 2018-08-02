import mxnet as mx
from mxnet import nd, init
from mxnet.gluon import nn
	
class PrimaryCap(nn.Block):
    def __init__(self, k_size, cap_channels, len_vectors, strides, **kwargs):
        super(PrimaryCap, self).__init__(**kwargs)
        self.k = k_size
        self.c = cap_channels
        self.l = len_vectors
        self.s = strides
        self.net = nn.Sequential()
        with self.name_scope():
            for _ in range(self.c):
                self.net.add(nn.Conv2D(channels=self.l, kernel_size=self.k, strides=self.s))
    def forward(self, x):
        out = []
        for i, net in enumerate(self.net):
            out.append(nd.reshape(net(x),(0,0,-1,1)))
        return Squash(nd.expand_dims(nd.concat(*out, dim=2),axis=4),axis=1)
		
def Squash(vector, axis):
    norm = nd.sum(nd.square(vector), axis, keepdims=True)
    v_j = norm/(1+norm)/nd.sqrt(norm, keepdims=True)*vector
    return v_j

class CapsuleLayer(nn.Block):
    def __init__(self, len_vectors_input, len_vectors_output, batch_size, num_input, num_output, num_routing, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.bs = batch_size
        self.lvi = len_vectors_input
        self.lvo = len_vectors_output
        self.ni = num_input
        self.no = num_output
        self.nr = num_routing
        with self.name_scope():
            self.W = self.params.get('weight', shape=(1, self.lvi, self.ni, self.lvo, self.no), init=init.Normal(0.5))
    def forward(self, x):
        #x.shape: (batchsize, 8, 1152, 1, 1)
        #routing_weight.shape: (1, 1, 1152, 1, 10)
        routing_weight = nd.softmax(nd.zeros(shape=(1, 1, self.ni, 1, self.no), ctx=x.context),axis=1)
        
        #u.shape: (batchsize, 1, 1152, 16, 10)
        u = nd.sum(x*self.W.data(), axis=1, keepdims=True)
        
        #s.shape: (batchsize, 1, 1, 16, 10)
        s = nd.sum(u*routing_weight, axis=2, keepdims=True)
        
        #v.shape: (batchsize, 1, 1, 16, 10)
        v = Squash(s, axis=3)
        
        for i in range(self.nr):
            
            #print(i, nd.sum(nd.sum(nd.sum(nd.square(u*v), axis=3, keepdims=True), axis=2, keepdims=True).reshape((self.bs,10)),axis=1))
            routing_weight = routing_weight + nd.sum(u*v, axis=3, keepdims=True)
            c = nd.softmax(routing_weight, axis=2)
            s = nd.sum(u*c, axis=2, keepdims=True)
            v = Squash(s, axis=3)
        
        return nd.reshape(v,shape=(-1, self.lvo, self.no))
	
class length(nn.Block):
    def __init__(self, axis=1, **kwargs):
        super(length, self).__init__(**kwargs)
        self.axis = axis
    def forward(self, x):
        out = nd.sqrt(nd.sum(nd.square(x), self.axis))
        return out