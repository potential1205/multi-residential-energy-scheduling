"""Device adapters for executing the legacy algorithms on CPU or CUDA."""
def model_on_device(cls,device):
    original_init=cls.__init__;original_forward=cls.forward
    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs);self.to(device)
    def forward(self,x):
        return original_forward(self,x.to(device))
    cls.__init__=initialize;cls.forward=forward
    return cls

def replay_on_device(cls,device):
    original_sample=cls.sample
    def sample(self,n):
        return tuple(t.to(device) for t in original_sample(self,n))
    cls.sample=sample
    return cls
