import torch
import torch.nn as nn
import math
import thop
from copy import deepcopy
from pathlib import Path
from general import LOGGER

anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
ch = [128, 256, 512]





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")






class Conv(nn.Module):
    def __init__(self, c1,c2,k,s,p=0,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(c1,c2,k,s,p,bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))
    






class Bottleneck(nn.Module):
    def __init__(self,c1,c2, shortcut=True,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        c_ = int(c2*1)
        self.cv1 = Conv(c1,c_,1,1)
        self.cv2 = Conv(c_,c2,3,1,1)
        self.add = shortcut and c1==c2

    def forward(self,x):
        return x+self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    def __init__(self, c1,c2,n,shortcut=True,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        c_ = int(c2*0.5)
        self.cv1 = Conv(c1,c_,1,1)
        self.cv2 = Conv(c1,c_,1,1)
        self.cv3 = Conv(2*c_,c2,1,1)
        self.m = nn.Sequential(*(Bottleneck(c_,c_,shortcut) for _ in range(n)))
    def forward(self,x):
        return self.cv3(torch.cat((self.m(self.cv1(x)),self.cv2(x)),1))
    

class SPPF(nn.Module):
    def __init__(self, c1,c2,k=5,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        c_=c1//2
        self.cv1=Conv(c1,c_,1,1)
        self.cv2=Conv(c_*4,c2,1,1)
        self.m=nn.MaxPool2d(kernel_size=k,stride=1,padding=k//2)
    def forward(self,x):
        x=self.cv1(x)
        y1=self.m(x)
        y2=self.m(y1)
        y3=self.m(y2)
        return self.cv2(torch.cat((x,y1,y2,y3),1))


class concat(nn.Module):
    def __init__(self, d,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d = d;
    def forward(self,x):
        return torch.cat(x,self.d)
        
class Detect(nn.Module):
    export = False
    def __init__(self,nc=80, anchors=(),ch=(),*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nc= nc
        self.no = nc+5
        self.nl = len(anchors)  #number of detection layers
        self.na = len(anchors[0])//2 #number of anctors
        self.grid = [torch.empty(0)  for _ in range(self.nl)]  #init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  #init anchor grid
        self.register_buffer('anchors',torch.tensor(anchors).float().view(self.nl,-1,2)) # shape(nl,na,2)
        self.m=nn.ModuleList(nn.Conv2d(x,self.no*self.na,1) for x in ch)  #output conv
        self.inplace = True
        self.stride = torch.zeros(3)
    def forward(self,x):
        z=[]
        for i in range(self.nl):
            x[i]=self.m[i](x[i])
            bs,_,ny,nx=x[i].shape  #bx(bs,255,20,20) to x(bs,3,20,20,85)
            
            x[i]=x[i].view(bs,self.na,self.no,ny,nx).permute(0,1,3,4,2).contiguous()
            if not self.training:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                
                xy,wh,conf = x[i].sigmoid().split((2,2,self.nc+1),4)
                xy = (xy*2+self.grid[i])*self.stride[i]
                wh = (wh*2)**2*self.anchor_grid[i]
                y=torch.cat((xy,wh,conf),4)
                z.append(y.view(bs,self.na*nx*ny,self.no))
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
    
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') #if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
    
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
        
                
class Net(nn.Module):
    def __init__(self, ch = 3, nc= 6, anchors = None,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)                   #input 640*640
        self.names = [str(i) for i in range(nc)]
        self.inplace = True
        self.model = nn.Sequential(
            *[Conv(3,32,6,2,2),                         # 0  32@320*320
            Conv(32,64,3,2,1),                          # 1  64@160*160
            C3(64,64,1),                                # 2  64@160*160
            Conv(64,128,3,2,1),                         # 3  128@80*80
            C3(128,128,2),                              # 4  128@80*80
            Conv(128,256,3,2,1),                        # 5  256@40*40
            C3(256,256,3),                              # 6  256@40*40
            Conv(256,512,3,2,1),                        # 7  512@20*20
            C3(512,512,1),                              # 8  512@20*20
            SPPF(512, 512,5),                           # 9  512@20*20
            Conv(512,256,1,1),                          #10  256@20*20
            nn.Upsample(None,2,'nearest'),              #11  256@40*40            #12 upsampling 
            Concat(1),                                  #12  cat #11#6  512@40*40
            C3(512,256,1,False),                        #13  256@40*40
            Conv(256,128,1,1),                          #14  128@40*40   
            nn.Upsample(None,2,'nearest'),              #15  128@80*80
            Concat(1),                                  #16 cat#15#4 256*80*8-
            C3(256,128,1,False),                        #17 128@80*80 
            Conv(128,128,3,2,1),                        #18 128@40*40
            Concat(1),                                  #19 #18#14 256@40*40
            C3(256,256,1,False),                        #20 256@40*40
            Conv(256,256,3,2,1),                        #21 256@20*20
            Concat(1),                                  #22 cat#21#10 512@20*20
            C3(512,512,1,False),                        #23 512@20*20 
            Detect(nc=nc, anchors=anchors, ch=[128,256,512]) ]       

        )
        m=self.model[-1]
        m.inplace = self.inplace
        
        forward = lambda x:self.forward(x)
        
        s = 640
        m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1,ch,s,s))])
        m.anchors /= m.stride.view(-1,1,1)
        self.stride = m.stride
        self.anchors = m.anchors
        self.nc = nc
        self.initialize_biases()
        self.initialize_weights()
        self.info()
                      
    def forward(self,x):        
        trace = []
        for i, layer in enumerate(self.model):
            if i==12:
                x = layer([x,trace[1]])
            elif i==16:
                x = layer([x,trace[0]])
            elif i==19:
                x = layer([x,trace[3]])
            elif i==22:
                x = layer([x,trace[2]])
            elif i==24:
                x = layer([trace[4],trace[5],trace[6]])
            else:
                x = layer(x)
            if i in [4,6,10,14,17,20,23]:
                trace.append(x)
        return  x
    def initialize_biases(self,cf=None):
        
        m = self.model[-1]
        for mi,s in zip(m.m,m.stride):
            b = mi.bias.view(m.na,-1)
            b.data[:,4] +=math.log(8/(640/s)**2)
            b.data[:,5:5+m.nc] += math.log(0.6/(m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1),requires_grad=True)
    
    def initialize_weights(self):
        for m in self.modules():
            t=type(m)
            if t is nn.Conv2d:
                pass
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True
    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]
        m.stride = fn(m.stride)
        m.grid = list(map(fn,m.grid))
        m.anchor_grid = list(map(fn,m.anchor_grid))
        return self
    def info(self,verbose=True,img_size=640):
        model_info(self,verbose,img_size)

def model_info(model,verbose,imgsz):
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    
    try:  # FLOPs
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # max stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
        fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'  # 640x640 GFLOPs
    except Exception:
        fs = ''
    for layer in model.modules():
        print(f'{layer}')
    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    LOGGER.info(f'{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}')











        


        
        



if __name__ == "__main__":

    model =  Net()

    # compute_loss = loss（model）



    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()





    im = torch.rand(1, 3, 640, 640).to(device)
    
    
    output = model(im)
    # # print("output",output[0].shape())
    # for layer in mode.:
    #     im = layer(im)
    #     print(layer.__call__.__name__,'output shepe: \t', im.shape)

