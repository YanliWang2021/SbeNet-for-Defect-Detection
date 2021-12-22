import torch.nn as nn
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt

plt.switch_backend('agg')
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find('ConvTranspose2d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.constant_(m.weight.data, 0.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

weights_init = weights_init_kaiming
            
def side_branch(input_channels,factor):
    #x = nn.Conv2d(3, 3, 1, stride=1, padding=1)    #kernel_size = (2*factor, 2*factor)
    x = nn.ConvTranspose2d(input_channels, 1  ,kernel_size=factor, stride=factor, padding=0,  output_padding=0, groups=1, bias=True, dilation=1)#
    return x

def draw_features(width,height,x,savename):
    fig = plt.figure(figsize=(10, 6))
    for i in range(width*height):
        ax=plt.subplot(height,width, i + 1)
        plt.xticks([])
        plt.yticks([])
        img = x[0, i, :, :]
        ax.imshow(img)
        ax.set_ylabel('{}'.format(i),rotation=0)
        ax.yaxis.set_label_coords(-0.2, 0.5) 
               
    fig.savefig(savename, dpi=100)
    fig.clf()
     
    plt.close()

nCnl=16

class SegmentNet(nn.Module):
    def __init__(self, in_channels=1, init_weights=True):
        super(SegmentNet, self).__init__()
        self.count = 0
        
        self.layer1 = nn.Sequential(
                            nn.Conv2d(in_channels, nCnl, 3, stride=1, padding=1),
                            nn.BatchNorm2d(nCnl),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(nCnl, nCnl, 3, stride=1, padding=1),
                            nn.BatchNorm2d(nCnl),
                            nn.ReLU(inplace=True),
                            #nn.MaxPool2d(2)  
                        )
        self.mpool2 = nn.MaxPool2d(2)  
        self.rconv1 = side_branch(nCnl,1)

        self.layer2 = nn.Sequential(
                            nn.Conv2d(nCnl, nCnl*2, 3, stride=1, padding=1),
                            nn.BatchNorm2d(nCnl*2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(nCnl*2, nCnl*2, 3, stride=1, padding=1),
                            nn.BatchNorm2d(nCnl*2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(nCnl*2, nCnl*2, 3, stride=1, padding=1),
                            nn.BatchNorm2d(nCnl*2),
                            nn.ReLU(inplace=True),
                            #nn.MaxPool2d(2)
                        )
        self.rconv2 = side_branch(nCnl*2,2)

        self.layer3 = nn.Sequential(
                            nn.Conv2d(nCnl*2, nCnl*2, 7, stride=1, padding=3),
                            nn.BatchNorm2d(nCnl*2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(nCnl*2, nCnl*2, 7, stride=1, padding=3),
                            nn.BatchNorm2d(nCnl*2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(nCnl*2, nCnl*2, 7, stride=1, padding=3),
                            nn.BatchNorm2d(nCnl*2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(nCnl*2, nCnl*2, 7, stride=1, padding=3),
                            nn.BatchNorm2d(nCnl*2),
                            nn.ReLU(inplace=True),
                            #nn.MaxPool2d(2)
                        )
        self.rconv3 = side_branch(nCnl*2,4)
        
        self.layer4 = nn.Sequential(
                            nn.Conv2d(nCnl*2, 1024, 15, stride=1, padding=7),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(inplace=True)
                        )
        self.rconv4 = side_branch(1024,8)

        self.layer5 = nn.Sequential(
                            nn.Conv2d(4, 1, 1),
                            nn.ReLU(inplace=True)
                        )

        if init_weights == True:
            self.layer1.apply(weights_init)
            self.layer2.apply(weights_init)
            self.layer3.apply(weights_init)
            self.layer4.apply(weights_init)
            pass
    def forward(self, x):
        x1 = self.layer1(x)
        n1 = self.rconv1(x1)
        x1 = self.mpool2(x1)

        x2 = self.layer2(x1)
        n2 = self.rconv2(x2)
        x2 = self.mpool2(x2)

        x3 = self.layer3(x2)
        n3 = self.rconv3(x3)
        x3 = self.mpool2(x3)
        
        x4 = self.layer4(x3)
        n4 = self.rconv4(x4)

        out = self.layer5(torch.cat((n1,n2,n3,n4),1))

        self.count += 1

        return {"seg":out}

class DecisionNet(nn.Module):
    
    def __init__(self, init_weights=True):
        super(DecisionNet, self).__init__()

        self.layer1 = nn.Sequential(
                            nn.Conv2d(1,512,1,stride=1),
                            nn.MaxPool2d(2),
                           
                            nn.Conv2d(512, 128, 5, stride=1, padding=2),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2),
                            
                            nn.Conv2d(128, 16, 5, stride=1, padding=2),
                            nn.BatchNorm2d(16),
                            nn.ReLU(inplace=True),
                            
                            nn.Conv2d(16, 32, 5, stride=1, padding=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True)
                        )

        self.fc =  nn.Sequential(
                            nn.Linear(64, 1, bias=False),
                            nn.Sigmoid()
                        )

        if init_weights == True:
            pass

    def forward(self,s):
        x1 = self.layer1(s)
        x2 = x1.view(x1.size(0), x1.size(1), -1)
        x_max, x_max_idx = torch.max(x2, dim=2)
        x_avg = torch.mean(x2, dim=2)
        y = torch.cat((x_max, x_avg), 1)
        y = y.view(y.size(0), -1)
        return self.fc(y)




