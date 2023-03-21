import torch
from torchviz import make_dot
from lib.models_yolov7 import get_net_yolov7

# YOLOP-v2
x = torch.randn((1,3, 384, 640))
model = torch.load("yolopv2.pt").cpu()
for param in model.parameters():
    param.requires_grad=True
#print(model)
output = model(x)
#print(type(output[0][1]))
#print(len(output[0][1]))
    
make_dot(torch.cat([output[0][0][0].flatten(), output[0][1][0].flatten(), output[0][1][1].flatten(), output[0][1][2].flatten(),
                    output[0][0][1].flatten(), output[1].flatten(), output[2].flatten()]), params=dict(model.named_parameters())).render("YoloPV2", format="svg")

# YOLOP-v2 (ours)
x = torch.randn((1,3, 384, 640))
# please change the path here
model = get_net_yolov7("/home/shadowpa0327/research/Aidea_contest/YOLOP-custom/lib/models_yolov7/yolovPP.yaml")
for param in model.parameters():
    param.requires_grad=True
    
output = model(x)    
make_dot(torch.cat([output[0][0][0].flatten(), output[0][1][0].flatten(), output[0][1][1].flatten(), output[0][1][2].flatten(),
                    output[0][0][1].flatten(), output[1].flatten(), output[2].flatten()]), params=dict(model.named_parameters())).render("YoloPV2(outs)", format="svg")
#print(output[1].shape)