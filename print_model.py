import torch



model  = torch.load('yolopv2.pt')
print(model)

example_forward_input = torch.randn((1, 3, 256, 256))
module = torch.jit.trace(model, example_forward_input)
print(module)