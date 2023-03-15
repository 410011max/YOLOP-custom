import cv2
import numpy as np
import torch

shapes = (5, 5)
seg_label = np.random.randint(0, 6, size=shapes)
print(seg_label)

# new encoding method
seg = np.zeros((5, shapes[0], shapes[1]))
for i in range(5):
    seg[i] = (seg_label == (i + 1))

bk_da = (seg_label == 0) | (seg_label >= 3)
bk_ll = (seg_label < 3)
bk_da = torch.Tensor(bk_da)
bk_ll = torch.Tensor(bk_ll)
seg = torch.Tensor(seg)

seg_label = torch.stack((bk_da, seg[0], seg[1]), dim=0)
lane_label = torch.stack((bk_ll, seg[2], seg[3], seg[4]), dim=0)


# print(seg_label)
# print(lane_label)