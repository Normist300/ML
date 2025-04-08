import torch
import cv2
import os

print(os.path.abspath("sample_photo.jpeg"))

img = cv2.imread("E:\Code\ML\sample_photo.jpeg")
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_t = torch.from_numpy(RGB_img).type(torch.float32).unsqueeze(0)
img_t