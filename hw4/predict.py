import torch
from cv2 import cv2
from LeNet5 import LeNet5

# # choose the best model
# net = LeNet5()
# net.load_state_dict(torch.load('./models/mnist_0.988.pkl'))

pic = cv2.imread('./1.png', cv2.IMREAD_COLOR)
pic_n = cv2.resize(pic, (28, 28))
cv2.imwrite('./1_28*28.png', pic_n)
