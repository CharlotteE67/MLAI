import numpy as np
import os
import cv2
from PIL import Image

for item in os.listdir('./data/raw/empty'):
    img = Image.open('./data/raw/empty/'+item)
    img_arr = np.array(img, np.uint8)
    if img_arr.shape[-1] == 4:
        img_arr = img_arr[::, ::, :3]
        img = Image.fromarray(img_arr)
        print(item, img_arr.shape, 'finished!')
    img.save("./data/processed/empty/"+item+".png")

for item in os.listdir('./data/raw/occupied'):
    img = Image.open('./data/raw/occupied/'+item)
    img_arr = np.array(img, np.uint8)
    if img_arr.shape[-1] == 4:
        img_arr = img_arr[::, ::, :3]
        img = Image.fromarray(img_arr)
        print(item, img_arr.shape, 'finished!')
    img.save("./data/processed/occupied/"+item+".png")

# img = Image.open('./data/empty/e163')
# # img.show()

# # matrix = ( 1, 0, 0, 0,
# #            0, 0, 0, 0,
# #            0, 0, 1, 0)
# # img = img.convert("RGB", matrix)

# img_arr = np.array(img, np.uint8)
# print(img_arr.shape)
# print(img_arr[-1])
# img_arr = img_arr[::, ::, :3]
# print(img_arr.shape)
# print(img_arr[-1])

# pic = cv2.imread('./data/occupied/o1')
# print(pic.shape)
# for item in os.listdir('./data/occupied'):
#     pic = cv2.imread('./data/occupied/'+item)
#     print(pic.shape)
# empty_files = os.listdir('./data/empty')
# occupied_files = os.listdir('./data/occupied')

# empty = []
# for fname in empty_files:
#     file = open('./data/empty/'+fname, 'r')

# print(empty_files)

# for item in os.listdir('./data/empty'):
#     img = Image.open('./data/empty/'+item)
#     # img.show()
#     img_arr = np.array(img, np.uint8)
#     if img_arr.shape[-1] == 4:
#         print(item, img_arr.shape)

# for item in os.listdir('./data/occupied'):
#     img = Image.open('./data/occupied/'+item)
#     # img.show()
#     img_arr = np.array(img, np.uint8)
#     if img_arr.shape[-1] == 4:
#         print(item, img_arr.shape)