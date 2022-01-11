import matplotlib.image as mpimg
# rgb_image = mpimg.imread('dataset/test/003/000/imgRGB_003_000_000.png')
# #print(rgb_image)
#
import matplotlib.pyplot as plt
#
# plt.imshow(rgb_image)
# plt.show()
# print(rgb_image.shape)
import cv2
Rgb_image =mpimg.imread(r'my_examples_1/fk_ali.jpg')
plt.imshow(Rgb_image)
plt.show()
print(Rgb_image.shape)
print(Rgb_image.dtype)

depth_image =cv2.imread(r'out_my_examples_1/out_my_examples_1/fk_ali.jpg')
depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
plt.imshow(depth_image, cmap='gray')

from PIL import Image
new_image = cv2.resize(depth_image, (640, 480))
cv2.imwrite('out_my_examples_1/out_my_examples_1/fk_ali_11.jpg', new_image)
print(depth_image.size) # Output: (1200, 776)
print(new_image.shape) # Output: (400, 400)
# plt.show()
# print(depth_image.shape)
# print(depth_image.dtype)

# Rgb_image =mpimg.imread(r'MiDaS-master//131.png')
# plt.imshow(Rgb_image, cmap='plasma')
# plt.show()
# print(Rgb_image.shape)
# print(Rgb_image.dtype)
#
#
# Rgb_image =mpimg.imread(r'DPT-main/output_monodepth/14.png')
# plt.imshow(Rgb_image, cmap='plasma')
# plt.show()
# print(Rgb_image.shape)
# print(Rgb_image.dtype)
#
# Rgb_image =mpimg.imread(r'demo_results/14.jpg')
# plt.imshow(Rgb_image, cmap='plasma')
# plt.show()
# print(Rgb_image.shape)
# print(Rgb_image.dtype)



# import pandas as pd
# import numpy as np
# from pyntcloud import PyntCloud
# from PIL import Image
#
# colourImg = Image.open(r"DPT-main/input/fk_ali.jpg")
# colourPixels = colourImg.convert("RGB")
#
# colourArray  = np.array(colourPixels.getdata()).reshape((colourImg.height, colourImg.width) + (3,))
# indicesArray = np.moveaxis(np.indices((colourImg.height, colourImg.width)), 0, 2)
# imageArray   = np.dstack((indicesArray, colourArray)).reshape((-1,5))
# df = pd.DataFrame(imageArray, columns=["x", "y", "red","green","blue"])
#
#
# depthImg = Image.open(r'DPT-main/output_monodepth/fk_ali.png').convert('L')
# depthArray = np.array(depthImg.getdata())
# df.insert(loc=2, column='z', value=depthArray)
#
# df[['x','y','z']] = df[['x','y','z']].astype(float)
# df[['red','green','blue']] = df[['red','green','blue']].astype(np.uint)
# cloud = PyntCloud(df)
# cloud.plot()

# Rgb_image =mpimg.imread(r'demo_results/14.jpg')
# plt.imshow(Rgb_image, cmap='plasma')
# plt.show()
# print(Rgb_image.shape)
# print(Rgb_image.dtype)

# image = cv2.imread(r'D:\Thermal_work\Laplace_depth\out_my_examples\out_my_examples\imgThermal_003_000_007.png')
# plt.imshow(image[:,:,1], cmap='plasma')
# plt.show()
# depth_image = mpimg.imread(r'H:\Faisal_Data\pandora_data\01\01\base_1_ID01\DEPTH\000000_DEPTH.png')
# plt.imshow(depth_image)
# plt.show()
# print(depth_image.shape)
# print(Rgb_image.dtype)
# ir_image = mpimg.imread('dataset/test/003/000/imgIR_003_000_000.png')
# plt.imshow(ir_image)
# plt.show()
# print(ir_image.shape)
# th_image = mpimg.imread('demo_results/imgThermal_003_000_000.png')
# plt.imshow(th_image)
# plt.show()
# print(th_image.shape)

# from PIL import Image
# image = Image.open(r'out_my_examples/out_my_examples/test2_new.jpg')
# new_image = image.resize((640, 480))
# new_image.save(r'out_my_examples/out_my_examples/test2_new.jpg')
#
# print(image.size) # Output: (1200, 776)
# print(new_image.size) # Output: (400, 400)
# #
# print()
