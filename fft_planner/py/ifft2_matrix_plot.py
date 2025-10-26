# coding=utf-8

import json
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
# 读取 JSON
with open("/home/jack/work/2004/ifft2.json", "r") as f:
    data = json.load(f)

rows, cols = data["rows"], data["cols"]

# 转换成 numpy 复数矩阵
mat = np.zeros((rows, cols), dtype=np.complex128)
for i in range(rows):
    for j in range(cols):
        real, imag = data["data"][i][j]
        mat[i, j] = real + 1j * imag

# 取绝对值，相当于 np.abs(np.fft.ifft2(...))
filtered_image = np.abs(mat)

# 截取原图大小（假设和 self.shape 一致）
h, w = 128, 128   # 你自己的图像 shape
filtered_image = filtered_image[:32, :128]
print(filtered_image / 255.0)
plt.imsave("/home/jack/filtered_image_c.png", filtered_image / 255 * 10.0, cmap="gray")

plt.imshow(filtered_image / 255.0, cmap="gray")
plt.show()
