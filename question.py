
import cv2
import numpy as np

# 假设contour1和contour2为两个轮廓
contour1 = np.array([[[10, 10]], [[100, 10]], [[100, 100]], [[10, 100]]])
contour2 = np.array([[[20, 20]], [[90, 20]], [[90, 90]], [[20, 90]]])

# 判断contour1是否包含在contour2内部
is_contour1_inside_contour2 = all(cv2.pointPolygonTest(contour2, tuple(point[0]), False) >= 0 for point in contour1)

if is_contour1_inside_contour2:
    print("contour1包含在contour2内部")
else:
    print("contour1不完全包含在contour2内部")