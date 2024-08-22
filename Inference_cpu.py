import os
import sys
import cv2
from ultralytics import YOLO
import numpy as np

current_dir=sys.path[0]
os.chdir(current_dir)
# print(os.getcwd())
files = os.listdir(os.curdir)  
print(files)

img_name = 'inf2'
img_path = current_dir+'/'+img_name+'.jpg'
model = YOLO(current_dir+'/best_150000.pt')
# Force the model to run on CPU
model.to('cpu')

img_cv = cv2.imread(img_path)


# Run the model
results = model(img_path)
corner_xy_list = []
bounding_box = []

for result in results:
    corner_xy_list.append(result.keypoints.xy)
    bounding_box.append(result.boxes.xyxy)
        

corner_xy_list = np.asarray(corner_xy_list[0][0])
bounding_box = np.asarray(bounding_box[0][0])


# Annotate image with keypoints
for keypoint_indx, keypoint in enumerate(corner_xy_list):
    cv2.putText(img_cv, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

x_min, y_min, x_max, y_max = bounding_box.astype(int)
cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Draw rectangle with blue color and thickness 2


(h, w) = img_cv.shape[:2]
new_width = 600
aspect_ratio = h / w
new_height = int(new_width * aspect_ratio)
img_cv = cv2.resize(img_cv, (new_width, new_height))

# Display the result
cv2.imshow('image',img_cv)

cv2.waitKey(0)
cv2.destroyAllWindows()
