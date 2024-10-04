# box_pose_estimation
Pose estimation for a specific box.

A YOLOv8 keypoint detection model was trained over the corners of the box. Then using PNP algorithm, the 6D box pose was calculated.

![](https://github.com/ozerbar/box_pose_estimation/blob/main/GIF.gif)


# RUN THE MODEL USING
```
python3 main.py -i bagfile.bag
```
