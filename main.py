# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenCV and Numpy Point cloud Software Renderer

This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.

Usage:
------
Mouse:
    Drag with left button to rotate around pivot (thick small axes),
    with right button to translate and the wheel to zoom.

Keyboard:
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from pyransac3d import Cuboid
from scipy import stats
import os.path
import argparse
from ultralytics import YOLO
import sys

current_dir=sys.path[0]
os.chdir(current_dir)
# print(os.getcwd())
# files = os.listdir(os.curdir)  
# print(files)



class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


############################################################################################################
################################################## CONFIG ##################################################
############################################################################################################

use_rosbag = True

box_color = "red"

color_to_hsv_range = {
    "pink": [((150, 100, 120), (179, 255, 255))], # LOWER HSV - UPPER HSV
    "red": [((0, 100, 20), (10, 255, 255)), ((160, 100, 20), (180, 255, 255))], # LOWER HSV1, LOWER HSV2, UPPER HSV1, UPPER HSV2
    "green": [((40, 100, 100), (80, 255, 255))],
    "blue": [((100, 100, 100), (140, 255, 255))],
    "gray": [((85, 25, 240), (179, 255, 255))],
}




fit_interval_seconds = 0.2  # ransac fit algorithm execution rate in seconds
Treshold = 0.05
MaxIter = 1000
extracted_verts_limit = 50

# Object dimensions in cm (L x W x H)
length = 0.20
width = 0.15
height = 0.135

object_centroid = np.array([length*0.5, width*0.5, height*0.5])# This is [0 0 0] in object system 

# 3D coordinates of the box corners in the object coordinate system
object_points = np.array([
    [-0.5*length, -0.5*width, -0.5*height],         # Point 0: Corner at the origin
    [0.5*length, -0.5*width, -0.5*height],    # Point 1: Along the x-axis (length)
    [0.5*length, 0.5*width, -0.5*height],# Point 2: Along the x and y-axis (length and width)
    [-0.5*length, 0.5*width, -0.5*height],     # Point 3: Along the y-axis (width)
    [-0.5*length, -0.5*width, 0.5*height],    # Point 4: Along the z-axis (height)
    [0.5*length, -0.5*width, 0.5*height],# Point 5: Along the x and z-axis (length and height)
    [0.5*length, 0.5*width, 0.5*height], # Point 6: Along the x, y, and z-axis (length, width, and height)
    [-0.5*length, 0.5*width, 0.5*height] # Point 7: Along the y and z-axis (width and height)
], dtype=np.float32)

object_centroid = np.mean(object_points, axis=0) # centroid in box coordinate system



############################################################################################################
############################################################################################################
############################################################################################################

last_fit_time = time.time()
best_inliers = np.array([])
model = YOLO(current_dir+'/best_150000.pt')
model.to('cpu')

state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

### THE FOLLOWING LINES ARE TO USE A REALSENSE INSTEAD OF ROSBAG!

if not use_rosbag:
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

### THE FOLLOWING LINES ARE TO USE A ROSBAG INSTEAD OF REALSENSE!

if use_rosbag:
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
    # Parse the command line arguments to an object
    args = parser.parse_args()
    # Safety if no parameter have been given
    if not args.input:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()
    # Check if the given file have bag extension
    if os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()

    rs.config.enable_device_from_file(config, args.input)


    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()


# Function to convert HSV to BGR
def hsv_to_bgr(hsv):
    # OpenCV expects the HSV value as an array of shape (1, 1, 3) and dtype uint8
    hsv_array = np.uint8([[hsv]])
    bgr_array = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2BGR)
    return tuple(bgr_array[0][0])

# Function to get the lower and upper BGR values for a specified box color range
def get_bgr_values(color_name):
    hsv_range = color_to_hsv_range.get(color_name)
    lower_hsv, upper_hsv = hsv_range
    lower_bgr = hsv_to_bgr(lower_hsv)
    upper_bgr = hsv_to_bgr(upper_hsv)
    return lower_bgr, upper_bgr

def get_hsv_values(color_name):
    return color_to_hsv_range.get(color_name)

def calculate_axis_endpoints(centroid, axis, scale=0.1):
    return centroid + axis * scale


def draw_green_line(out,end_point):
    origin = np.array([0, 0, 0], dtype=np.float32)
    # end_point = np.array([0, 2, 0], dtype=np.float32)  # 2 meters away perpendicularly

    p0 = project(view(origin.reshape(-1, 3)))[0]
    p1 = project(view(end_point.reshape(-1, 3)))[0]

    if np.isnan(p0).any() or np.isnan(p1).any():
        return

    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))

    cv2.line(out, p0, p1, (0, 255, 0), 2, cv2.LINE_AA)  # Green color


def draw_axes(out, centroid, x_axis, y_axis, z_axis):# Draws the axes of the box
    scale = 0.5  # Adjust scale for visual representation
    x_end = calculate_axis_endpoints(centroid, x_axis, scale)
    y_end = calculate_axis_endpoints(centroid, y_axis, scale)
    z_end = calculate_axis_endpoints(centroid, z_axis, scale)

    points_3d = np.array([centroid, x_end, y_end, z_end])
    points_2d = project(view(points_3d)).astype(np.int32)

    centroid_2d = tuple(points_2d[0])
    x_end_2d = tuple(points_2d[1])
    y_end_2d = tuple(points_2d[2])
    z_end_2d = tuple(points_2d[3])

    cv2.line(out, centroid_2d, x_end_2d, (0, 0, 255), 2)  # Red for x-axis
    cv2.line(out, centroid_2d, y_end_2d, (0, 255, 0), 2)  # Green for y-axis
    cv2.line(out, centroid_2d, z_end_2d, (255, 0, 0), 2)  # Blue for z-axis

def remove_outliers_z_score(data, threshold=2):
    """
    Removes outliers from the data based on Z-score.
    
    Parameters:
    - data (ndarray): The input data.
    - threshold (float): The Z-score threshold for identifying outliers.

    Returns:
    - ndarray: Data without outliers.
    """
    z_scores = np.abs(stats.zscore(data))
    filtered_data = data[(z_scores < threshold).all(axis=1)]
    return filtered_data


def get_extracted_verts(colored_verts):# Returns the extracted 3D vertices based on hsv mask
    hsv_ranges = get_hsv_values(box_color)
    bgr_colors = colored_verts[:, 3:6].astype(np.uint8)
    hsv_colors = cv2.cvtColor(bgr_colors.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    # Initialize a mask with zeros (same shape as the number of vertices)
    final_mask = np.zeros(len(hsv_colors), dtype=bool)
    
    # Apply masks for each range
    for lower_hsv, upper_hsv in hsv_ranges:
        mask = (hsv_colors[:, 0] >= lower_hsv[0]) & (hsv_colors[:, 0] <= upper_hsv[0]) & \
               (hsv_colors[:, 1] >= lower_hsv[1]) & (hsv_colors[:, 1] <= upper_hsv[1]) & \
               (hsv_colors[:, 2] >= lower_hsv[2]) & (hsv_colors[:, 2] <= upper_hsv[2])
        final_mask |= mask  # Combine the masks using logical OR
    
    extracted_verts = colored_verts[final_mask, :3]
    return extracted_verts


def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


def consistent_axes_with_camera(normal1, normal2, normal3, camera_axes):
    '''
    This function selects which surcafe normal correspond to which box axis based on the biggest
    dot product of each surface normal with the camera axis.

    '''
    normals = [normal1, normal2, normal3]
    
    # Camera axes
    camera_x = camera_axes[0]
    camera_y = camera_axes[1]
    camera_z = camera_axes[2]
    
    # Calculate dot products for the x axis
    dot_products_x = np.array([np.dot(camera_x, normal1), np.dot(camera_x, normal2), np.dot(camera_x, normal3)],)
    sorted_x_index = np.argmax(np.abs(dot_products_x))# Select x axis of the box as the most aligned normal with the camera x axis
    new_x_axis = normals[sorted_x_index]
    if dot_products_x[sorted_x_index] < 0:
        new_x_axis = - new_x_axis
    
    normals = np.delete(normals, sorted_x_index, axis=0)# Delete the selection out of the normals list

    # Calculate dot products for the y axis
    dot_products_y = np.array([np.dot(camera_y, normals[0]), np.dot(camera_y, normals[1])])
    sorted_y_index = np.argmax(np.abs(dot_products_y))
    new_y_axis = normals[sorted_y_index]
    if dot_products_y[sorted_y_index] < 0:
        new_y_axis = - new_y_axis

    new_z_axis = np.cross(new_x_axis,new_y_axis)

    return new_x_axis, new_y_axis, new_z_axis

def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)

def visualize_extracted_cloud(out, verts, painter=True):
    '''
    This function is used to see which part of the point cloud is extracted using the hsv color mask
    '''
    # lower_bgr, _ = get_bgr_values("pink")
    white = (255,255,255)
    color = white
    if verts.size == 0:
        return

    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5 ** state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    out[i[m], j[m]] = color

def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]



out = np.empty((h, w, 3), dtype=np.uint8)

while True:
    # Grab camera data
    if not state.paused:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # img_cv = color_image

        
        # Run the model
        results = model(color_image)
        corner_xy_list = []
        bounding_box = []

        for result in results:
            corner_xy_list.append(result.keypoints.xy)
            bounding_box.append(result.boxes.xyxy)
                

        corner_xy_list = np.asarray(corner_xy_list[0][0])
        bounding_box = np.asarray(bounding_box[0][0])
        # Detected 2D image points (from your corner_xy_list)
        image_points = corner_xy_list.astype(np.float32)


        # Annotate image with keypoints
        for keypoint_indx, keypoint in enumerate(corner_xy_list):
            cv2.putText(color_image, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        x_min, y_min, x_max, y_max = bounding_box.astype(int)
        cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Draw rectangle with blue color and thickness 2
        

        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        if state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics() 

                
        # Define the camera matrix and distortion coefficients using the real-time intrinsics
        camera_matrix = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.array(color_intrinsics.coeffs)

        # Solve for rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,     # 3D object points
            image_points,      # 2D image points
            camera_matrix,     # Camera intrinsic matrix
            dist_coeffs        # Distortion coefficients
        )

        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Print the results
        print("Rotation Matrix:\n", rotation_matrix)
        print("Translation Vector:\n", translation_vector)


        axis_length = 10  # You can adjust this based on the scale of your object

        # Define the 3D points for the axes (in the object's coordinate system)
        axes_points_3D = np.float32([
            [0, 0, 0],  # Origin of the axes
            [axis_length, 0, 0],  # X-axis
            [0, axis_length, 0],  # Y-axis
            [0, 0, axis_length]   # Z-axis
        ]).reshape(-1, 3)

        # Project the 3D points onto the 2D image plane
        axes_points_2D, _ = cv2.projectPoints(
            axes_points_3D,        # 3D points of the axes
            rotation_vector,       # Rotation vector from solvePnP
            translation_vector,    # Translation vector from solvePnP
            camera_matrix,         # Camera intrinsic matrix
            dist_coeffs            # Distortion coefficients
        )

        # Extract the 2D coordinates of the projected points
        # Convert to integer tuples
        origin_2D = tuple(map(int, axes_points_2D[0].ravel()))
        x_axis_2D = tuple(map(int, axes_points_2D[1].ravel()))
        y_axis_2D = tuple(map(int, axes_points_2D[2].ravel()))
        z_axis_2D = tuple(map(int, axes_points_2D[3].ravel()))

        resized_color_image = cv2.resize(color_image, (out.shape[1], out.shape[0]))

        cv2.imshow('color_image',resized_color_image)

        current_time = time.time()
        if current_time - last_fit_time >= fit_interval_seconds:
            last_fit_time = current_time

            # Vectorized extraction of color information from the color image
            u = texcoords[:, 0]
            v = texcoords[:, 1]
            u = (u * color_image.shape[1]).astype(np.int32)
            v = (v * color_image.shape[0]).astype(np.int32)
            u = np.clip(u, 0, color_image.shape[1] - 1)
            v = np.clip(v, 0, color_image.shape[0] - 1)

            colors = color_image[v, u]
            colored_verts = np.hstack((verts, colors))

            extracted_verts = get_extracted_verts(colored_verts)
            centroid_extracted_verts = np.mean(extracted_verts, axis=0)
            cleaned_extracted_verts = remove_outliers_z_score(extracted_verts)



    # Render
    now = time.time()

    out.fill(0)

    grid(out, (0, 0.5, 1), size=1, n=10)
    frustum(out, depth_intrinsics)
    axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

    if not state.scale or out.shape[:2] == (h, w):
        pointcloud(out, verts, texcoords, color_source)
        if cleaned_extracted_verts.size > 0:  # Ensure there are vertices to render
            if best_inliers.size > 0:
                visualize_extracted_cloud(out, inliers)
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, texcoords, color_source)
        if cleaned_extracted_verts.size > 0:  # Ensure there are vertices to render
            if best_inliers.size > 0:
                visualize_extracted_cloud(tmp, inliers)
        tmp = cv2.resize(
            tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)



    if any(state.mouse_btns):
        axes(out, view(state.pivot), state.rotation, thickness=4)

    # draw_green_line(out,translation_vector)
    centroid = translation_vector.ravel()  # Use the translation vector as the centroid
    x_axis_new = rotation_matrix[:, 0]  # The first column of the rotation matrix
    y_axis_new = rotation_matrix[:, 1]  # The second column of the rotation matrix
    z_axis_new = rotation_matrix[:, 2]  # The third column of the rotation matrix
    draw_axes(out, centroid, x_axis_new, y_axis_new, z_axis_new)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)

    if key == ord("r"):
        state.reset()

    if key == ord("p"):
        state.paused ^= True

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("z"):
        state.scale ^= True

    if key == ord("c"):
        state.color ^= True

    if key == ord("s"):
        cv2.imwrite('./out.png', out)

    if key == ord("e"):
        points.export_to_ply('./out.ply', mapped_frame)

    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

# Stop streaming
pipeline.stop()
