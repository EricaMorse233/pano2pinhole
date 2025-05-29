import cv2
import numpy as np
import os

def equirectangular_to_perspective(equirect_img, fov, pitch, yaw, width, height):
    """
    将 equirectangular 全景图转化为透视图。
    :param equirect_img: 输入的 equirectangular 图像
    :param fov: 视场角(Field of View), 以度为单位
    :param pitch: 透视图中心的俯仰角（单位：度）
    :param yaw: 透视图中心的偏航角（单位：度）
    :param width: 输出图像宽度
    :param height: 输出图像高度
    :return: 转换后的透视图
    """
    h, w, _ = equirect_img.shape
    fov = np.deg2rad(fov)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    # 输出透视图的网格
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - width / 2) / (width / 2)  # 归一化到[-1, 1]
    y = (y - height / 2) / (height / 2)

    # 透视坐标到球面坐标
    z = 1 / np.tan(fov / 2)
    x_sphere = z * x
    y_sphere = z * y
    norm = np.sqrt(x_sphere**2 + y_sphere**2 + z**2)

    x_sphere /= norm
    y_sphere /= norm
    z /= norm

    # 旋转到指定的视角
    rot_matrix = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ]).dot(
        np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
    )

    coords = np.dot(rot_matrix, np.stack([x_sphere.flatten(), y_sphere.flatten(), z.flatten()]))

    lon = np.arctan2(coords[0], coords[2])  # 经度
    lat = np.arcsin(coords[1])              # 纬度

    # 球面坐标转像素坐标
    u = (lon / np.pi + 1) / 2 * w
    v = (lat / (np.pi / 2) + 1) / 2 * h

    u = u.reshape(height, width).astype(np.float32)
    v = v.reshape(height, width).astype(np.float32)

    # 映射像素值
    perspective_img = cv2.remap(equirect_img, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return perspective_img

def split_equirectangular_to_perspectives(input_image_path, output_dir, fov=90, yaw_start=0,step=(50, 60), width=1500, height=1500):
    """
    将 equirectangular 全景图分割成多个透视图。
    :param input_image_path: 输入的全景图路径
    :param output_dir: 输出文件夹
    :param fov: 每个分块的视场角
    :param step: 俯仰角和偏航角的步进 (pitch_step, yaw_step)
    :param width: 每个分块的宽度
    :param height: 每个分块的高度
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(input_image_path)
    if img is None:
        raise ValueError(f"Cannot read input image at {input_image_path}")

    input_filename = os.path.basename(input_image_path)
    rows, cols = 2, 6
    pitch_step = step[0]
    yaw_step = step[1]
    top_pitch = 20  # 图像中心对应的 pitch
    num = 0
    yaw = yaw_start

    for row in range(rows):
        for col in range(cols):
            pitch = top_pitch - pitch_step * row  
            yaw = -180 + yaw_step * col  
            perspective_img = equirectangular_to_perspective(img, fov, pitch, yaw, width, height)

            output_filename = f"{input_filename}_{num}.jpg"
            cv2.imwrite(os.path.join(output_dir, output_filename), perspective_img)
            num += 1

def process_directory(input_dir, output_dir, fov=90, step=(20, 60), width=1500, height=1500):
    """
    在给定的文件夹中读取所有 JPG 图像，并对每张图像执行透视分割。
    :param input_dir: 包含全景图的文件夹
    :param output_dir: 输出透视图文件夹
    :param fov: 视场角
    :param step: (pitch_step, yaw_step)
    :param width: 输出透视图的宽度
    :param height: 输出透视图的高度
    """
    
    # 遍历文件夹中的所有文件
    yaw_step = 20
    yaw_start = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg"):
            # 生成输入文件的完整路径
        
            input_image_path = os.path.join(input_dir, filename)
            print(f"Processing: {input_image_path}")
            if yaw_start < 360:
                yaw_start += yaw_step
            else:
                yaw_start = yaw_start -360 + yaw_step
            print(f"Yaw start: {yaw_start}")
            split_equirectangular_to_perspectives(input_image_path,
                                                  output_dir,
                                                  fov=fov,
                                                  yaw_start=yaw_start,
                                                  step=step,
                                                  width=width,
                                                  height=height)

if __name__ == "__main__":
    input_directory = "Image_20241228_124434"
    time_stamp = input_directory.split("_")[1:]
    output_directory = f"SplitTest_{time_stamp[0]}_{time_stamp[1]}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if not os.path.exists(input_directory):
        raise ValueError(f"Directory {input_directory} does not exist")
    process_directory(input_directory, output_directory, fov=90, step=(50, 60), width=1500, height=1500)
