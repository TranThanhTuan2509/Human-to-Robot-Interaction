from transforms3d import euler
import numpy as np
import xml.etree.ElementTree as ET
import pybullet
from config import *

# def get_top_surface_corners(object_id, margin=0.009):
#     aabb_min, aabb_max = p.getAABB(object_id)
#     aabb_min = np.array(aabb_min)
#     aabb_max = np.array(aabb_max)
#     top_z = aabb_max[2] - 0.001
#
#     x_min = aabb_min[0] + margin
#     x_max = aabb_max[0] - margin
#     y_min = aabb_min[1] + margin
#     y_max = aabb_max[1] - margin
#
#     p0 = np.array([x_min, y_min, top_z])  # bottom-left
#     p1 = np.array([x_max, y_min, top_z])  # bottom-right
#     p2 = np.array([x_max, y_max, top_z])  # top-right
#     p3 = np.array([x_min, y_max, top_z])  # top-left
#
#     return p0, p1, p2, p3
#
# def sample_point_on_surface(p0, p1, p2, p3):
#     u, v = np.random.rand(), np.random.rand()
#     point = (
#             (1 - u) * (1 - v) * p0 +
#             u * (1 - v) * p1 +
#             u * v * p2 +
#             (1 - u) * v * p3
#     )
#     return point, u, v

# randomly sampled arbitrary contacting points on horizontal surface
def sample_point_on_top_of_arbitrary(object_id, margin=0.01, max_tries=10, seed=None):
    """
    Ray‐casts down through a randomly chosen (x,y) inside the object's AABB
    (shrunk by 'margin') to find a point on its top surface.
    Returns:
      sampled_world_point (3,)
    Raises:
      RuntimeError if no hit after max_tries.
    """
    np.random.seed(seed)
    aabb_min, aabb_max = pybullet.getAABB(object_id)
    amin = np.array(aabb_min)
    amax = np.array(aabb_max)
    # shrink the XY region by margin
    x_min, y_min = amin[0] + margin, amin[1] + margin
    x_max, y_max = amax[0] - margin, amax[1] - margin

    for _ in range(max_tries):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        # cast from just above the top down past the bottom
        start = [x, y, amax[2] + 0.05]
        end = [x, y, amin[2] - 0.05]
        hits = pybullet.rayTest(start, end)
        # (hitObjectId, hitLinkIndex, hitFraction, hitPosition, hitNormal)
        obj_hit, _, hit_pos, _ = hits[0][0], hits[0][2], hits[0][3], hits[0][4]
        if obj_hit == object_id:
            return np.array(hit_pos)
    raise RuntimeError(f"Could not sample top surface after {max_tries} tries")

def world_to_local(point_world, object_id):
    """Convert a world‐frame point into the object’s local frame."""
    pos, orn = pybullet.getBasePositionAndOrientation(object_id)
    R = np.array(pybullet.getMatrixFromQuaternion(orn)).reshape(3, 3)
    return R.T.dot(point_world - np.array(pos))

def local_to_world(point_local, object_id):
    """Convert a local‐frame point back into world‐coordinates."""
    pos, orn = pybullet.getBasePositionAndOrientation(object_id)
    R = np.array(pybullet.getMatrixFromQuaternion(orn)).reshape(3, 3)
    return R.dot(point_local) + np.array(pos)

def read_config_from_node(root_node, parent_name, child_name, dtype=int):
    # find parent
    parent_node = root_node.find(parent_name)
    if parent_node is None:
        quit("Parent %s not found" % parent_name)

    # get child data
    child_data = parent_node.get(child_name)
    if child_data is None:
        quit("Child %s not found" % child_name)

    config_val = np.array(child_data.split(), dtype=dtype)
    return config_val


def get_config_root_node(config_file_name=None, config_file_data=None):
    # get root
    if config_file_data is None:
        with open(config_file_name) as config_file_content:
            config = ET.parse(config_file_content)
        root_node = config.getroot()
    else:
        root_node = ET.fromstring(config_file_data)

    # get root data
    root_data = root_node.get("name")
    assert isinstance(root_data, str)
    root_name = np.array(root_data.split(), dtype=str)

    return root_node, root_name

def eulerXYZ_to_quatXYZW(rotation):  # pylint: disable=invalid-name
  """Abstraction for converting from a 3-parameter rotation to quaterion.

  This will help us easily switch which rotation parameterization we use.
  Quaternion should be in xyzw order for pybullet.

  Args:
    rotation: a 3-parameter rotation, in xyz order tuple of 3 floats

  Returns:
    quaternion, in xyzw order, tuple of 4 floats
  """
  euler_zxy = (rotation[2], rotation[0], rotation[1])
  quaternion_wxyz = euler.euler2quat(*euler_zxy, axes='szxy')
  q = quaternion_wxyz
  quaternion_xyzw = (q[1], q[2], q[3], q[0])
  return quaternion_xyzw


def quatXYZW_to_eulerXYZ(quaternion_xyzw):  # pylint: disable=invalid-name
  """Abstraction for converting from quaternion to a 3-parameter toation.

  This will help us easily switch which rotation parameterization we use.
  Quaternion should be in xyzw order for pybullet.

  Args:
    quaternion_xyzw: in xyzw order, tuple of 4 floats

  Returns:
    rotation: a 3-parameter rotation, in xyz order, tuple of 3 floats
  """
  q = quaternion_xyzw
  quaternion_wxyz = np.array([q[3], q[0], q[1], q[2]])
  euler_zxy = euler.quat2euler(quaternion_wxyz, axes='szxy')
  euler_xyz = (euler_zxy[1], euler_zxy[2], euler_zxy[0])
  return euler_xyz

def get_camera_image(env):
    image_size = (800, 800)
    fx = fy = 280  # lower than 800 for wider FOV
    cx = image_size[0] / 2
    cy = image_size[1] / 2

    intrinsics = (fx, 0, cx, 0, fy, cy, 0, 0, 1)
    color, _, _, _, _ = render_image(image_size, intrinsics)
    return color

def get_observation(env):
    observation = {}

    # Render current image.
    color, depth, position, orientation, intrinsics = render_image(env, noise=env.noise)

    # Get heightmaps and colormaps.
    points = get_pointcloud(env, depth, intrinsics)
    position = np.float32(position).reshape(3, 1)
    rotation = pybullet.getMatrixFromQuaternion(orientation)
    rotation = np.float32(rotation).reshape(3, 3)
    transform = np.eye(4)
    transform[:3, :] = np.hstack((rotation, position))
    points = transform_pointcloud(env, points, transform)
    heightmap, colormap, xyzmap = get_heightmap(env, points, color, BOUNDS, PIXEL_SIZE)

    observation["image"] = colormap
    observation["xyzmap"] = xyzmap
    return observation

def set_alpha_transparency(env, alpha: float) -> None:
    for id in range(20):
        visual_shape_data = pybullet.getVisualShapeData(id)
        for i in range(len(visual_shape_data)):
            object_id, link_index, _, _, _, _, _, rgba_color = visual_shape_data[i]
            rgba_color = list(rgba_color[0:3]) + [alpha]
            pybullet.changeVisualShape(env.robot_id, linkIndex=i, rgbaColor=rgba_color)
            pybullet.changeVisualShape(env.suction.body, linkIndex=i, rgbaColor=rgba_color)

def get_camera_image_top(env,
                         image_size=(240, 240),
                         intrinsics=(2000., 0, 2000., 0, 2000., 2000., 0, 0, 1),
                         position=(0, -0.5, 5),
                         orientation=(0, np.pi, -np.pi / 2),
                         zrange=(0.01, 1.),
                         set_alpha=True):
    set_alpha and set_alpha_transparency(env, alpha=0)
    color, _, _, _, _ = env.render_image_top(image_size,
                                             intrinsics,
                                             position,
                                             orientation,
                                             zrange,
                                             noise=env.noise)
    set_alpha and set_alpha_transparency(env, alpha=1)
    return color


def render_image_top(env,
                     image_size=(240, 240),
                     intrinsics=(2000., 0, 2000., 0, 2000., 2000., 0, 0, 1),
                     position=(0, -0.5, 5),
                     orientation=(0, np.pi, -np.pi / 2),
                     zrange=(0.01, 1.),
                     noise=False):
    # Camera parameters.
    orientation = pybullet.getQuaternionFromEuler(orientation)

    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)
    rotation = pybullet.getMatrixFromQuaternion(orientation)
    rotm = np.float32(rotation).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    updir = (rotm @ updir).reshape(-1)
    lookat = position + lookdir
    focal_len = intrinsics[0]
    znear, zfar = (0.01, 10.)
    viewm = pybullet.computeViewMatrix(position, lookat, updir)
    fovh = (image_size[0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

    # Notes: 1) FOV is vertical FOV 2) aspect must be float
    aspect_ratio = image_size[1] / image_size[0]
    projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

    # Render with OpenGL camera settings.
    _, _, color, depth, segm = pybullet.getCameraImage(
        width=image_size[1],
        height=image_size[0],
        viewMatrix=viewm,
        projectionMatrix=projm,
        shadow=1,
        flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

    # Get color image.
    color_image_size = (image_size[0], image_size[1], 4)
    color = np.array(color, dtype=np.uint8).reshape(color_image_size)
    color = color[:, :, :3]  # remove alpha channel
    if noise:
        color = np.int32(color)
        color += np.int32(np.random.normal(0, 3, color.shape))
        color = np.uint8(np.clip(color, 0, 255))

    # Get depth image.
    depth_image_size = (image_size[0], image_size[1])
    zbuffer = np.float32(depth).reshape(depth_image_size)
    depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
    depth = (2 * znear * zfar) / depth
    if noise:
        depth += np.random.normal(0, 0.003, depth.shape)

    intrinsics = np.float32(intrinsics).reshape(3, 3)
    return color, depth, position, orientation, intrinsics


def render_image(env, image_size=(720, 720), intrinsics=(360., 0, 360., 0, 360., 360., 0, 0, 1), noise=False):
    # Camera parameters.
    position = (0, -0.85, 0.4)
    orientation = (np.pi / 4 + np.pi / 48, np.pi, np.pi)
    orientation = pybullet.getQuaternionFromEuler(orientation)
    zrange = (0.01, 10.)

    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)
    rotation = pybullet.getMatrixFromQuaternion(orientation)
    rotm = np.float32(rotation).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    updir = (rotm @ updir).reshape(-1)
    lookat = position + lookdir
    focal_len = intrinsics[0]
    znear, zfar = (0.01, 10.)
    viewm = pybullet.computeViewMatrix(position, lookat, updir)
    fovh = (image_size[0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

    # Notes: 1) FOV is vertical FOV 2) aspect must be float
    aspect_ratio = image_size[1] / image_size[0]
    projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

    # Render with OpenGL camera settings.
    _, _, color, depth, segm = pybullet.getCameraImage(
        width=image_size[1],
        height=image_size[0],
        viewMatrix=viewm,
        projectionMatrix=projm,
        shadow=1,
        flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

    # Get color image.
    color_image_size = (image_size[0], image_size[1], 4)
    color = np.array(color, dtype=np.uint8).reshape(color_image_size)
    color = color[:, :, :3]  # remove alpha channel
    if noise:
        color = np.int32(color)
        color += np.int32(np.random.normal(0, 3, color.shape))
        color = np.uint8(np.clip(color, 0, 255))

    # Get depth image.
    depth_image_size = (image_size[0], image_size[1])
    zbuffer = np.float32(depth).reshape(depth_image_size)
    depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
    depth = (2 * znear * zfar) / depth
    if noise:
        depth += np.random.normal(0, 0.003, depth.shape)

    intrinsics = np.float32(intrinsics).reshape(3, 3)
    return color, depth, position, orientation, intrinsics


def get_pointcloud(env, depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.
    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points

def transform_pointcloud(env, points, transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
      points: HxWx3 float array of 3D points in camera coordinates.
      transform: 4x4 float array representing a rigid transformation matrix.
    Returns:
      points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,
                            'constant', constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points


def get_heightmap(env, points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.
    Args:
      points: HxWx3 float array of 3D points in world coordinates.
      colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
      bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
        region in 3D space to generate heightmap in world coordinates.
      pixel_size: float defining size of each pixel in meters.
    Returns:
      heightmap: HxW float array of height (from lower z-bound) in meters.
      colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
      xyzmap: HxWx3 float array of XYZ points in world coordinates.
    """
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)
    xyzmap = np.zeros((height, width, 3), dtype=np.float32)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[py, px, c] = colors[:, c]
        xyzmap[py, px, c] = points[:, c]
    colormap = colormap[::-1, :, :]  # Flip up-down.
    xv, yv = np.meshgrid(np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], height),
                         np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], width))
    xyzmap[:, :, 0] = xv
    xyzmap[:, :, 1] = yv
    xyzmap = xyzmap[::-1, :, :]  # Flip up-down.
    heightmap = heightmap[::-1, :]  # Flip up-down.
    return heightmap, colormap, xyzmap