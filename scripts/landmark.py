
import argparse
import copy
from itertools import product

try:
    # import lxml.etree as et
    import xml.etree.cElementTree as et
except ImportError:
    import xml.etree.ElementTree as et

import numpy as np
from scipy.misc import imread, imsave
import cv2

fov = 50.0 * 2 * np.pi / 360.0
f = 1 / (2 * np.tan(fov / 2))
ideal_calibration_matrix = np.array([
                                    [800*f, 0, 400],
                                    [0, 800*f, 300],
                                    [0, 0, 1]
                                    ])

def keypoints_str(keypoints):
    
    tuples = [kp.pt + (kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]
    tuples = [map(repr, t) for t in tuples]
    tuples = [" ".join(t) for t in tuples]
    text = "\n".join(tuples)
    
    return text

def keypoints_element(root, name, keypoints):
    
    element = et.SubElement(root, name)
    element.text = keypoints_str(keypoints)
    return element

def opencvmat_element(root, name, array):
    
    element = et.SubElement(root, name)
    
    element.attrib["type_id"] = "opencv-matrix"
    
    rows = et.SubElement(element, "rows")
    rows.text = str(array.shape[0])
    
    cols = et.SubElement(element, "cols")
    cols.text = str(array.shape[1])
    
    dt = et.SubElement(element, "dt")
    dt.text = "u" # TODO: Change this
    
    data = et.SubElement(element, "data")
    data.text = "\n".join([" ".join(map(repr, row)) for row in array])
    # data.text = " ".join(map(repr, array.ravel()))
    
    return element

def size_element(root, name, size):
    
    element = et.SubElement(root, name)
    element.text = " ".join(map(str, size))
    
    return element

def transform_points(points, roll, pitch, t_z=0):
    
    roll_transform = np.array([
                            [np.cos(roll), np.sin(roll), 0],
                            [-np.sin(roll), np.cos(roll), 0],
                            [0, 0, 1]
                        ])
    
    pitch_transform = np.array([
                            [1, 0, 0],
                            [0, np.cos(pitch), np.sin(pitch)],
                            [0, -np.sin(pitch), np.cos(pitch)]
                        ])
    
    res = np.dot(pitch_transform, np.dot(roll_transform, points.T)).T
    res[:, 2] += t_z
    return res

def hom2cart(points, calibration_matrix):
    points = np.dot(calibration_matrix, points.T).T
    return points[:, :2] / points[:, [2]]

def tight_frame(points, margin=0):
    """
    Translate points and compute the shape of a tight frame around
    those points.
    
    Returns (new_points, shape)
    """
    
    min_coords = np.min(points, 0) - margin
    new_points = points - min_coords
    new_size = np.max(new_points, 0) + margin
    
    return new_points, new_size

def get_homography(pitch, roll,
                   image_size, real_size,
                   calibration_matrix,
                   t_z=None, margin=0):
    """
    image_size : tuple, ndarray
        Image size (cols, rows) in pixels
    
    real_size : tuple, ndarray
        Real size of the landmark (width, height) in real world units (mm, cm...)
    
    """
    
    template = np.array([[-0.5, -0.5, 0],
                         [0.5, -0.5, 0],
                         [0.5, 0.5, 0],
                         [-0.5, 0.5, 0]])
    real_points = np.hstack([real_size, 0]) * template
    
    if t_z is None:
        t_z = 5 * np.max(real_size)
    
    points = transform_points(real_points, roll, pitch, t_z)
    
    if np.any(points[:, 2] < 0):
        print "WARNING: Part of the template falls behind the camera!"
    
    proj_points = hom2cart(points, calibration_matrix)
    
    new_points, new_size = tight_frame(proj_points, margin)
    
    template_points = np.array([
                                [0, 0],
                                [image_size[0], 0],
                                [image_size[0], image_size[1]],
                                [0, image_size[1]]
                               ])
    h = cv2.getPerspectiveTransform(np.float32(template_points),
                                    np.float32(new_points))
    
    return h, tuple(np.int_(np.round(new_size)))

def project_keypoints(keypoints, homography):
    
    kp_positions = np.array([kp.pt for kp in keypoints])
    new_kp_positions = cv2.perspectiveTransform(kp_positions[None, :, :], homography)
    
    new_keypoints = copy.deepcopy(keypoints)
    for new_kp, old_kp, new_pos in zip(new_keypoints, keypoints, new_kp_positions[0]):
        new_kp.pt = tuple(new_pos)
    
    return new_keypoints

def process_landmark(image,
                     pixels_per_unit,
                     feature_extractor,
                     orientations,
                     calibration_matrix=None,
                     debug=0):
    
    image_size = image.shape[:2][::-1]
    real_size = np.asarray(image.shape[:2])[::-1] / pixels_per_unit
    
    if calibration_matrix is None:
        calibration_matrix = ideal_calibration_matrix
    
    keypoints = []
    descriptors = []
    for pitch, roll in orientations:
        h, new_shape = get_homography(pitch, roll,
                                      image_size, real_size,
                                      calibration_matrix,
                                      margin=50)
        image2 = cv2.warpPerspective(image, h, new_shape,
                                     None,
                                     cv2.INTER_LINEAR,
                                     cv2.BORDER_CONSTANT,
                                     (255, 255, 255))
        
        cur_keypoints, cur_descriptors = feature_extractor.detectAndCompute(image2, None)
        
        if debug:
            image3 = np.empty_like(image2)
            cv2.drawKeypoints(image2, cur_keypoints, image3)
            imsave("debug_pitch{:.2}_roll{:.2}.png".format(pitch, roll), image3)
        
        # Project back the keypoints to the template
        back_keypoints = project_keypoints(cur_keypoints, np.linalg.inv(h))
        
        keypoints.extend(back_keypoints)
        descriptors.extend(cur_descriptors)
    
    return keypoints, np.vstack(descriptors), image_size, real_size

def save(out_filename, keypoints, descriptors, image_size, real_size):
    
    root = et.Element("opencv_storage")
    size_element(root, "image_size", image_size)
    size_element(root, "real_size", real_size)
    keypoints_element(root, "keypoints", keypoints)
    opencvmat_element(root, "descriptors", descriptors)
    
    etree = et.ElementTree(root)
    etree.write(out_filename, encoding="utf-8", xml_declaration=True)

def get_orientations(num_orientations):
    
    orientations = [(0.0, 0.0)]
    
    rolls = [2 * np.pi * i / num_orientations for i in xrange(num_orientations)]
    pitchs = [np.pi / 3]
    
    aux = [(pitch, roll) for pitch, roll in product(pitchs, rolls)]
    orientations.extend(aux)
    
    return orientations

def main():
    
    parser = argparse.ArgumentParser(description="Landmark manager.")
    parser.add_argument("image", type=str, help="frontal view of the landmark")
    parser.add_argument("output", type=str, help="output xml file name")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--ppu", type=float, default=20.0, help="pixels per (real world) unit")
    
    args = parser.parse_args()
    
    image = imread(args.image)[..., :3]
    
    fextractor = cv2.ORB_create(100)
    orientations = get_orientations(8)
    keypoints, descriptors, image_size, real_size = process_landmark(image,
                                              pixels_per_unit=args.ppu,
                                              feature_extractor=fextractor,
                                              orientations=orientations,
                                              debug=args.debug)
    save(args.output, keypoints, descriptors, image_size, real_size)

if __name__ == "__main__":
    main()
