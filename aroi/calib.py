# -*- coding: utf-8 -*-
"""Calibration and 3D Reconstruction.
"""
import math
import time
import numpy as np
import cv2

def get_chessboard_points(
        img, pattern_rows, pattern_columns, square_size=1, draw=False):
    """
    :param img: input image
    :param pattern_rows: the number of rows in chessboard
    :param pattern_columns: the number of columns in chessboard
    :param square_size: each square size of chessboard
    :param draw: draw chessboard corners to ``img``
    """
    size = (pattern_rows, pattern_columns)
    found, corners = cv2.findChessboardCorners(img, size)
    if not found:
        return None
    if draw:
        cv2.drawChessboardCorners(img, size, corners, found)
    cv2.cornerSubPix(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        corners, (3, 3), (-1, -1),
        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 20, 0.03))
    # (y, 1, 2) -> (y, 2) by reshape(-1, 2)
    img_points = corners.reshape(-1, 2)
    obj_points = np.zeros((np.prod(size), 3), np.float32)
    obj_points[:,:2] = np.indices(size).T.reshape(-1, 2)
    obj_points *= square_size
    return img_points, obj_points


def get_circles_grid_points(
        img, pattern_rows, pattern_columns, distance=1, draw=False):
    """
    :param img: input image
    :param pattern_rows: the number of rows in chessboard
    :param pattern_columns: the number of columns in chessboard
    :param square_size: each distance between neighber circles
    :param draw: draw chessboard corners to ``img``
    """
    size = (pattern_rows, pattern_columns)
    found, corners = cv2.findCirclesGridDefault(img, size)
    if not found:
        return None
    if draw:
        cv2.drawChessboardCorners(img, size, corners, found)
    cv2.cornerSubPix(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        corners, (3, 3), (-1, -1),
        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 20, 0.03))
    img_points = corners.reshape(-1, 2)
    obj_points = np.zeros((np.prod(size), 3), np.float32)
    obj_points[:,:2] = np.indices(size).T.reshape(-1, 2)
    obj_points *= distance
    return img_points, obj_points
    

def get_square_points(square_points, square_size=1):
    """
    :param square_points: 4 points representing square in clockwise order.
    :param square_size: Square width(=height). An object whose size is ``square_size`` norm in OpenGL coordinates is drawed as real square size in the picture.
    """
    obj_points = np.float32(
        [[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]]) - [0.5, 0, 0.5]
    obj_points *= square_size
    return np.float32(square_points), obj_points
        

class Reconstructor(object):
    """3D reconstructor class.
    """
    
    def __init__(self, calibrator):
        self.calibrator = calibrator
    
    def get_camera_info(self,
            size=(640, 480), aperture_height=0, aperture_width=0):
        """Return fovy and aspectRatio from camera matrix.
        These parameters can be used in ``gluPerspective()``.
        
        :param size: tuple of input image width and height
        :type size: (int, int)
        :param aperture_height: physical height of the camera
        :param aperture_width: physical width of the camera
        
        .. caution::
            The accuracy in using this method and gluPerspective() is not good.
            You had better use :func:`Reconstructor.get_gl_frustum_parameters`.

        .. note::
            I don't understand the reason why ``aperture_height`` and
            ``aperture_width`` are needed, so I set 0 to them as default.
        """
        fovx, fovy, focal_length, principal_point, aspect_ratio = \
            cv2.calibrationMatrixValues(self.calibrator.camera_matrix, size,
                aperture_height, aperture_width)
        return fovx, fovy, focal_length, principal_point, aspect_ratio
            
    def get_gl_frustum_params(self, width, height, near, far):
        """Return arguments of ``glFrustum()``.

        :param width: image width
        :param height: image height
        :param near: near clip
        :param far: far clip
        """
        w = float(width)
        h = float(height)
        m = self.calibrator.camera_matrix
        fx = m[0, 0]
        fy = m[1, 1]
        cx = m[0, 2]
        cy = m[1, 2]
        fovy = 1 / (fy / h*2)
        aspect = w / h * fx / fy
        fh = near * fovy
        fw = fh * aspect
        offx = (w*0.5-cx) / w * fw * 2
        offy = (h*0.5-cy) / h * fh * 2
        return -fw-offx, fw-offx, -fh-offy, fh-offy, near, far
    
    # def get_glu_perspective_params(self, width, height, near, far):
    #     """Return arguments of ``gluPerspective()``.
    #     
    #     .. note:: You should use :func:`Reconstructor.get_gl_frustum_params`
    #      
    #     .. seealso: http://d.hatena.ne.jp/ousttrue/20081110/1226291802
    #     """
    #     left, right, bottom, top, near, far = self.get_gl_frustum_params(
    #         width, height, near, far)
    #     f = 2 * near / (top - bottom)
    #     aspect = float(width) / height
    #     fovy_rad = math.atan(1.0 / f) * 2
    #     fovy_deg = np.rad2deg(fovy_rad)
    #     return fovy_deg, aspect, near, far
    
    def estimate_square_pose(self, img_points, obj_points):
        """Estimate pose from points of square.
        
        :param img_points: 2D points in image.
        :param obj_points: 3D points in your virtual space.
        """
        rms, rvec, tvec = cv2.solvePnP(obj_points, img_points,
            self.calibrator.camera_matrix, self.calibrator.dist_coeffs)
        strip = lambda vec: np.array(map(lambda a: a[0], vec))
        # OpenCV coordinate -> OpenGL coordinate
        # http://jibaravr.blog51.fc2.com/blog-entry-83.html
        t = strip(tvec) * [1, -1, -1]
        r = strip(rvec) * [1, -1, -1]
        return r, t
    

class Calibrator(object):
    """Camera calibrator class.
    """
    
    def __init__(self, camera_matrix=None, dist_coeffs=None, record=None):
        """
        Camera matrix (``camera_matrix``) and distortion coefficients
        (``dist_coeffs``) will be estimated by OpenCV's camera calibration.
        
        :param camera_matrix: initial camera matrix.
        :param dist_coeffs: initial distortion coefficients.
        :param record: initial calibration record data.
        
        .. seealso:: `cv2.calibrateCamera <http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#calibratecamera>`_
        """
        # properties
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.record = record or {"points": []}
    
    @classmethod
    def from_npz(cls, calib_file=None, record_file=None):
        """Create instance from npz file.
        
        :param calib_file: npz file that contains "camera_matrix" and "dist_coeffs" properties.
        :param record_file: npz file that contains record data used for calculation of calibration data.
        """
        if calib_file is None:
            data = {"camera_matrix": None, "dist_coeffs": None}
        else:
            data = np.load(calib_file)
            if not("camera_matrix" in data and "dist_coeffs" in data):
                raise Error("Invalid calibration file.")
        if record_file is None:
            record = None
        else:
            record = dict(np.load(record_file))
            record["points"] = record["points"].tolist()
        return cls(data["camera_matrix"], data["dist_coeffs"], record)
    
    def save_as_npz(self,
            calib_file="calib.npz", record_file="calib-record.npz"):
        """Save information of camera calibration data as Numpy .npz format.
        
        .. seealso::
            `About NPZ file <http://docs.scipy.org/doc/numpy/reference/routines.io.html#npz-files>`_
        """
        np.savez(calib_file,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs)
        np.savez(record_file, **self.record)
    
    def append_points(self, img_points, obj_points):
        self.record["points"].append({
            "img": img_points,
            "obj": obj_points,
        })
    
    def calibrate(self, width, height):
        size = (width, height)
        self.record["size"] = size
        self.record["timestamp"] = time.time()
        
        img_points_list = []
        obj_points_list = []
        for points in self.record["points"]:
            img_points_list.append(points["img"])
            obj_points_list.append(points["obj"])
        
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_list, img_points_list, size)
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        return rvecs[0], tvecs[0]
    

#-------------------------------------

def main():
    """Main function of this module.
    You can start this program with the command below.
    
        $ python -m area.calib ...
        
    """
    import os
    import argparse
    from . import utils
    parser = argparse.ArgumentParser("python -m area.calib")
    utils.Capture.setup_argparser(parser)
    parser.add_argument("--size", metavar=("COLUMNS", "ROWS"),
        type=int, nargs=2, required=True, help="""
    The number of columns and rows in the target.
    """)
    parser.add_argument("-t", "--target", metavar="TARGET=chessboard",
        default="chessboard", choices=["chessboard", "circlegrid"])
    parser.add_argument("-o", "--output", metavar="OUTPUT=calib.npz", 
        default="calib.npz", help="""
    Output npz file including result of calibration. Record file name is
    generated automatically by using this file name.
    (ex. calib.npz --> calib-record.npz)
    """)
    parser.add_argument("--fps", metavar="FPS=1", type=int, default=1)
    parser.add_argument("--record", metavar="EXISTING_RECORD_FILE", help="""
    If you set it, you can resume calibration.
    """)
    args = parser.parse_args()
    
    capture = utils.Capture.create(args)
    calibrator = Calibrator.from_npz(None, args.record)
    frame_time = 1000.0 / (capture.fps or args.fps)
    window = utils.Window("Calibration")
    
    count = len(calibrator.record["points"])
    while True:
        img = capture.get()
        if img is not None:
            if args.target == "chessboard":
                points = get_chessboard_points(
                    img, args.size[0], args.size[1], draw=True)
            elif args.target == "circlegrid":
                points = get_circles_grid_points(
                    img, args.size[0], args.size[1], draw=True)
            if points is not None:
                calibrator.append_points(*points)
                count += 1
            cv2.putText(img, str(count), (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            window.update(img)
        if window.wait(frame_time) == window.ESCAPE:
            break
    
    print "calculate..."
    calibrator.calibrate(capture.width, capture.height)
    print "camera matrix:"
    print calibrator.camera_matrix
    print "dist coeffs:"
    print calibrator.dist_coeffs
    print "save..."
    record_file = os.path.splitext(args.output)[0] + "-record.npz"
    calibrator.save_as_npz(args.output, record_file)
    print "finish!"
    
    
if __name__ == "__main__":
    main()

