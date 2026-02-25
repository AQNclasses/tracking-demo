import numpy as np
import cv2
import glob

# default calibration image is a jpg named "calibrate.jpg"
# make sure still image has same camera settings as video frames!

# CODE HEAVILY BORROWED FROM: https://docs.opencv.org/4.5.2/dc/dbb/tutorial_py_calibration.html

CORNER_DIST = 0.02 # each square in grid is 20x20mm
DEBUG = False


def calibrate_grid(fname = "calibrate.jpg", dims = (7,9)):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((dims[0]*dims[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:dims[0],0:dims[1]].T.reshape(-1,2)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    print("calibrating camera with", fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    print(f"Looking for {dims} chess board")
    ret, corners = cv2.findChessboardCorners(gray, dims, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("found checkerboard...")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, dims, corners2, ret)
    else:
        return False, [], [], [], [], 0.

    # use openCV camera calibration to get camera matrix, distortion coefficients, rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # compute mean error of calibration
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
        print( "total calibration error (px): {}".format(mean_error/len(objpoints)) )

    # compute scale in image in workspace
    px_dist = cv2.norm(imgpoints2[0], imgpoints2[1], cv2.NORM_L2)
    meters_per_px = CORNER_DIST/px_dist
    print("meters per pixel:",meters_per_px)

    return ret, mtx, dist, rvecs, tvecs, meters_per_px

# get calibration params
# camera matrix, distortion coefficients, meters per pixel
def calibrate(fname="calibrate.jpg", dims=(7,9)):
    ret, mtx, dist, rvecs, tvecs, meters_per_pix = calibrate_grid(fname, dims)
    if ret:
        fx, fy, cx, cy = mtx[0][0], mtx[1][1], mtx[0][2], mtx[1][2]
        camera_params = (fx, fy, cx, cy)
        print("camera params:", camera_params)
        return True, camera_params, mtx, dist, meters_per_pix
    else:
        return False, [], [], 0., 0.

# default is to calibrate from calibrate.jpg but you can also use a vid clip
# will return values from last valid frame of video
def calibrate_from_video(fname = "calibrate.mp4", dims=[]):
    i = 0
    cap = cv2.VideoCapture(fname)
    interval=24
    CAL = False
    ## Loop untill the end of the video
    while (cap.isOpened()):
        ret, frame = cap.read()
        if (frame is None):
            break
        elif (i % interval == 0 and not CAL):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("%02d.jpg"%i,gray)
            i += 1
    # release the video capture object
    cap.release()
    images = glob.glob("[0-9][0-9].jpg")
    for fname in images:
        ret, c, m, d, mpp = calibrate(fname, dims)
        if ret:
            return ret, c, m, d, mpp
        else:
            pass
    raise ValueError("calibration from video failed!")


# apply calibration params to an image, and make black and white version
def process_image(img_color, mtx=None, dist=None, cal=False):
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    #img = cv2.blur(img, (5,5))

    #undistort
    if cal:
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # if the roi is significantly smaller than the original image its a sign something went wrong
        if (roi[2] < w/2) or (roi[3] < h/2):
            if DEBUG:
                print("Not using camera calibration data, check calibration")
            pass
        else:
            img = cv2.undistort(img, mtx, dist, None, newcameramtx)
            img_color = cv2.undistort(img_color, mtx, dist, None, newcameramtx)
            x, y, w, h = roi
            img = img[y:y+h, x:x+w]
            img_color = img_color[y:y+h, x:x+w]
            if DEBUG:
                print("Calibration successful")
    return img, img_color
