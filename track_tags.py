#! /usr/bin/python
# Alli Nilles CEI Lab 2023

import cv2
import numpy as np
from collections import defaultdict
import glob
import csv
import pandas as pd
import yaml # pip install pyyaml

# using duckytown API for apriltags
import dt_apriltags as ap # pip install dt-apriltags

# local code
from utilities import rotationMatrixToEulerAngles
from calibrate import calibrate, calibrate_from_video, process_image
from speedup import VideoGet, VideoShow, CountsPerSec

def viz_and_log_detections(detections, blank_img, color_img, data, count, pose):

    #print("found ",len(detections)," apriltags")

    for tag in detections:

        # outline tag
        raw_corners = tag.corners
        (lb, rb, rt, lt) = [(int(coord[0]), int(coord[1])) for coord in raw_corners]
        # outline tag
        cv2.line(color_img, lb, rb, (0, 0, 255), 2)
        cv2.line(color_img, rb, rt, (0, 0, 255), 2)
        cv2.line(color_img, rt, lt, (0, 0, 255), 2)
        cv2.line(color_img, lt, lb, (0, 0, 255), 2)

        # get tag size and orientation
        topvec = np.array(rt)-np.array(lt)
        orientation = np.arctan2(float(topvec[1]), float(topvec[0]))
        len = np.linalg.norm(topvec)
        r = int(np.sqrt(2.)*len/2.)

        # draw circles on center of apriltags
        (cX, cY) = (int(tag.center[0]), int(tag.center[1]))
        cv2.circle(color_img, (cX, cY), 5, (0, 0, 255), thickness=-1)
        cv2.circle(blank_img, (cX, cY), 5, (0, 0, 255), thickness=-1)

        if pose:
            # get orientation as Euler angles and draw
            rotation = rotationMatrixToEulerAngles(tag.pose_R)
            orientation = rotation[2]
            center_meters = tag.pose_t[:2]


        # draw orientation
        scale = 10.
        vectorend = (int(cX+scale*np.cos(orientation)), int(cY+scale*np.sin(orientation)))
        cv2.line(color_img, (cX, cY), vectorend, (0, 255, 0), 2)
        cv2.line(blank_img, (cX, cY), vectorend, (0, 255, 0), 2)
        #draw ID
        cv2.putText(color_img, str(tag.tag_id),(cX,cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(blank_img, str(tag.tag_id),(cX,cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # log data by bot ID: (id, center in px, orientation in radians, length of tag side in px)
        data[tag.tag_id] = (tag.tag_id, (cX, cY), orientation, len)



    return blank_img, color_img, data

# TODO:
# long term: add features to auto-calibrate if grid is in frame, or auto-calibrate from april tags themselves
def detectTags(detector, tag_size, input_file, output_file="out.mp4", VID=0, camera_params=None, m_per_pix=None, mtx=None, dist=None):
    print(f"Loading {input_file}")
    video_getter = VideoGet(input_file).start()
    fps = video_getter.FPS
    frame_count = video_getter.frame_num
    width  = video_getter.w
    height = video_getter.h
    print(f"FPS of {fps}, total of {frame_count} frames")
    outputf = output_file.split('.')
    if VID:
        out = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width),int(height)))
        #video_shower = VideoShow(video_getter.frame).start()

    count = 0
    RESET = 0 # for completion progress output
    cps = CountsPerSec().start()

    alldata = np.full(frame_count, defaultdict(list), dtype=object)

    # Loop through video
    print(f"Loaded {height}x{width} vid successfully, looping through frames...")
    while True:
        data = defaultdict(list)
        # read next frame or end
        if video_getter.stopped:#or video_shower.stopped:
            video_getter.stop()
            #if VID:
            #    video_shower.stop()
            print(f"End of video")
            break

        frame = video_getter.next_frame()
        #video_shower.frame = frame
        cps.increment()
        data = defaultdict(list)
        # detect tags
        if camera_params:
            gray, color_img = process_image(frame, mtx, dist, cal=True)
            h, w = gray.shape[:2]
            # set up images
            blank_img = np.zeros((int(h),int(w),3), np.uint8)
            window = int(h/200)
            if window % 2 == 0:
                window += 1
            #th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=window, C=5) 
            tags = detector.detect(gray, estimate_tag_pose=True,
                                   camera_params=camera_params, tag_size=tag_size)
            blank_img, color_img, data = viz_and_log_detections(tags, blank_img, color_img, data, count, pose=True)
            #cv2.imshow("img", color_img)
            #cv2.waitKey(1)
        else:
            # set up images
            gray, color_img = process_image(frame)
            h, w = gray.shape[:2]
            blank_img = np.zeros((int(h),int(w),3), np.uint8)
            window = int(h/200)
            if window % 2 == 0:
                window += 1
            #th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,window,0)

            # detect and draw tags
            tags = detector.detect(gray, tag_size=tag_size)
            blank_img, color_img, data = viz_and_log_detections(tags, blank_img, color_img, data, count, pose=False)
            #cv2.imshow("img", color_img)
            #cv2.waitKey(1)
        # log data
        if VID:
            out.write(color_img)
        tag_ids = list(data.keys())
        alldata[count] = data.copy()

        # track progress
        count +=1
        progress = round(100*count/frame_count)
        if (progress%10 == 0) and not RESET:
            RESET = 1
            print(cps.countsPerSec(), "frames per second")
            print(progress, "percent complete")
        elif (progress%10 == 0):
            pass
        else:
            RESET = 0


    video_getter.stop()
    if VID:
        out.release()
        print(f"Wrote annotated video to to {output_file}")
    #cv2.destroyAllWindows()

    fname = outputf[0]+".csv"
    fieldnames=['Frame','Tag ID', 'X', 'Y', 'Theta', 'SideLen']
    with open(fname, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for frame in range(frame_count):
            dat = alldata[frame]
            for (id, (x,y), orientation, len) in list(dat.values()):
                writer.writerow({
                    'Frame':frame,
                    'Tag ID':id,
                    'X': x,
                    'Y':y,
                    'Theta':orientation,
                    'SideLen': len})
                    

    print(f"Wrote results to {fname}")

# TODO
# if no calibration grid, calibrate from apriltags themselves

if __name__ == '__main__':

    with open('config.yaml','r') as file:
        cnfg = yaml.safe_load(file)

    type = cnfg['tag-type']
    size = float(cnfg['tag-size'])
    threadN = float(cnfg['num-threads'])

    TAGCAL = True
    if 'calibration' in cnfg:
        calibration_file = cnfg['calibration']
        TAGCAL = False

    datadir = cnfg['raw-vids-dir']
    ext = cnfg['vid-ext']
    vidfiles = glob.glob(datadir+"*."+ext)
    print(f"Found {len(vidfiles)} vids in {datadir}")
    VID = int(cnfg['write-tracked-vid'])

    for i,input_file in enumerate(vidfiles):

        #cv2.startWindowThread()
        # open vid file and get info
        fname = input_file.split('.')
        output_file=fname[0]+"_tracked.mp4"

        print(f"Loading file {i+1} of {len(vidfiles)}")
        if TAGCAL:
            detector = ap.Detector(families=type,
                                   nthreads=threadN,
                                   refine_edges=1,
                                   decode_sharpening=0.25,
                                   debug=0)

            detectTags(detector, size, input_file, output_file, VID)
        else:
            # calibrate() uses an image
            # calibrate_from_video() extracts a frame from video, then calls calibrate()
            if ext in ["mp4", "mkv", "avi"]:
                ret, camera_params, mtx, dist, meters_per_pix = calibrate_from_video(calibration_file, dims=(6,9))
            else:
                ret, camera_params, mtx, dist, meters_per_pix = calibrate(calibration_file, dims=(6,9))

            detector = ap.Detector(families=type,
                                   nthreads=threadN,
                                   refine_edges=1,
                                   decode_sharpening=0.25,
                                   debug=0)

            detectTags(detector, size, input_file, output_file, VID, camera_params, meters_per_pix, mtx, dist)
