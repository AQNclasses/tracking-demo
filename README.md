# AprilTags

This library consists of three Python files and a Jupyter notebook that shows how to use
the functions within them.

- `calibrate.py` contains code for calibrating camera parameters automatically from an image or video containing a checkerboard of known dimensions.
- `utilities.py` contains some helper functions for the rotation matrices.
- `track_tags.py` contains the code that loops through the video, detects tags, and logs data.

Configuration settings are stored in `config.yaml`. In this file, you should
provide the directory containing files to be processed (`raw-vids-dir`), the
file extension of the videos (ie: mp4), and the april tag type and size. The
size is the length of one side of the tag, including the white pixel on either 
side of the tag.

By default, running the `track_tags` function will generate three files:

- `out-i.mp4`, each original video with overlaid annotations showing the AprilTags being tracked. A circle is drawn at the center of the AprilTag with a line indicating orientation.
- `out-i_blank.mp4`, a white background with just the AprilTags trackers shown.
- a CSV (comma-separated value) file for each video, containing the video
frame, bot id, and center position of each detected Apriltag in pixels.

The code also prints data on the framerate and resolution of the video, as well
as the extracted camera calibration parameters.

To run the code, either execute the cells of the Jupyter notebook in order, or run `python3 track_tags.py` from the command line.

# TODOs

If you're interested in improving the code, here are some features to-be-done:

- Data visualization tools
- Clean up code for logging data, and log pose data if it's available
