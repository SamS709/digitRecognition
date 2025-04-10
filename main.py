from pyniryo import *
import numpy as np
import time
import cv2 as cv
import matplotlib.pyplot as plt

class Robot:

    def __init__(self):
        # connect to the robot when a new Robot object is created
        robot_ip_address = "10.10.10.10"
        robot = NiryoRobot(robot_ip_address)
        robot.calibrate_auto()
        robot.update_tool()
        robot.set_arm_max_velocity(100)
        self.robot = robot
        self.stock = PoseObject(x = -0.0220, y = -0.1308, z = 0.0989,
                                roll = -0.248, pitch = 1.259, yaw = 2.945)  # position of the stock of circles (pieces played by the robot)
        self.observation_pose = PoseObject(x = 0.0096, y = -0.1932, z = 0.1566,
                                           roll = 3.117, pitch = 1.335, yaw = 1.620
) # position adapted to analyse the board
        self.home_pos = PoseObject(x = -0.0003, y = -0.1231, z = 0.1630,
                                   roll = -0.014, pitch = 1.053, yaw = -1.560)

    def cam_pos(self):  # the robot moves towards a position from which it can analyse the board game
        self.robot.move_pose(self.observation_pose)

    def photo(self): # returns 2 undistort image returned by the robot's camera : the first one is just the conversion of colored image to B&W # the second one is processed to have a white board (rgb = 1,1,1) and black pieces(rgb = 0,0,0)
        self.cam_pos()
        time.sleep(0.5) # avoid problems of pieces detection : let time to the camera to adapt its luminosity
        mtx, dist = self.robot.get_camera_intrinsics() # see Niryo docuentation
        img = self.robot.get_img_compressed()
        img_uncom = uncompress_image(img)
        img_undis = undistort_image(img_uncom, mtx, dist)
        img_gray = cv.cvtColor(img_undis, cv.COLOR_BGR2GRAY) # convert image to greyscale
        img_gblur = cv.GaussianBlur(img_gray, (5, 5), 0) # apply blur
        ret, img_thres = cv.threshold(img_gblur, 150, 180, cv.ADAPTIVE_THRESH_GAUSSIAN_C)  # apply otsu's binary
        image_copy = img_gray.copy()
        return image_copy, img_thres


    def contour(self):
        image_copy, img_thres = self.photo()
        contours, hierarchy = cv2.findContours(image=img_thres, mode=cv2.RETR_TREE,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
        coordinates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if  1500>area > 500:  # pieces have an area between 7000 and 2000
                cv2.drawContours(image=image_copy, contours=cnt, contourIdx=-1, color=(1, 0, 0), thickness=2,
                                 lineType=cv2.LINE_4)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                if w/2<h<2*w:
                    coordinates.append([x, x+w, y, y+h])
                    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 5)
        return image_copy,coordinates

    def numbers_coords(self):
        image_copy, img_thres = self.photo()
        contours, hierarchy = cv2.findContours(image=img_thres, mode=cv2.RETR_TREE,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
        coordinates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if  1500>area > 500:  # pieces have an area between 7000 and 2000
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                if w/2<h<2*w:
                    coordinates.append([x, x+w, y, y+h])

        '''x0,x1,y0,y1 = 245,346,118,217
        h,w = image_copy.shape[0],image_copy.shape[1]
        img_thres[:y0, :], img_thres[y1:h, :], img_thres[:, :x0], img_thres[:, x1:w] = 0, 0, 0, 0
        image_copy[:y0,:],image_copy[y1:h,:],image_copy[:,:x0],image_copy[:,x1:w]=0,0,0,0
        img_thres_copy = np.array([[0 for j in range(x0,x1)]for i in range(y0,y1)])
        img_thres_copy[:y1-y0,:] = img_thres[y0:y1,x0:x1].copy()
        img_thres_copy = np.flip(img_thres_copy,axis=0)
        img_thres_copy = np.flip(img_thres_copy,axis=1)
        plt.imshow(img_thres_copy)
        plt.show()'''
        return image_copy,coordinates

    def get_HSV_and_mousePos(self): # useful to set upper and lower bound of red and yellow masks (HSV color) defined in red_yellow_pos()
                                    # also to set x and y in pos_grid(i,j) function
        def on_mouse(event, x, y, flags, param):
            # Check if the event was the left mouse button being clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                # Get the BGR pixel value at the clicked location
                pixel = frame[y, x]

                # Convert BGR to HSV and print the pixel value

                print("pixel pos: (", x, ',', y, ')')
                print()
                # Append the pixel value to the values list
                vals.append(1)

        def get_thresh_from_vals(vals: np.array) -> np.array:
            # Calculate the minimum and maximum values for each channel
            min_h, min_s, min_v = np.min(vals, axis=0)
            max_h, max_s, max_v = np.max(vals, axis=0)
            lower_color = [min_h, min_s, min_v]
            upper_color = [max_h, max_s, max_v]
            # Output the results
            print(f"lower bound: {lower_color}")
            print(f"upper bound: {upper_color}")
            return lower_color, upper_color
        # Open a connection to the webcam (you may need to change the index)
        frame = self.contour()[0]
        print(frame.shape)
        vals = []

        while True:
            # Capture frame-by-frame
            frame = self.contour()[0]
            # Display the frame
            cv2.imshow('frame', frame)
            # Set the callback function for mouse events
            cv2.setMouseCallback('frame', on_mouse)  # Make sure 'Frame' matches the window name in cv2.imshow
            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
        # Release the capture when everything is done
        cv2.destroyAllWindows()
        low, up = get_thresh_from_vals(vals)

    def display_photo(self):
        image_copy, img_thres = self.photo()
        key = cv2.imshow('coucou', image_copy)
        cv2.waitKey(0)

    def numbers_images(self):
        eps = 5
        image_init,coordinates=self.numbers_coords()
        images = []
        for coord in coordinates:
            x0, x1, y0, y1 = coord[0]-eps,coord[1]+eps,coord[2]-eps,coord[3]+eps
            image = np.array([[0 for j in range(x0, x1)] for i in range(y0, y1)])
            image[:y1 - y0, :] = image_init[y0:y1, x0:x1].copy()
            image = np.flip(image, axis=0)
            image = np.flip(image, axis=1)
            image.resize((28,28))
            images.append(image)
        return images



if __name__=="__main__":
    robot1 = Robot()
    print(robot1.robot.get_pose())
    #robot1.get_HSV_and_mousePos()
    coordinates = robot1.contour()[1]
    print(coordinates)
    images = robot1.numbers_images()
    for image in images:
        plt.figure()
        plt.imshow(image)
    plt.show()
