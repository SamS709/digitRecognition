import numpy as np
import time
import cv2 as cv
import cv2
import matplotlib.pyplot as plt

class Camera:

    def __init__(self):
        # connect to the robot when a new Robot object is created
        pass

    def photo(self): # returns 2 undistort image returned by the robot's camera : the first one is just the conversion of colored image to B&W # the second one is processed to have a white board (rgb = 1,1,1) and black pieces(rgb = 0,0,0)
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        _okok , img = camera.read()
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert image to greyscale
        img_gblur = cv.GaussianBlur(img_gray, (5, 5), 0) # apply blur
        ret, img_thres = cv.threshold(img_gblur, 110, 250, cv.THRESH_BINARY)  # apply otsu's binary
        image_copy = img_gray.copy()
        return image_copy, img_thres


    def contour(self):
        image_copy, img_thres = self.photo()
        contours, hierarchy = cv2.findContours(image=img_thres, mode=cv2.RETR_TREE,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
        coordinates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if  5000>area > 500:  # pieces have an area between 7000 and 2000
                cv2.drawContours(image=image_copy, contours=cnt, contourIdx=-1, color=(1, 0, 0), thickness=2,
                                 lineType=cv2.LINE_4)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                if w/8<h<8*w:
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
            if  50000>area > 500:  # pieces have an area between 7000 and 2000
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                if w/10<h<10*w:
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

    def to_square(self,image):
        L = max(image.shape)
        l = min(image.shape)
        if image.shape[0]<image.shape[1]:
            img = np.array([[255 for j in range(L)]for i in range(L)])
            img[(L-l)//2:(L+l)//2,0:L] = image
        else:
            img = np.array([[255 for j in range(L)] for i in range(L)])
            img[0:L, (L-l)//2:(L+l)//2] = image
        return img

    def pix_vals(self,image):
        image = 255-image
        std = np.std(image)
        image[image<=np.max(image)-1.7*std]=0
        max = np.max(image)
        diff = 255-max
        image[image!=0] = image[image!=0]+diff
        return image


    def numbers_images(self):
        eps = 8
        image_init,coordinates=self.numbers_coords()
        print(coordinates)
        images = []
        for coord in coordinates:
            x0, x1, y0, y1 = coord[0]-eps,coord[1]+eps,coord[2]-eps,coord[3]+eps
            image = image_init[y0:y1, x0:x1].copy()
            image = self.to_square(image)
            image = self.pix_vals(image)
            image = cv2.resize(np.array(image, dtype='uint8'), (28, 28),interpolation= cv2.INTER_LINEAR)
            image = np.flip(image, axis=0)
            image = np.flip(image, axis=1)
            images.append(image)
            if np.mean(image[0]) >= 20 or np.mean(image[27]) >= 20 or np.mean(image[:,27]) >= 20 or np.mean(image[:,0]) >= 20:

                images.pop()
        print(images[0])
        return np.array(images)



if __name__=="__main__":
    robot1 = Camera()
    #robot1.get_HSV_and_mousePos()
    #coordinates = robot1.contour()[1]
    #print(coordinates)
    #robot1.get_HSV_and_mousePos()
    images = robot1.numbers_images()
    for image in images:
        plt.figure()
        plt.imshow(image)
    plt.show()
    print(images.shape)