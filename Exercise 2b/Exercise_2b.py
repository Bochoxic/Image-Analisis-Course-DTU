import time
import cv2
import numpy as np


def show_in_moved_window(win_name, img):
    """"
    Show an image in a window, there the position of the
    window can be given
    
    """
    cv2.namedWindow(win_name)
    #cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)

def add_text(image, text, org, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    image = cv2.putText(image, text, org, font, fontScale, color)
    return image
    
def capture_from_camera_and_show_images():
    alpha=0.98
    T=20
    A=20
    print("Starting image capture")
    
    print("Opening connection to camera")
    url=0
    use_droid_cam=False
    if use_droid_cam:
        url="http://192.168.1.120:4747/video"
    cap=cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    print("Starting camera loop")
    # Get first image
    ret, frame=cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        exit()
        
    # Transform image to gray scale and then to float, so we can do some processing
    frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # To keep track of the frames per second
    start_time=time.time()
    n_frames=0
    stop=False
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
        # Transform image to gray scale and then to float, so we can do some processing
        new_frame_gray= cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Compute difference image
        dif_img = np.abs(new_frame_gray- frame_gray)
        
        # Create a binary image 
        ret,thresh1 = cv2.threshold(dif_img,T,255,cv2.THRESH_BINARY)
        
        # Computes the total number of foreground, F, pixels in the foreground image.
        F = np.count_nonzero(thresh1)
        totalF = thresh1.size
        F_percentage = F/totalF*100
        
        # Set an alarm
        if F_percentage>A:
            print(f"Change Detected! Has changed {F_percentage}% of the foreground")
        
        # Get some stats
        max_value = np.max(dif_img)
        min_value = np.min(dif_img)
        mean_value = np.mean(dif_img)
        
        
        # Draw text
        text1 = "Max value = " + str(max_value)
        text2 = "Min value = " + str(min_value)
        text3 = "Number of pixel changed = " + str(F_percentage)
        text4 = "Mean value = " + str(mean_value)
        
        org1 = (10, 30)
        org2 = (10, 60)
        org3 = (10, 90)
        org4 = (10, 120)

        color = (245, 245, 245)
        dif_img = add_text(dif_img, text1, org1, color)
        dif_img = add_text(dif_img, text2, org2, color)
        dif_img = add_text(dif_img, text3, org3, color)
        dif_img = add_text(dif_img, text4, org4, color)
        
        # Show the images
        #show_in_moved_window('Input', new_frame)
        #show_in_moved_window('Background', new_frame_gray.astype(np.uint8))
        show_in_moved_window('Difference image', dif_img.astype(np.uint8))
        show_in_moved_window('Binary image', thresh1.astype(np.uint8))
        
        # Old frame is updated
        frame_gray=alpha*frame_gray+(1-alpha)*new_frame_gray
        
        # Stop the loop is key 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            stop = True
    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    capture_from_camera_and_show_images()