import cv2
import os 
from uuid import uuid4


POS_PATH = "./positive"
ANC_PATH = "./anchor"

# Establish a connection to the webcam
cap = cv2.VideoCapture(-1)

while cap.isOpened(): 
    ret, frame = cap.read()
   
    # Cut down frame to 250x250px
    frame = frame[120:120+250,200:200+250, :]
    
    # Collect anchors -- everytime you hit "a" frame will be saved in anchor folder
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Create the unique file path 
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid4()))
        # Write out anchor image
        print(imgname)
        cv2.imwrite(imgname, frame)
    
    # Collect positives -- -- everytime you hit "p" frame will be saved in positive folder
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Create the unique file path 
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid4()))
        print(imgname)
        # Write out positive image
        cv2.imwrite(imgname, frame)
    
    # Show image back to screen
    cv2.imshow('Image Collection', frame)
    
    # Breaking gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()