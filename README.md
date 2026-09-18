## DIPT-WORKSHOP-5
## License Plate Detection using OpenCV and Haar Cascade Classifier
## Name : Thamizh S
## Reg.no : 212224040350


```
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
img = cv2.imread('car_plate.jpg')

if img is None:
    print("Image not loaded")
else:
    print("Image loaded successfully")
# Function to display image properly in matplotlib

def display(img):
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12,8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
cascade_path = 'haarcascade_licence_plate_rus_16stages.xml'

plate_cascade = cv2.CascadeClassifier(cascade_path)

if plate_cascade.empty():
    print("Error loading cascade classifier")
    print("Current Working Directory:")
    print(os.getcwd())
else:
    print("Cascade classifier loaded successfully!")

def detect_plate(img):
    
    plate_img = img.copy()
    
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )
    
    for (x, y, w, h) in plates:
        
        cv2.rectangle(
            plate_img,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )
    
    return plate_img

def blur_plate(img):
    
    plate_img = img.copy()
    
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )
    
    for (x, y, w, h) in plates:
        
        # Region of Interest (ROI)
        roi = plate_img[y:y+h, x:x+w]
        
        # Blur the ROI
        blurred_roi = cv2.medianBlur(roi, 35)
        
        # Replace original ROI with blurred ROI
        plate_img[y:y+h, x:x+w] = blurred_roi
    
    return plate_img
```
### original: 

<img width="723" height="410" alt="image" src="https://github.com/user-attachments/assets/96cdd905-6666-46b9-98f7-d682605133ed" />

### Output:
<img width="723" height="410" alt="image" src="https://github.com/user-attachments/assets/9e8c3a27-d7b1-4779-9b4b-8d8d62e2e3b2" />


