import cv2

method = cv2.TM_SQDIFF_NORMED 

# Read the images from the file
crops = cv2.imread('C:/Users/admin/Desktop/crops/2aa82aa0-a2fd-518d-8541-a135c51e1f85.jpg')  # Copy the image link
imagess = cv2.imread('C:/Users/admin/Desktop/images/c833b0d0-de9b-530a-8619-39aa6adf346d.jpg') #Copy the crop link

result = cv2.matchTemplate(crops, imagess, method)

# We want the minimum squared difference
mn,_,mnLoc,_ = cv2.minMaxLoc(result)

# Draw the rectangle and extract the coordinates of our best match
MPx,MPy = mnLoc

# Step 2: Get the size of the template. This is the same size as the match.
trows,tcols = crops.shape[:2]

# Step 3: Draw the rectangle on large_image
cv2.rectangle(imagess, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)

# Display the original image with the rectangle around the match where cropped image is present
cv2.imshow('output',imagess)
print(mnLoc)
print(tcols,trows)


