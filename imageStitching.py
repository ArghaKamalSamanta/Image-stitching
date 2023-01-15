import cv2
import numpy as np
import imutils

cv2.ocl.setUseOpenCL(False)

trainImg = cv2.imread('foto5A.jpg')
trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)

queryImg = cv2.imread('foto5B.jpg')
queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_BGR2GRAY)

# Extracting key points and making SIFT descriptor
descriptor = cv2.SIFT_create()
keypointA, descriptorA = descriptor.detectAndCompute(trainImg_gray, None)
keypointB, descriptorB = descriptor.detectAndCompute(queryImg_gray, None)

# Matching descriptors
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
best_matches = bf.match(descriptorA, descriptorB)
print(len(best_matches))

# Sorting the matched descriptors in order of distance
# matches = sorted(best_matches, key=lambda x: x.distance)

# Converting the key points to numpy arrays
keypointA = np.float32([kp.pt for kp in keypointA])
keypointB = np.float32([kp.pt for kp in keypointB])

# Constructing the two sets of matched points
ptsA = np.float32([keypointA[m.queryIdx] for m in best_matches])
ptsB = np.float32([keypointB[m.trainIdx] for m in best_matches])

# Estimating the homography between the sets of points
homography, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4)

# Applying panorama correction
width = trainImg.shape[1] + queryImg.shape[1]
height = trainImg.shape[0] + queryImg.shape[0]

result = cv2.warpPerspective(trainImg, homography, (width, height))
result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg

# Transforming the panorama image into grayscale and thresholding it
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

# Finding contours from the binary image
contour = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = imutils.grab_contours(contour)

# Finding the maximum contour area
c = max(contour, key=cv2.contourArea)

# Fetching info of a bbox from the contour area
x, y, w, h = cv2.boundingRect(c)

# Cropping the image according to the bbox coordinates
result = result[y:y + h, x:x + w]

cv2.namedWindow("Train image", cv2.WINDOW_NORMAL)
cv2.imshow("Train image", trainImg)

cv2.namedWindow("Query image", cv2.WINDOW_NORMAL)
cv2.imshow("Query image", queryImg)

cv2.namedWindow("Stitched image", cv2.WINDOW_NORMAL)
cv2.imshow("Stitched image", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
