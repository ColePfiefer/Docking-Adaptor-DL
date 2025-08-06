

import cv2
import numpy as np
from itertools import combinations
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

img = cv2.imread("dataset\IMG_10072025161331.png",0)
# Gaussian blur to reduce noise
img1 = cv2.GaussianBlur(img, (5, 5), 0)
# CLAHE
clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(11,11))
img2 = clahe.apply(img)
# Morphological gradient
# kernal = np.ones((3,3),np.uint8)
# img3 = cv2.morphologyEx(th2,cv2.MORPH_GRADIENT, kernal)
# Thresholding
# Adaptive thresholding
th1 = cv2.adaptiveThreshold( img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 63, 1)
th2 = cv2.adaptiveThreshold( img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 63, 1)
# OTSU thresholding
# ret,th3 = cv2.threshold(img3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Inverted image
img4 = cv2.bitwise_not(th2)
kernal = np.ones((3,3),np.uint8)
img3 = cv2.morphologyEx(th2,cv2.MORPH_GRADIENT, kernal)

# Contours
contour_img = np.zeros_like(img)
filtered_contours = []
filtered_hierarchy = []
        
contours, hierarchy = cv2.findContours(img4,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for i, cnt in enumerate(contours):
#   approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    area = cv2.contourArea(cnt)
#   n = len(approx)
    if area > 50:
        filtered_contours.append(cnt)
        filtered_hierarchy.append(hierarchy[0][i])
round_contours = []
other_contours = []
centroids_round_contours = []
round_contours_img = np.zeros_like(img)
for i, cnt in enumerate(filtered_contours):
    
    if len(cnt) < 5:
        other_contours.append(cnt)
        continue
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        other_contours.append(cnt)
        continue
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    if len(cnt.shape) == 3 and cnt.shape[1] ==1:
        pts = cnt.squeeze()
    if len(pts.shape) != 2:
        other_contours.append(cnt)
        continue
    dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    mean_r = np.mean(dists)
    std_r = np.std(dists)
    ratio = std_r / mean_r
    ellipse = cv2.fitEllipse(cnt)
    (center, axes, angle) = ellipse
    major_axis, minor_axis = max(axes), min(axes)
    ratio_ellipse = minor_axis / major_axis
#     if ratio <= 0.1:
    if ratio <= 0.1 and ratio_ellipse >= 0.9:
        
#     if ratio_ellipse >= 0.87:
        round_contours.append(cnt)
        centroids_round_contours.append((cx,cy,mean_r))
        
          
    else:
        other_contours.append(cnt)
        
# Seperate inner and outer circles on different images
# Unique centroids
unique_centroids_round_contours = []
unique_centroids_round_contours_centroid = []

for i, c in enumerate(centroids_round_contours):
    is_unique = True
    for uc in unique_centroids_round_contours:
        dist = np.linalg.norm(np.array(c[0:2]) - np.array(uc[0:2]))
#         if dist/uc[2] < 0.01:
        if dist < 5:
            is_unique = False
            break
    if is_unique:
        unique_centroids_round_contours.append(c)
        unique_centroids_round_contours_centroid.append(c[0:2])
        


# for i, cnt in enumerate(round_contours):
unique_centroids_round_contours_np = np.array(unique_centroids_round_contours_centroid)
valid_targets = []
for combo in combinations(range(len(unique_centroids_round_contours_np)), 4):
    pts = unique_centroids_round_contours_np[list(combo)]
    dists_1 = pdist(pts) 
    ratios = dists_1 / np.max(dists_1)
    if np.std(ratios) < 0.2: 
        valid_targets.append(pts)   

cv2.drawContours(contour_img,filtered_contours,-1,255,3)
cv2.drawContours(round_contours_img,round_contours,-1,255,3)
for i,u_c in enumerate(unique_centroids_round_contours):
    cv2.circle(round_contours_img,(int(u_c[0]),int(u_c[1])),2,255,-1)
    
    
# for idx, group in enumerate(valid_targets, start=1):
#     cluster_img = np.zeros_like(img)
#     for pt in group:
#         cv2.circle(cluster_img, (int(pt[0]), int(pt[1])), 5, 255, -1)
#     filename = f"group_{idx}_dots.png"
#     cv2.imwrite(filename, cluster_img)
    
    
    
plt.figure(figsize=(10, 5))    
results = []
for group in valid_targets:
    pts2 = np.array(group, dtype=np.float32)
    mean_dist = np.mean(pdist(pts2))
    area1 = cv2.contourArea(pts2.astype(np.int32))
    results.append({'pts': pts2, 'mean_dist': mean_dist, 'area': area1})

# Sort by mean_dist
results_sorted = sorted(results, key=lambda x: x['mean_dist'])

# Compare first and second
if len(results_sorted) >= 2:
    r1 = results_sorted[0]
    r2 = results_sorted[1]
    
    mean_dist_ratio = r2['mean_dist'] / r1['mean_dist'] if r1['mean_dist'] != 0 else float('inf')
    area_ratio = r2['area'] / r1['area'] if r1['area'] != 0 else float('inf')

    print(f"Mean Dist Ratio (2/1): {mean_dist_ratio:.3f}")
    print(f"Area Ratio (2/1): {area_ratio:.3f}")

    if 0.9 <= mean_dist_ratio <= 1.1 and 0.9 <= area_ratio <= 1.1:
#         print("Group 1 and Group 2 are similar (separated group pair)")
        groups_to_keep = [r1, r2]
    else:
#         print("Only Group 1 is accepted as separated group")
        groups_to_keep = [r1]
elif len(results_sorted) == 1:
#     print("Only one group found")
    groups_to_keep = [results_sorted[0]]
else:
#     print("No valid groups found")
    groups_to_keep = []
centroids_for_pose_estimation = []   
for idx, group in enumerate(groups_to_keep, start=1):
    cluster_img = np.zeros_like(img)
    centroids_for_pose_estimation.append(group['pts'])
    for pt in group['pts']:
        cv2.circle(cluster_img, (int(pt[0]), int(pt[1])), 5, 255, -1)
    cv2.imwrite(f"selected_group_{idx}.png", cluster_img)
    plt.subplot(4, 3, 9+idx)
    plt.imshow(cluster_img, cmap='gray')
    plt.title(f'centroids_for_pose_estimation {idx}')
    plt.axis('off')  # Turn off axis labels

# cv2.drawContours(round_contours_img,centroids_round_contours,-1,255,3)      
#Canny Edge Detection
# canny_th1 = cv2.Canny(th1,50, 215)
# canny_th2 = cv2.Canny(th2,50, 215)
# canny_th3 = cv2.Canny(th3,50, 215)
#Dilation

# Segmentation and clustering

# cv2.imshow("Image window", canny_th1)
# cv2.waitKey(0)

cv2.imwrite('clahe_image.png', img2)
# plt.figure(figsize=(10, 5))
plt.subplot(4, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')  # Turn off axis labels

plt.subplot(4, 3, 2)
plt.imshow(img1, cmap='gray')
plt.title('Guassian blur image')
plt.axis('off')  # Turn off axis labels

plt.subplot(4, 3, 3)
plt.imshow(img2, cmap='gray')
plt.title('CLAHE image')
plt.axis('off')  # Turn off axis labels


plt.subplot(4, 3, 4)
plt.imshow(th1, cmap='gray')
plt.title('Threshold image guassian')
plt.axis('off')  # Turn off axis labels

plt.subplot(4, 3, 5)
plt.imshow(th2, cmap='gray')
plt.title('Threshold image mean')
plt.axis('off')  # Turn off axis labels

plt.subplot(4, 3, 6)
plt.imshow(img3, cmap='gray')
plt.title('Morph gradient')
plt.axis('off')  # Turn off axis labels

plt.subplot(4, 3, 7)
plt.imshow(img4, cmap='gray')
plt.title('Inverted image')
plt.axis('off')  # Turn off axis labels

plt.subplot(4, 3, 8)
plt.imshow(contour_img, cmap='gray')
plt.title('Contour image')
plt.axis('off')  # Turn off axis labels

plt.subplot(4, 3, 9)
plt.imshow(round_contours_img, cmap='gray')
plt.title('Round Contour image')
plt.axis('off')  # Turn off axis labels




# Show the plot
plt.show()


