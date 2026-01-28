from tempfile import template
import cv2
import torch
import fastcv 

img = cv2.imread("artifacts/pg.png", cv2.IMREAD_UNCHANGED)
template = cv2.imread("artifacts/ball.png", cv2.IMREAD_UNCHANGED)

#copy of an image for display
img_display = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) #we want only RGB from image

img_tensor = torch.from_numpy(img).float().cuda()
template_tensor = torch.from_numpy(template).float().cuda()
img_tensor = img_tensor.permute(2,0,1) #converts into format channels x height x width
template_tensor = template_tensor.permute(2,0,1)

#function returns x and y coordinates of the mid of the matched template
x,y = fastcv.template_match(img_tensor, template_tensor)


#template.shape -> height, width, channels
height, width, _ = template.shape
cv2.rectangle(img_display, (x - width//2,y - height//2), (x+width//2, y+height//2), (255,0,0), 2)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("output", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()


