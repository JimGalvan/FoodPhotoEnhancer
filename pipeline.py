import cv2
from food_enhancement_cv import enhance_food_plate

img = cv2.imread("plate_crops/plate_0.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

enhanced = enhance_food_plate(img)

cv2.imwrite(
    "plate_crops/plate_0_enhanced.png",
    cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
)
