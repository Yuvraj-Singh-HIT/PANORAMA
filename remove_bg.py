import cv2
import numpy as np

def remove_background():
    # Read image
    img = cv2.imread('assets/logo.png')
    h, w = img.shape[:2]
    
    # Create mask for floodFill (requires size + 2)
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Flood fill from all 4 corners to ensure all connected black background is caught
    # We use (255, 0, 255) (Magenta) as a temporary fill color
    for pt in [(0,0), (w-1,0), (0,h-1), (w-1,h-1)]:
        cv2.floodFill(img, mask, pt, (255, 0, 255), (30, 30, 30), (30, 30, 30), cv2.FLOODFILL_FIXED_RANGE)
    
    # Convert to BGRA (add alpha channel)
    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # Find all magenta pixels and set them to completely transparent
    is_magenta = (img[:,:,0] == 255) & (img[:,:,1] == 0) & (img[:,:,2] == 255)
    img_bgra[is_magenta] = [0, 0, 0, 0]
    
    # Save the processed image
    cv2.imwrite('assets/logo.png', img_bgra)
    print("Background removed successfully.")

if __name__ == "__main__":
    remove_background()
