import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

###############################################################################################################################
                                                    # FILTER FUNCTION #
###############################################################################################################################
def imageFilter(kernalSize, sigma, ogImage):
    
    def convolution(paddedImage, kernal, ogImage):
        opRow, opCol = ogImage.shape
        outputImage = np.zeros((opRow, opCol))
        kernalSize = len(kernal)
    
        for i in range(opRow):
            for j in range(opCol):
                outputImage[i, j] = np.sum(kernal * paddedImage[i:i+kernalSize, j:j+kernalSize])
    
        return outputImage
    
    temp = np.linspace(-(kernalSize // 2), kernalSize // 2, kernalSize) # If size is 3, the array will be [-1, 0, 1]
    gaussian1dKernal = list()
   
    for i in temp:
        gaussian1dKernal.append((1/(np.sqrt(2*np.pi) * sigma))* (np.e ** ((-np.square(i))/(2*np.square(sigma))))) # Creates the 1D Gaussian Kernal(a.k.a. filter)

    gaussian2dKernal = np.outer(gaussian1dKernal, gaussian1dKernal) # Creates the Gaussian 2D Kernal that we need
    kernal = gaussian2dKernal/(np.sum(gaussian2dKernal))

    padSize = len(kernal) // 2
    cv.imwrite("Original_Image.png", ogImage)
    paddedImage = np.pad(ogImage, padSize, 'edge')
    # cv.imwrite("Padded_image.png", paddedImage)
    gaussianOutput = convolution(paddedImage, kernal, ogImage)
    cv.imwrite("Gaussian_Output_Image.png", gaussianOutput)

    plt.imshow(gaussianOutput, cmap = 'gray')
    plt.title("Gaussian Filtered Image")
    plt.show()

    xAxisSobelKernal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    yAxisSobelKernal = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    padSize_Sobel = len(xAxisSobelKernal) // 2
    paddedImage_Sobel = np.pad(gaussianOutput, padSize_Sobel, 'edge')
    edgesXAxis = convolution(paddedImage_Sobel, xAxisSobelKernal, ogImage)
    edgesYAxis = convolution(paddedImage_Sobel, yAxisSobelKernal, ogImage) 
    # plt.imshow(edgesXAxis, cmap = 'gray')
    # plt.title("X-Axis Output")
    # plt.show()
    # plt.imshow(edgesYAxis, cmap = 'gray')
    # plt.title("Y-Axis Output")
    # plt.show()

    gradientMagnitude = np.sqrt(np.square(edgesXAxis) + np.square(edgesYAxis))
    gradientMagnitude *= 255.0 / gradientMagnitude.max()
    cv.imwrite("Gradient_Magnitude.png", gradientMagnitude)

    plt.imshow(gradientMagnitude, cmap = 'gray')
    plt.title("Gradient Magnitude")
    plt.show()

    return gradientMagnitude   
    
###############################################################################################################################
                                                # NON-MAXIMUM SUPPRESSION #
###############################################################################################################################
def nonMaxSuppression (filteredImage):
    paddedImage = np.pad(filteredImage, 1, 'edge')
    angleRows, angleCols = paddedImage.shape
    angleMatrix = np.zeros_like(filteredImage)

    for i in range(angleRows - 2):
        for j in range(angleCols - 2):
            window = paddedImage[i:i+3, j:j+3]
            horizontalCheck = np.abs(window[1, 0] - window[1, 2]) 
            verticalCheck = np.abs(window[0, 1] - window[2, 1])
            diagonalLTopToRBtmCheck = np.abs(window[0, 0] - window[2, 2]) # ↘↖ Direction
            diagonalLBtmToRTopCheck = np.abs(window[0, 2] - window[2, 0]) # ↗↙ Direction

            if(verticalCheck > max(horizontalCheck, diagonalLBtmToRTopCheck, diagonalLTopToRBtmCheck)):
                angleMatrix[i, j] = 270 # ⬆⬇ Directon
            elif (horizontalCheck > max(verticalCheck, diagonalLBtmToRTopCheck, diagonalLTopToRBtmCheck)):
                angleMatrix[i, j] = 180 # ➡⬅ Direction
            elif (diagonalLTopToRBtmCheck > max(horizontalCheck, verticalCheck, diagonalLBtmToRTopCheck)):
                angleMatrix[i, j] = 135 # ↘↖ Direction
            elif (diagonalLBtmToRTopCheck > max(horizontalCheck, verticalCheck, diagonalLTopToRBtmCheck)):
                angleMatrix[i, j] = 225 # ↗↙ Direction

    edgeMatrix = np.zeros_like(filteredImage)
    for i in range(angleRows - 2):
        for j in range(angleCols - 2):
            window = paddedImage[i:i+3, j:j+3]

            if(angleMatrix[i, j] == 270):
                if(window[1, 1] > max(window[0, 1], window[2, 1])):
                    edgeMatrix[i, j] = filteredImage[i, j]
            elif(angleMatrix[i, j] == 180):
                if(window[1, 1] > max(window[1, 0], window[1, 2])):
                    edgeMatrix[i, j] = filteredImage[i, j]
            elif(angleMatrix[i, j] == 135):
                if(window[1, 1] > max(window[0, 0], window[2, 2])):
                    edgeMatrix[i, j] = filteredImage[i ,j]
            elif(angleMatrix[i, j] == 225):
                if(window[1, 1] > max(window[0, 2], window[2, 0])):
                    edgeMatrix[i, j] = filteredImage[i, j]
    
    # plt.imshow(edgeMatrix, cmap = 'gray')
    # plt.title("edge_detection")
    # plt.show()

    return edgeMatrix


###############################################################################################################################
                                                    # MAIN FUNCTION #
###############################################################################################################################
sigma = int(input("Enter the sigma for the Gaussian Filter: ") or 1)
# if(ogImage)
filterSize = sigma * 6
if(filterSize % 2 == 0):
    filterSize -= 1

ogImage = cv.imread(input("Enter the filename or provide the path to the image: ") or 'plane.pgm')
ogImage = cv.cvtColor(ogImage, cv.COLOR_BGR2GRAY)

filteredImage = imageFilter(filterSize, sigma, ogImage)

nmsImage = nonMaxSuppression(filteredImage)
cv.imwrite("NMS_Image.png", nmsImage)
plt.imshow(nmsImage, cmap = 'gray')
plt.title("Non-Maximum Suppression")
plt.show()