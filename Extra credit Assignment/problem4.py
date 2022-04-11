import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import heapq


###############################################################################################################################
                                                # PADDING IMAGE FUNCTION #
###############################################################################################################################
def padding(image, paddingSize):
    xAxis = image.shape[0] + paddingSize * 2
    yAxis = image.shape[1] + paddingSize * 2
    paddedImage = np.zeros((xAxis,yAxis))

    for row in range(paddingSize):
        for column in range(paddingSize):
            paddedImage[row][column] = image[0][0]
            paddedImage[row+xAxis-paddingSize][column] = image[image.shape[0]-1][0]
            paddedImage[row+xAxis-paddingSize][column+yAxis-paddingSize] = image[image.shape[0]-1][image.shape[1]-1]
            paddedImage[row][column+yAxis-paddingSize] = image[0][image.shape[1]-1]

    for row in range(image.shape[0]):
        paddedImage[row+paddingSize][0:paddingSize] = image[row][0]
        paddedImage[row+paddingSize][paddingSize:yAxis-paddingSize] = image[row]
        paddedImage[row+paddingSize][yAxis-paddingSize:yAxis] = image[row][image.shape[1]-1]
    paddedImage = paddedImage.T

    for row in range(image.shape[1]):
        paddedImage[row+paddingSize][0:paddingSize] = image[0][row]
        paddedImage[row+paddingSize][xAxis-paddingSize-1:xAxis] = image[image.shape[0]-1][row]
    paddedImage = paddedImage.T
    return paddedImage


def removePadding(image, paddingSize):
    xAxis = image.shape[0] - paddingSize * 2
    yAxis = image.shape[1] - paddingSize * 2
    nonPaddedImage = np.zeros((xAxis, yAxis))
    for row in range(xAxis):
        nonPaddedImage[row] = image[row + paddingSize][paddingSize:paddingSize+yAxis]
    return nonPaddedImage

###############################################################################################################################
                                                        # CONVOLUTION #
###############################################################################################################################
def sobelConvolution(image, kernel):
    paddingXAxis = kernel.shape[0]//2
    paddingYAxis = kernel.shape[1]//2
    image = padding(image, paddingXAxis)
    xAxis, yAxis = image.shape
    paddedImage = np.copy(image)
    for row in range(paddingXAxis,xAxis-paddingXAxis):
        for column in range(paddingYAxis,yAxis-paddingYAxis):
            total = 0
            for i in range(-1 *  paddingXAxis, paddingXAxis + 1):
                for j in range(-1 * paddingYAxis , paddingYAxis + 1):
                    total += kernel[paddingXAxis + i][paddingYAxis + j] * image[row - i][column - j]
            paddedImage[row][column] = total
    image = removePadding(image,paddingXAxis)
    paddedImage = removePadding(paddedImage,paddingXAxis)
    return paddedImage


def gaussianConvolution(image, sigma):
    padding = sigma * 3
    gaussian = [(sigma ** -1) * (2 * np.pi) ** (-1/2) * np.exp((-1/2) * (x/sigma)**2) for x in range(-1 * padding, padding+1)]
    gaussian = np.outer(gaussian, gaussian)
    image = sobelConvolution(image, gaussian)
    return image


###############################################################################################################################
                                                        # GRADIENT #
###############################################################################################################################
def gradientInfo(xGradient, yGradient, threshold):
    xAxis, yAxis = xGradient.shape
    magnitude = np.zeros(xGradient.shape)
    direction = np.zeros(xGradient.shape)
    for i in range(xAxis):
        for j in range(yAxis):
            distance =  (xGradient[i][j] ** 2 + yGradient[i][j] ** 2) ** .5
            if (distance > threshold):
                magnitude[i][j] = distance
    direction = np.arctan2(yGradient,xGradient) * 180 / np.pi
    return magnitude,direction

def maxValue(ret, magnitude, row, col, x, y):
    if ((magnitude[row + x][col + y] > magnitude[row][col]) or (magnitude[row - x][col -y ] > magnitude[row][col])):
        ret[row][col] = 0
    else:
        ret[row][col] = magnitude[row][col]

def nonMaxSuppression(magnitude, direction):
    padding(magnitude, 1)
    xAxis, yAxis = magnitude.shape   
    ret = np.zeros(magnitude.shape)
    for row in range(1, xAxis-1):
        for col in range(1, yAxis-1):
            gradientDirection = direction[row][col]
            if ((gradientDirection > -22.5) and (gradientDirection <= 22.5) or (gradientDirection > 157.5) and (gradientDirection <= -157.5)):
                maxValue(ret, magnitude, row, col, 1, 0)
            elif ((gradientDirection > 22.5) and (gradientDirection <= 67.5) or (gradientDirection > -157.5) and (gradientDirection <= -112.5)):
                maxValue(ret, magnitude, row, col, 1, 1)
            elif ((gradientDirection > 67.5) and (gradientDirection < 112.5) or (gradientDirection > -112.5) and (gradientDirection < -67.5)):
                maxValue(ret, magnitude, row, col, 1, 0)
            else:
                maxValue(ret, magnitude, row, col, 1, -1)
    removePadding(ret, 1)
    removePadding(magnitude, 1)
    return ret



###############################################################################################################################
                                                    # SNIC FUNCTIONS #
###############################################################################################################################

def initialCenters(image):
    colors = []
    centers = []
    xsize, ysize, _ = image.shape
    xblocks = xsize // 50 
    yblocks = ysize // 50
    for i in range(xblocks):
        for j in range(yblocks):
            x = 25 + i * 50
            y = 25 + j * 50
            centers.append([x,y])
            colors.append(image[x][y])
    return centers, colors

def updateCluster(clusters, index, element):
    cluster = clusters[index]
    cluster[5] += 1
    elements = cluster[5]
    cluster[0] = ((cluster[0] * (elements-1)) + element[0]) / cluster[5]
    cluster[1] = ((cluster[1] * (elements-1)) + element[1])  / cluster[5]
    for i in range(2,5):
        cluster[i] = ((cluster[i] * (elements-1)) + element[i]) / cluster[5]
    return clusters[index]

def squaredDifference(a,b):
    total = 0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return total


def elementDistance(a,b,s,m):
    pixelDistance = squaredDifference([a[0],a[1]],[b[0],b[1]]) / s
    colorDistance = squaredDifference(a[2:],b[2:]) / m 
    return (pixelDistance + colorDistance) ** (1/2)

def snic(image, compactnessFactor, centers, colors):
    m = compactnessFactor
    heap = []
    s = (image.shape[0] * image.shape[1] / len(centers)) ** (1/2)
    xsize, ysize, _ = image.shape    
    ret = np.zeros(image.shape)
    labelMap = np.zeros((xsize,ysize))
    clusters = [[0,0,0,0,0,0] for i in range(len(centers))]
    directions =[ [-1, 0], [-1,-1] , [-1,1] , [1,-1], [1, 0] , [1,1], [0,1], [0,-1] ]
    for i in range(len(centers)):
        center = centers[i]
        color = colors[i]
        element = (0,i+1,color[0],color[1],color[2],center[0],center[1])
        heapq.heappush(heap,element)
    while len(heap) != 0:
        target = heapq.heappop(heap)
        distance, k, colorR, colorG, colorB, x, y = target
        if labelMap[x][y] == 0: 
            labelMap[x][y] = k           
            old = updateCluster(clusters, k-1, [x,y,colorR,colorG,colorB])
            for move in directions:
                moveX = x + move[0]
                moveY = y + move[1]
                if moveX >= 0 and moveX < xsize and moveY >= 0 and moveY < ysize:
                    if labelMap[moveX][moveY] == 0:
                        targetColor = image[moveX,moveY]
                        target = [moveX, moveY, targetColor[0], targetColor[1],targetColor[2]]
                        d = elementDistance(target,old,s,m)
                        e = (d, k, targetColor[0], targetColor[1], targetColor[2], moveX, moveY)
                        heapq.heappush(heap,e)
    for i in range(xsize):
        for j in range(ysize):
            clusterIndex = labelMap[i][j]
            target = clusters[int(clusterIndex-1)]
            ret[i][j] = [target[2]/255,target[3]/255,target[4]/255]

    return labelMap, ret


def drawBorders(labelMap, ret):
    xsize, ysize = labelMap.shape
    for i in range(xsize-1):
        for j in range(ysize-1):
            if labelMap[i][j] != labelMap[i+1][j] or labelMap[i][j] != labelMap[i][j+1]:
                ret[i][j] = [0,0,0]
    return ret
    
###############################################################################################################################
                                                    # MAIN FUNCTION #
###############################################################################################################################
image = cv.imread("wt_slic.png")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
imageSlic = image.astype('float32')

centers, colors = initialCenters(image)

labelMap, ret = snic(image, 1, centers, colors)
plt.imshow(ret)
plt.show()

snicWithBorders = drawBorders(labelMap, ret)
plt.imshow(snicWithBorders)
plt.show()