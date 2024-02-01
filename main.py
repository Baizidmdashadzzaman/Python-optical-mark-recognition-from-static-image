# breakdown original_image->grayscale_image->edge_image->contours->biggest_rect
# ->warp_prespective_image(birds_eye_view)->theshold_image->find_mark->save_with_score

import cv2
import numpy as np
import utlis

##############################
path = "1.jpg"
widthImg = 600
heightImg = 600
questions = 5
choices = 5
ans = [1,2,0,1,4]
##############################

img = cv2.imread(path)
img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgBiggestContours = img.copy()
imgFinal = img.copy()

####gray scal#########
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#####GaussianBlur#####
imgBlur= cv2.GaussianBlur(imgGray, (5, 5),1)

#####Image kannay edgy detect(contours)######
imgCanny = cv2.Canny(imgBlur, 10, 50)
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours,-1,(0,255,0),10) #-1 hocche index -1 hole sob nibe

# find rect angle in image
rectCon = utlis.rectContour(contours)
biggestContour = utlis.getCornerPoints(rectCon[0])
gradePoints = utlis.getCornerPoints(rectCon[1])

if biggestContour.size != 0 or gradePoints.size != 0:
          cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
          cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

          # point reorder korbe
          biggestContour = utlis.reorder(biggestContour)
          gradePoints = utlis.reorder(gradePoints)

          # birds eye view for biggest react ans section
          pt1 = np.float32(biggestContour)
          pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
          #transformation matrix
          matrix = cv2.getPerspectiveTransform(pt1,pt2)
          # birds eye view for biggest react
          imgWarpColored = cv2.warpPerspective(img,matrix, (widthImg,heightImg))

          # birds eye view for 2nd biggest react result section
          ptG1 = np.float32(gradePoints)
          ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
          # transformation matrix
          matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
          # birds eye view for 2nd biggest react
          imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
          #cv2.imshow('img', imgGradeDisplay)

          #for marking pointing threshold applied , based on pixed mark
          imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
          imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV)[1]

          #pottek row k split korbe
          boxes = utlis.splitBoxes(imgThresh)
          #cv2.imshow('imgThresh', boxes[2])
          # count korbe not zero pixels of every boxes
          # print(cv2.countNonZero(boxes[1]))

          # getting the pixle value of non zero pixeld value on box
          myPixelVal = np.zeros((questions,choices))
          countC = 0
          countR = 0
          for image in boxes:
              totalPixels = cv2.countNonZero(image)
              myPixelVal[countR][countC] = totalPixels
              countC += 1
              if (countC == choices): countR +=1 ; countC=0
          #print(myPixelVal)

          #show the invex val of my mark sheet inputed
          myIndex = []
          for x in range (0,questions) :
            arr = myPixelVal[x]
            #print("arr",arr)
            myIndexVal = np.where(arr==np.amax(arr))
            #print(myIndexVal[0])
            myIndex.append(myIndexVal[0][0])
          #print(myIndex)

          #grading process start, jodi index match hoy 1 hobe mane true ar na hole 0 mane false
          grading=[]
          for x in range (0,questions) :
              if ans[x] == myIndex[x]:
                  grading.append(1)
              else: grading.append(0)
          #print(grading)

          # final grade calculation
          score = (sum(grading)/questions) * 100
          #print(score)

          # displaying ans and wrong ans in color
          imgResult = imgWarpColored.copy()
          imgResult = utlis.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
          #only show the dot ans
          imgRawDrawing = np.zeros_like(imgWarpColored)
          imgRawDrawing = utlis.showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)
          # now placing the warp view in the image
          invmatrix = cv2.getPerspectiveTransform(pt2, pt1)
          imgInvWarp = cv2.warpPerspective(imgRawDrawing, invmatrix, (widthImg, heightImg))

          #grade in image show process for 2nd largest box
          imgRawGrade = np.zeros_like(imgGradeDisplay)
          cv2.putText(imgRawGrade, str(int(score))+"%", (50,100),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,255),3)
          #cv2.imshow("Grade", imgRawGrade)
          # now placing the warp grade mark in the image
          invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
          imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

          imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp,1,0)
          imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)
#print(gradePoints)


imgBlank = np.zeros_like(img)
imageArray = ([img,imgGray,imgBlur,imgCanny],
              [imgContours,imgBiggestContours,imgWarpColored,imgThresh],
              [imgResult,imgRawDrawing,imgInvWarp,imgFinal]
             )
lables=[
    ["Original","Gray","Blur","Canny"],
    ["Contours","Biggest con","Warp","Threshold"],
    ["Result","Raw drawing","Inverse warp","Final result"]
]
imgStacked = utlis.stackImages(imageArray,0.4,lables)

cv2.imshow("Final output", imgFinal)
cv2.imshow("Full process", imgStacked)

cv2.waitKey(0)
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Asadzaman')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
