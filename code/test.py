import cv2
from file_exists import *

file_path = "G:\deepfake\deepfake\dfdetect\Images"


fake_list = ["0T7VV962H7.jpg","0TFAZGTDKM.jpg","0TMADFV2AF.jpg","0VNZZ53QEI.jpg","1GP5VDAZC5.jpg"]
real_list = ["00007.jpg","00114.jpg","00152.jpg","00236.jpg","00761.jpg"]



def cont(x):
    if x in fake_list:
        path = file_path + "\\" +"fake\\"+ x
        info = "fake"
    
    elif x in real_list:
        path = file_path + "\\" +"real\\"+ x
        info = "real"
        
        
    return path,info


file_name = file_exists_check()
# while file_name!=""
# print(file_name)
# print(cont(str(file_name))[0])
# print(cont(file_name)[0])

image_data = cv2.imread(cont(file_name)[0])
# print
cv2.putText(image_data,text=cont(file_name)[1],org=(0,50),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=2,color=(0,0,255),thickness=2)
cv2.imwrite("output.png",image_data)
cv2.imshow("FRAME",image_data)
cv2.waitKey(0)

        