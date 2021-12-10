import os
path = "G:\deepfake\deepfake\dfdetect\media"
file_list = os.listdir(path)

def file_exists_check():
    if file_list != []:
        for files in file_list:
            if files.endswith(".png") or files.endswith(".jpg") or files.endswith(".jpeg"):
                # image_path = path + "\\" + files
                image_path =  files
                
               
    return image_path
            
            
print(file_exists_check())