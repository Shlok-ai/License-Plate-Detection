import cv2
import pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
cascade=cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

states={"AN":"Andaman and Nicobar","AP":"Andra Pradesh","AR":"Arunachal Pradesh",
        "CH":"Chattisgarh","DL":"Delhi","DN":"Dadra and Nagar Havelli","DD":"Daman and diu",
        "HR":"Haryana","HP":"Himachal Pradesh","JK":"Jammu and Kashmir",
        "MP":"Madhya Pradesh","MH":"Mahrashtra","MN":"Manipur","PY":"Pondicherry",
        "PN":"Punjab","RJ":"Rajasthan","UP":"Uttar Pradesh","TS":"Telengana",
        "WB":"West Bengal","CG":"Chattisgarh","TS":"Telengana","KL":"Kerela"
        }
def extract_num(img_name):
    global read
    # Detect the no.plate
    img=cv2.imread(img_name)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nplate=cascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in nplate:
        a,b=(int(0.02*img.shape[0]),int(0.025*img.shape[1]))
        plate=img[y+a:y+h-a,x+b:x+w-b, :]
        # Image processing technique
        kernel=np.ones((1,1),np.uint8)
        plate=cv2.dilate(plate,kernel,iterations=1)
        plate=cv2.erode(plate,kernel,iterations=1)
        plate_gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        (thresh,plate)=cv2.threshold(plate_gray,127,225,cv2.THRESH_BINARY_INV)

        # Recognise the number plate through OCR pyteserract engine
        read=pytesseract.image_to_string(plate_gray)
        print(read)
        read=''.join(e for e in read if e.isalnum())
        stat=read[0:2]
        try:
            print("Car Belongs to",states[stat])
        except:
            print("State not recognised!!!")
        print(read)
        cv2.rectangle(img,(x,y),(x+w,y+h),(51,51,255),2)
        cv2.rectangle(img,(x,y - 40),(x+w,y),(51,51,255),-1)
        cv2.putText(img,read, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.imshow("Result",img)
    cv2.imwrite('result.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

extract_num('./img/carimg3.png')




