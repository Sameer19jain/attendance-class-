import os
import pickle
import cv2
import cvzone
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import datetime
cred=credentials.Certificate('student-busAttendance.json')
firebase_admin.initialize_app(cred,{
    'databaseURL':'https://student-bus-boarding-default-rtdb.firebaseio.com',
    'storageBucket':'student-bus-boarding.appspot.com'

})

cap=cv2.VideoCapture(0)
cap.set(3 , 640)
cap.set(4 ,480)
imgBackground = cv2.imread('Resources/background.png')

bucket = storage.bucket()
folderModePath='Resources/Modes'
modePathList=os.listdir(folderModePath)

imgModeList=[]

for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))
# Load the encoding file

file=open('EncodeFile.p','rb')
print('Loading Encode File ...')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown,studentIds=encodeListKnownWithIds
#print(studentIds)
print('done Encode File ...')


counter=0
modeType=0
id=-1
while True:

    success,img = cap.read()

    imgS=cv2.resize(img,(0,0),None , 0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceCurrFrame=face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)
    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[3]


    for encodeFace, faceLoc in zip(encodeCurrFrame,faceCurrFrame):
         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
         faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
         print('matches',matches)
         print('faceDis',faceDis)

         matchIndex=np.argmin(faceDis)
         if matches[matchIndex]:
            #print('matchIndex',matchIndex)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
            bbox =55+x1,162*y1,x2-x1,y2-y1
            id = studentIds[matchIndex]
            imgBackground=cvzone.cornerRect(imgBackground,bbox,rt=0)
            if counter == 0:
                cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                cv2.imshow("Face Attendance", imgBackground)
                cv2.waitKey(1)
                counter = 1
                modeType = 1
    if counter != 0:

        if counter == 1:
            # Get the Data
            print("the student id",id)
            studentInfo = db.reference(f'studentBoarding/data/{id}').get()
            print(studentInfo)

            # Update data of attendance
            datetimeObject = datetime.datetime.strptime(studentInfo['AttendanceTime'],
                                               "%Y-%m-%d %H:%M:%S")
            secondsElapsed = (datetime.datetime.now() - datetimeObject).total_seconds()
            print(secondsElapsed)
            if secondsElapsed > 30:
                ref = db.reference(f'studentBoarding/data/{id}')

                if int(id)>0:
                    ref.child('AttendanceTime').set(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            else:
                modeType = 3
                counter = 0
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    cv2.imshow("Face Attendance",imgBackground)
    cv2.waitKey(1)