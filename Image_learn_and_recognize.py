#!/usr/bin/env python
# coding: utf-8

# In[12]:


import matplotlib.pyplot as plt
import numpy as np
import cv2 


# In[13]:


face_detection=cv2.CascadeClassifier('/Users/amlanjyotipatnaik/haars/opencv/data/haarcascades/haarcascade_frontalface_default.xml')


# In[14]:


path='/Users/amlanjyotipatnaik/Documents/leaning_photo/'


# In[15]:


#import sqlite3
#conn = sqlite3.connect(path+'subjects.db')
#cursor=conn.cursor()
#result=cursor.execute("select max(subject_id) from subject_xref").fetchall()
#
#for res in result:
#    currMaxID=res[0]
#
#print(currMaxID)
#currMaxID=currMaxID+1
#cursor.close()
#cursor=conn.cursor()
#cursor.execute("insert into subject_xref values("+str(currMaxID)+",\'PQR\')",)
#conn.commit()
#cursor.close()


# In[16]:


import sqlite3
conn = sqlite3.connect(path+'subjects.db')
cursor=conn.cursor()


# In[17]:


##Function to check user existance in DB
def check_user_existance_in_db(db,usr):
    usr=usr.upper()
    ret=False
    conn = sqlite3.connect(db)
    cursor=conn.cursor()
    sql_existing='select subject_id as id from subject_xref where upper(subject_name)='+"'"+usr+"'"
    result=cursor.execute(sql_existing).fetchall()
    for res in result:
        if res[0] is None:
            ret=False
        else:
            ret=True
    cursor.close()
    return ret

    
    


# In[18]:


##Function to get max ID from DB
def get_max_ID(db):
    
    ret=-1
    conn = sqlite3.connect(db)
    cursor=conn.cursor()
    sql_max='select max(subject_id) as id from subject_xref'
    result=cursor.execute(sql_max).fetchall()
    for res in result:
        if res[0] is None:
            ret=-1
        else:
            ret=res[0]
    cursor.close()
    return ret


# In[19]:


##Function to insert user xref details to db
def insert_subject_details(db,sid,usr):
    usr=usr.upper()
    ret=-1
    conn = sqlite3.connect(db)
    cursor=conn.cursor()
    sql_insert='insert into subject_xref values('+str(sid)+','+"'"+usr+"'"+')'
   
    cursor.execute(sql_insert)
    conn.commit()
    cursor.close()
    print(sql_insert)
   


# In[20]:


##Function to get subject name from DB
def get_subject_Name(db,sid):
    
    ret='unknown'
    conn = sqlite3.connect(db)
    cursor=conn.cursor()
    sql_subject_name='select subject_name  from subject_xref where subject_id='+str(sid)
    result=cursor.execute(sql_subject_name).fetchall()
    for res in result:
        if res[0] is None:
            ret='unknown'
        else:
            ret=res[0]
    cursor.close()
    return ret


# In[21]:


##Function to get subject ID from DB
def get_subject_ID(db,usr):
    usr=usr.upper()
    ret=-1
    conn = sqlite3.connect(db)
    cursor=conn.cursor()
    sql_existing='select subject_id as id from subject_xref where upper(subject_name)='+"'"+usr+"'"
    result=cursor.execute(sql_existing).fetchall()
    for res in result:
        if res[0] is None:
            ret=-1
        else:
            ret=res[0]
    cursor.close()
    return ret


# In[22]:


#check_user_existance_in_db(path+'subjects.db','pst')
#get_max_ID(path+'subjects.db')
#insert_subject_details(path+'subjects.db',5,'stp')
#get_subject_Name(path+'subjects.db',0)
get_subject_ID(path+'subjects.db','AJ')


# In[ ]:





# # Capturing subjet photo and storing it in disk for learning

# In[36]:


import time
vc=cv2.VideoCapture(0)
rval,cap=vc.read()
no_of_photo_of_subject=0
name=input("Enter Subject's Name : ")
name=name.upper()
if(check_user_existance_in_db(path+'subjects.db',name)):
    print('existing subject...')
    subjectId=get_subject_ID(path+'subjects.db',name)
else:
    subjectId=get_max_ID(path+'subjects.db')+1
    insert_subject_details(path+'subjects.db',subjectId,name)
    
    


while True:
    #no_of_photo_of_subject=0
    rval,cap=vc.read()
    if(cap is not None):
        grey=cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
        faces=face_detection.detectMultiScale(grey,1.3,5)
        for (x,y,w,h) in faces:
            no_of_photo_of_subject=no_of_photo_of_subject+1
            print(no_of_photo_of_subject)
            cv2.imwrite('/Users/amlanjyotipatnaik/Documents/leaning_photo/'+'img_'+str(subjectId)+'_'+str(no_of_photo_of_subject)+'.jpg',grey[y:y+h,x:x+w])
        if(no_of_photo_of_subject >= 10):
            vc.release()
            break
            
        #time.sleep(5)
                
              
            
            
 
            
        
        


# In[ ]:





# In[24]:


no_of_photo_of_subject


# In[25]:


# Make our model learn the subject photo


# In[37]:


from PIL import Image


# In[38]:


recognizer=cv2.face.LBPHFaceRecognizer_create()
#path='/Users/amlanjyotipatnaik/Documents/leaning_photo/'


# In[28]:


##Function to read image files and return the np array 


# In[39]:


def getSubjectnImage(path):
    import glob
    
    images=glob.glob(path+'/*.jpg')
    subject_names=[]
    faces=[]
    for image in images:

        subject_names.append(int(image.split('.')[-2].split('_')[-2]))  #This is to be used when id needs to be string
        #subject_names.append(1)
        img=(Image.open(image)).convert('L')
        
        imgNP=np.array(img,'uint8')
        print(imgNP.shape)
        faces.append(imgNP)
        
    return subject_names,faces


    


# In[40]:


subjects, faces=getSubjectnImage(path)
print(subjects)
#print(faces[0].shape)


# In[31]:


##Training the model


# In[41]:


recognizer.train(faces,np.array(subjects))
recognizer.save(path+'image_recognizer.yml') #saving the model


# In[33]:


##Predicting the face


# In[43]:


faceDetect=cv2.CascadeClassifier('/Users/amlanjyotipatnaik/haars/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read(path+'image_recognizer.yml')

faceID=0
txt='unknown'
cap=cv2.VideoCapture(0)
rval,frame=cap.read()

while True:
    if frame is not None:
        #plt.imshow(frame)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #plt.imshow(gray)
        rval,frame=cap.read()
        #plt.imshow(frame)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray)
        #print(faces)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            faceID,conf=recognizer.predict(gray[y:y+h,x:x+w])
            #print(faceID)
            txt=get_subject_Name(path+'subjects.db',faceID)
        
            cv2.putText(frame,txt,(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,0,0), 2)
            cv2.imshow("detect",frame)
            
        if cv2.waitKey(1) & 0xFF==ord('q'):
            cap.release()    
            cv2.destroyAllWindows()
            break


# In[ ]:




