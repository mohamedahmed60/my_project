import cv2 # استدعاء المكتبة
#===== قيم المعنى النموذجي =====
MODEL_MEAN_VALUES = (78.4463377603,
                     87.7689143744,
                     114.895847746)
#===== انشاء ليستة للاعمار ======
age_list =['(0, 2)','(4, 6)','(8, 12)',
           '(15,20)','(25,32)','(38, 43)','(48, 53)',
           '(60, 100)'
           ]
#===== ليستة تحديد الجنس ========
gender_list =['Male','Female']
#===== استدعاء ملفات التي تتعرف على العمر والجنس =====
def filesGet():
    age_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_age.prototxt',
        'data/age_net.caffemodel'
    )
    gender_net= cv2.dnn.readNetFromCaffe(
        'data/deploy_gender.prototxt',
        'data/gender_net.caffemodel'
    )
    return(age_net, gender_net)

def read_from_camera(age_net,gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX # نوع الخط
    image= cv2.imread('images/girl2.jpg') # استدعاء الصورة
    #===== الملف الخاص بتحديد الوجه ======
    face_cascade =cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    #===== تحديد نظام الاولوان ====
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    #===== كشف وجوه متعددة في الصورة الواحدة =====
    faces = face_cascade.detectMultiScale(gray, 1.1,5)
    if(len(faces)>0): #تحديد عدد الوجوه
        print("Found {} Faces".format(str(len(faces))))
    
    for(x, y, w, h)in faces:
        #رسم مستطيل
        cv2.rectangle(image, (x,y),(x+w, y+h),(255,255,0),2)
        #جلب وجه ونسخه ارسالها الى الخوازمية
        face_img= image[y:y+h, h:h+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227,227),MODEL_MEAN_VALUES, swapRB=False)
        #=== توقع الجنس =====
        gender_net.setInput(blob)
        gender_p =gender_net.forward() # output
        gender = gender_list[gender_p[0].argmax()]
        print("Gender : " + gender)
        #=== توقع العمر =====
        age_net.setInput(blob)
        age_p =age_net.forward() # output
        age = age_list[age_p[0].argmax()]
        print("Age : " + age)
        G_A = "%s %s" % (gender , age)
        cv2.putText(image, G_A, (x,y) , font , 1 , (255,255,255) , 2 , cv2.LINE_AA)
        cv2.imshow('My_project', image)
    cv2.waitKey(0)
if __name__ == "__main__":
    age_net, gender_net = filesGet()
    read_from_camera(age_net,gender_net)
