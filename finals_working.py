
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import time
angle =0

#function to calculate angle
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    #print("a = ",a,"b = ",b,"b = ",c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle
#declaring some variables
status = " "
#curl counter
counter=0
stage =  None
## Setup mediapipe instance
a=0

cap = cv2.VideoCapture('KneeBendVideo.mp4')

with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:

    start_time=0
    end_time=0
    flag = False
    time_spent = 0

    while cap.isOpened():

        ret, frame = cap.read()
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            #if k <0 -> person has his right side of his body towards the camera;
            #if k >0 -> person has his left side of his body towards the camera;

            k = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x-landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
            #print(k)
            if k >0:
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            elif k<0:
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            #print("hip",hip,"knee",knee,"ankle",ankle)

            angle= calculate_angle(hip, knee, ankle)


            # Curl counter logic
            if angle <= 140 and start_time==0: #leg curled
                stage = "start"
                start_time=int(time.time())
    
            elif angle <= 140:  #BONUS
                time_spent = int(time.time())-start_time 
            #user performance stat
                if(angle >100 and angle<140):
                    print("good",end = "\r")
                    status = "status = good"
                else:     #(angle>20 and angle<80):
                   
                    print("excellent",end = "\r")
                    status = "status = excellent"
            


            elif angle >140 and stage == "start": #leg straightened
                end_time=int(time.time())
                status = "status = leg straight"
                if end_time-start_time <8: #unsuccessful rep
                    stage ='Keep your knee bent'
                   
                else : #successful rep
                    stage ="done"
                    counter+=1

                start_time=0
                end_time=0
        except:
            pass

        cv2.rectangle(image, (0,0), (510,150), (245,157,56), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        # # #print time
        cv2.putText(image, 'Time Spent', (10,80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

        cv2.putText(image, str(time_spent), 
                    (195,80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, 'seconds', (240,80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

        # # #print status
        cv2.putText(image,status,(10,120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        # Stage data
        cv2.putText(image, 'STAGE', (180,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (180,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        #visualize angle
        cv2.putText(image, "ANGLE = ", (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                       
        cv2.putText(image, str(round(angle, 2)), (65,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                        

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )
        cv2.imshow('Video Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
