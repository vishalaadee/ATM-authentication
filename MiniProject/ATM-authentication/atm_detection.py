# Imports
import os
import torch
from torchvision import transforms
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import time
import pickle
from PIL import Image

# Transform on the imgs
transform=transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor()
])

# Parameters
card_number=input('Enter the card_number : ').strip()
frame_rate=16
prev=0
batch_size=32
image_size=600
threshold=0.85
device=device=torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu")
bbx_color=(0,255,255)

current_person=None

def detect_imgs(img):
    global current_person
    person_=None
    img=transform(img)
    img=torch.unsqueeze(img,0)
    img=model(img)
    minimum=torch.tensor(99)
    for face_,name in zip(faces,face_names):
        temp=torch.min(torch.norm((face_-img),dim=1))
        if temp<minimum and temp<threshold:
            minimum=temp
            person_=name
            current_person=name
    return person_,minimum.item()


def show_images(frames,boxes,color):
    temp=None
    for f in range(len(frames)):
        img=np.asarray(frames[f])
        box=boxes[f]
        if len(box.shape)==3:
            #Go into loop only when there is atleast 1 face in image
            # Loop for num of boxes in each image
            for b in range(box.shape[1]):
                start=(np.clip(int(box[0][b][0])-15,0,600),np.clip(int(box[0][b][1])-20,0,600))
                end=(np.clip(int(box[0][b][2])+15,0,600),np.clip(int(box[0][b][3])+15,0,600))
                img=cv2.rectangle(img,start,end,color,2)
                crop_pic = img[start[1]:end[1], start[0]:end[0]]
                crop_pic=Image.fromarray(crop_pic)
                person,diff=detect_imgs(crop_pic)
                if person is not None:
                    cv2.putText(img, person+': '+'{:.2f}'.format(diff), (start[0], start[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    temp=1
                else:
                    cv2.putText(img, 'Unknown'+': '+'{0}'.format(diff), (start[0], start[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    temp=0
        cv2.imshow('Detection',img)
        if temp==1:
            return 1
        else:
            return 0

# Init MTCNN object
mtcnn=MTCNN(image_size=image_size,keep_all=True,device=device,post_process=True)
model = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
# Real time data from webcam
frames=[]
boxes=[]

# Load stored face data related to respective card number
faces=[]
face_names=[]
face_file=None
try:
    for person in os.listdir(card_number):
        face_file=open(card_number+'/'+person,'rb')
        if face_file is not None:
            face=pickle.load(face_file)
            faces.append(face)
            face_names.append(str(person))
except FileNotFoundError:
    print('Face data doesnt exist for this card.')
    exit()

# Infinite Face Detection Loop
v_cap = cv2.VideoCapture(0)
v_cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size)
v_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size)
flag=False
face_results=[]
start=time.time()
while (True):
    time_elapsed=time.time()-prev
    break_time = time.time() - start
    if break_time > 10:
        break
    ret,frame=v_cap.read()
    if time_elapsed>1./frame_rate: # Collect frames every 1/frame_rate of a second
        prev=time.time()
        frame_=Image.fromarray(frame)
        frames.append(frame_)
        batch_boxes,prob,landmark=mtcnn.detect(frames, landmarks=True)
        frames_duplicate=frames.copy()
        boxes.append(batch_boxes)
        boxes_duplicate=boxes.copy()
        # show imgs with bbxs
        face_results.append(show_images(frames_duplicate,boxes_duplicate,bbx_color))
        frames=[]
        boxes=[]
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
v_cap.release()
cv2.destroyAllWindows()
accuracy=(sum(face_results)/len(face_results))*100
print('Percentage match '+'{:.2f}'.format(accuracy))
if accuracy>0.75:
    print('Authorization Successful')
else:
    print('Authorization Unsuccessful')
    exit()



