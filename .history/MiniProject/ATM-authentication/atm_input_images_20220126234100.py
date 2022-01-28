# Script to take training data of users and save it somewhere.
import cv2
import time
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN,InceptionResnetV1
import pickle

# Parameters
image_size=600
frame_rate=64
vid_len=20  # Length of video in seconds
card_no=input('Enter the card number: ').strip()
person_name=input('Enter the person\'s name: ').strip()
device=torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu")

# Save all face images of a person as a pickle file
def save_face_images(frames,boxes):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    for f in range(len(frames)):
        img=np.asarray(frames[f])
        box=boxes[f]
        if len(box.shape)==3:
            #Go into loop only when there is atleast 1 face in image
            # Loop for num of boxes in each image
            for b in range(box.shape[1]):
                start=(np.clip(int(box[0][b][0])-15,0,480),np.clip(int(box[0][b][1])-50,0,640))
                end=(np.clip(int(box[0][b][2])+15,0,480),np.clip(int(box[0][b][3])+20,0,640))
                crop_pic=img[start[1]:end[1],start[0]:end[0]]
            img_crop=Image.fromarray(crop_pic)
            img_crop=transform(img_crop)
            img_crop = torch.unsqueeze(img_crop, 0)
            save_tensor=model(img_crop)
            return save_tensor

v_cap =cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.destroyAllWindows()
v_cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size)
v_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size)
count=1
prev=0
try:
    os.mkdir(card_no)
except FileExistsError:
    pass

mtcnn=MTCNN(image_size=image_size,keep_all=True,device=device,post_process=True)
model = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
start=time.time()
frames=[]
boxes=[]
print('Try to keep your face at the centre of the screen and turn ur face slowly in order to capture diff angles of your face')
time.sleep(3)
print('A window will pop up in abt 3 seconds')
time.sleep(3)
save_tensor=None

# 20 sec loop to input truth face images
while True:
    time_elapsed=time.time()-prev
    curr=time.time()
    if curr-start>=vid_len:
        break
    ret, frame = v_cap.read()
    cv2.imshow('Recording and saving Images',frame)
    if time_elapsed>1./frame_rate: # Collect frames every 1/frame_rate of a second
        prev = time.time()
        frame_ = Image.fromarray(frame)
        frames.append(frame_)
        batch_boxes, prob, landmark = mtcnn.detect(frames, landmarks=True)
        frames_duplicate = frames.copy()
        boxes.append(batch_boxes)
        boxes_duplicate = boxes.copy()
        # show imgs with bbxs
        if save_tensor==None:
            save_tensor=save_face_images(frames_duplicate, boxes_duplicate)
        else:
            temp=save_face_images(frames_duplicate, boxes_duplicate)
            if temp is not None:
                save_tensor=torch.cat([temp,save_tensor],dim=0)
                print(save_tensor.shape)
        count+=1
        frames = []
        boxes = []
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
        
# Open file for pickling
face_file=open(card_no+'/'+person_name,'ab')
pickle.dump(save_tensor,face_file)
face_file.close()
v_cap.release()
cv2.destroyAllWindows()

