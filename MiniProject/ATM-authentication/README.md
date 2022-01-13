We aim to make ATMs more secure by two factor authentication using face recognition along with the existing
PIN system for ATMs.
Each card/bank account supports a maximum of four faces.

We will be using MTCNN algorithm to detect faces and a Siamese Network to recognize the face of the card user.
The MTCNN gives the Region of Interest that is, Face of the card user, thus eliminating any sort of background.
The Siamese Network is trained to recognize the 4 faces associated with the given card_no.
This produces a unique weights file that is capable of determining if the face of the user matches
with any of the 4 faces the network was trained on.
