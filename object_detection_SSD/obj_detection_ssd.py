'''from computer vision A-Z course
   coded by trishit nath thakur'''


'''Object Detection using SSD'''

# Importing the libraries
import torch
import cv2
from ssd import build_ssd
import imageio
from data import BaseTransform, VOC_CLASSES as labelmap



'''Defining the function to perform Detections'''

def detect(frame, net, transform):  # We define a detect function that will take as inputs, a frame, a ssd neural network, and a transformation to be applied on the images, and that will return the frame with the detector rectangle.
    
                                                                                                                     # -> net - SSD network, transform - transforming the frame to make it compatible with the network
    height, width = frame.shape[0], frame.shape[1]                                                                          
    frame_t = transform(frame)[0]                                                                                    # -> Transforming the original frame to match the dimensions & requirements of the NN    
    x = torch.from_numpy(frame_t).permute(2, 0, 1)                                                                   # -> Convert Numpy Array to Torch Tensor. Permute used to change color channels from RBG(0,1,2) to GRB(2,0,1) as network was trained on that sequence.
    x = x.unsqueeze(0)                                                                                               # -> Expand the Dimensions to include the batch size
    y = net(x)                                                                                                       # -> Apply Network to model                                                                                            
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    
    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    
    for i in range(detections.size(1)):                                                                             # -> detections.size(1) = int(num_of_classes)
        j = 0
        while detections[0, i, j, 0] >= 0.6:                                                                        # -> confidence_score > 0.6
            pt = (detections[0, i, j, 1:] * scale).numpy()                                                          # -> Coordinates the detected object. Rectangles can drawn only on Numpy arrays
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)                # -> Draw the Rectangle
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)  # -> Putting the Label 
            j += 1
    return frame





# Creating the SSD neural network
net = build_ssd('test') # We create an object that is our neural network ssd.
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).




# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.




# Doing some Object Detection on a video
reader = imageio.get_reader('funny_dog.mp4') # We open the video.
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
writer = imageio.get_writer('output.mp4', fps = fps) # We create an output video with this same fps frequence.
for i, frame in enumerate(reader): # We iterate on the frames of the output video:
    frame = detect(frame, net.eval(), transform) # We call our detect function (defined above) to detect the object on the frame.
    writer.append_data(frame) # We add the next frame in the output video.
    print(i) # We print the number of the processed frame.
writer.close() # We close the process that handles the creation of the output video.