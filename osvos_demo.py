from __future__ import print_function
"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.

@Inproceedings{Cae+17,
  Title          = {One-Shot Video Object Segmentation},
  Author         = {S. Caelles and K.K. Maninis and J. Pont-Tuset and L. Leal-Taix\'e and D. Cremers and L. {Van Gool}},
  Booktitle      = {Computer Vision and Pattern Recognition (CVPR)},
  Year           = {2017}
}
"""


from operator import itemgetter
import collections



import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib.pyplot as plt
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos
from dataset import Dataset
os.chdir(root_folder)



def rename(path):
    files = os.listdir(path)
    i = 0
    prefix = "0000"


    for file in files:
        prefix = "0000"
        if (i >= 10 and i < 100) :
            prefix = "000"

        elif (i >= 100 and i < 1000):
            prefix = "00"

        elif( i >= 1000):
            prefix = "0"
        os.rename(os.path.join(path, file), os.path.join(path, prefix + str(i)+'.jpg'))

        i = i + 1



#x = 273
#y = 180


def getMiddle(left,right,bot,top):
    x = left + (right - left)
    y = top + (bot - top)

    return(x,y)



def convertToGraph(image):
    dict = {}
    width, height = image.size
    rgb_im = image.convert('RGB')
    for x in range(width - 1):
        for y in range(height - 1):
            dict[(x,y)] = []
            if (rgb_im.getpixel((x,y)) == (255,255,255)):

                if ( x!= width -1 and rgb_im.getpixel((x + 1, y)) == (255, 255, 255)):
                    dict[(x,y)].append((x+1,y))

                if (x!= width- 1 and y != height - 1 and rgb_im.getpixel((x + 1, y + 1)) == (255, 255, 255)):
                    dict[(x,y)].append((x+1,y+1))

                if (x!= width - 1 and y != 0 and rgb_im.getpixel((x + 1, y - 1)) == (255, 255, 255)):
                    dict[(x,y)].append((x+1,y-1))

                if (y != height- 1 and rgb_im.getpixel((x , y + 1)) == (255, 255, 255)):
                    dict[(x,y)].append((x,y+1))

                if (y != 0 and rgb_im.getpixel((x , y - 1)) == (255, 255, 255)):
                    dict[(x,y)].append((x,y-1))

                if (y != height - 1 and x != 0 and rgb_im.getpixel((x - 1, y + 1)) == (255, 255, 255)):
                    dict[(x,y)].append((x-1,y+1))

                if (x != 0 and rgb_im.getpixel((x - 1, y)) == (255, 255, 255)):
                    dict[(x,y)].append((x-1,y))

                if (y != 0 and x != 0 and rgb_im.getpixel((x - 1, y - 1)) == (255, 255, 255)):
                    dict[(x,y)].append((x-1,y-1))

    return dict


def writeLineToTxt(OutputPath,left,right,top,bot,start,finish,label):
    #start_offset end_offset RECTANGLE <id> <left> <top> <right> <bottom> [label]
    text_file = open(OutputPath, "w")
    text_file.write(str(start) + " " + str(finish) + " " + "RECTANGLE" + " " + " " + "1" + " " + str(left) + " " + str(top) +" " +  str(right) + " " + str(bot) + " " + "[" + " " + str(label) + " " + "]" + "\n")
    text_file.close()

def returnString(left,right,top,bot,start,finish,label):
    return str(start) + " " + str(finish) + " " + "RECTANGLE" + " " + " " + "1" + " " + str(left) + " " + str(top) +" " +  str(right) + " " + str(bot) + " " + "[" + " " + str(label) + " " + "]" + "\n"


def bfs(graph, root,visited):
    queue =  collections.deque([root])
    visited.add(root)
    while queue:
        vertex = queue.popleft()
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)



#visited = set()
#bfs(convertToGraph(im), (273,180),visited)

def getMaxMin(leftShift,topShift,visited) :


    right = max(visited,key=itemgetter(0))[0] + leftShift

    bot = max(visited,key=itemgetter(1))[1]  + topShift

    left = min(visited,key=itemgetter(0))[0] + leftShift

    top = min(visited,key=itemgetter(1))[1] + topShift

    return (left,right,bot,top)






# User defined parameters
seq_name = "2_01ReverseCut"
gpu_id = 0
train_model = False
result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)
middleX = 277
middleY = 130
rightShift = 473
downShift = 572


rename(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name))

# Train parameters
parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
logs_path = os.path.join('models', seq_name)
max_training_iters = 200

# Define Dataset
test_frames = sorted(os.listdir(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)))
test_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in test_frames]
if train_model:
    train_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, '00000.jpg')+' '+
                  os.path.join('DAVIS', 'Annotations', '480p', seq_name, '00000.png')]
    dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
else:
    dataset = Dataset(None, test_imgs, './')

# Train the network
if train_model:
    # More training parameters
    learning_rate = 1e-8
    save_step = max_training_iters
    side_supervision = 3
    display_step = 10
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                 save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        checkpoint_path = os.path.join('models', seq_name, seq_name+'.ckpt-'+str(max_training_iters))
        osvos.test(dataset, checkpoint_path, result_path)

# Show results
overlay_color = [255, 0, 0]
transparency = 0.6
plt.ion()
for img_p in test_frames:
    frame_num = img_p.split('.')[0]
    img = np.array(Image.open(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, img_p)))
    mask = np.array(Image.open(os.path.join(result_path, frame_num+'.png')))
    mask = mask//np.max(mask)
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
    im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1]*transparency + (1-transparency)*img[:, :, 1])
    im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])
    plt.imshow(im_over.astype(np.uint8))
    plt.axis('off')
    plt.show()
    plt.pause(0.1)
    plt.clf()



#get the annotated pictures
result_frames = sorted(os.listdir(os.path.join('DAVIS', 'Results', 'Segmentations','480p','OSVOS', seq_name)))
result_imgs = [os.path.join('DAVIS', 'Results','Segmentations', '480p','OSVOS', seq_name, frame) for frame in result_frames]

#initialize the results so that getMiddle does the right thing
top = 0
bottom = middleY
left = 0
right = middleX

whites = set()
strings = []
for image in result_imgs:
    im = Image.open(image)
    rgb_im = im.convert('RGB')
    dic = convertToGraph(rgb_im)
    (middleX,middleY) = getMiddle(left,right,bottom,top)
    print(rgb_im.getpixel((middleX,middleY)))
    print((middleX,middleY))
    print((left, right, bottom, top))
    bfs(dic,(middleX,middleY),whites)
    (left, right, bottom, top) = getMaxMin(rightShift,downShift,whites)
    #writeLineToTxt(os.path.join('DAVIS', 'Annotations', '480p', seq_name,'text.txt'),left,right,top,bottom,1,1,'video')
    strings.append(returnString(left,right,top,bottom,1,1,'video'))
    left -= rightShift
    right -= rightShift
    top -= downShift
    bottom -= downShift

text_file = open(os.path.join('DAVIS', 'Annotations', '480p', seq_name,'text.txt'), "w")
text_file.writelines(strings)
text_file.close()