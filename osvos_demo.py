from __future__ import print_function
"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.

"""


from operator import itemgetter
import collections



import os
import sys
from PIL import Image,ImageDraw
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




def rename(path,type):
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
        os.rename(os.path.join(path, file), os.path.join(path, prefix + str(i)+'.' + type))

        i = i + 1



#x = 273
#y = 180


def getMiddle(left,right,bot,top):


    x = left + (right - left)/2
    y = top + (bot - top)/2


    #if (image.getpixel((x,y))!= (255,255,255)):



        #return getMiddle(left-1,right-1,bot,top,image)

    return(int(x),int(y))


def correction(x, y,image,index ):
    xi = x
    yi = y
    if (index%8 == 0):
        xi += 1 + index//8


    elif (index % 8 == 1):
        xi += -1 - index//8


    #doesnt matter(7)
    elif (index % 8 == 7):
        yi +=1 + index//8

    elif (index % 8 == 2):
        yi += 1 - index//8

    elif (index % 8 == 3):
        xi += 1 + index//8
        yi += 1 + index//8

    elif (index % 8 == 4):
        xi += -1 - index//8
        yi += -1 - index//8

    elif (index % 8 == 5):
        xi += 1 + index//8
        yi += -1 - index//8

    elif (index % 8 == 6):
        xi += -1 - index//8
        yi += 1 + index//8
    width, height = image.size
    if (xi >= 0 and yi >= 0 and xi < width and yi < height and image.getpixel((xi,yi)) == (255,255,255)):
        return (int(xi),int(yi))

    if (index == 100):
        return (0,0)
    return correction(x,y,image,index+1)


def convertToGraph(image):
    dict = {}
    width, height = image.size
    rgb_im = image.convert('RGB')
    for x in range(width ):
        for y in range(height ):
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
    text_file.write(str(start) + "\t" + str(finish) + "\t" + "RECTANGLE" + "\t" + "1" + "\t" + str(left) + "\t" + str(top) +"\t" +  str(right) + "\t" + str(bot) + "\t" + str(label) + "\n")
    text_file.close()

def returnString(left,right,top,bot,start,finish,label):
    return (str(start) + "\t" + str(finish) + "\t" + "RECTANGLE" + "\t" + "1" + "\t" + str(left) + "\t" + str(top) +"\t" +  str(right) + "\t" + str(bot) + "\t" + str(label) + "\n")


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



def getFirstMiddle(image):
    width, height = image.size

    for x in range(width - 1):
        for y in range(height - 1):
            if (image.getpixel((x,y)) == (255,255,255)):
                return (x,y)
    return (-1,-1)

# User defined parameters
seq_name = "fsGelbeAutos"
gpu_id = 0
train_model = True
result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)
#middleX = 54
#middleY = 120
rightShift = 0 #- 5
downShift = 0 #- 157


rename(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name),"jpg")
rename(os.path.join('DAVIS', 'Annotations', '480p', seq_name),"png")

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





png_im = Image.open(os.path.join('DAVIS', 'Annotations', '480p', seq_name,'00000.png'))

(middleX,middleY) = getFirstMiddle(png_im)


print(middleX,middleY)
#initialize the results so that getMiddle does the right thing
top = middleY
bottom = middleY
left = middleX
right = middleX

whites = set()
strings = []
counter =1
amount = 0
array = []
for image in result_imgs:
    amount += 1

for image in result_imgs:
    im = Image.open(image)
    rgb_im = im.convert('RGB')
    dic = convertToGraph(rgb_im)
    (middleX,middleY) = getMiddle(left,right,bottom,top)
    if (rgb_im.getpixel((middleX,middleY))!= (255,255,255)):
        (middleX,middleY) = correction(middleX,middleY,rgb_im,0)


    #print(str((left, right, bottom, top)) +"  "+ str(counter))

    bfs(dic,(middleX,middleY),whites)
    (left, right, bottom, top) = getMaxMin(rightShift,downShift,whites)
    whites = set()

    #writeLineToTxt(os.path.join('DAVIS', 'Annotations', '480p', seq_name,'text.txt'),left,right,top,bottom,1,1,'video')
    strings.append(returnString(left,right,top,bottom,(-1)*(amount-counter)*100,(-1)*(amount - counter + 1)*100,'video'))
    array.append([left,top,right,bottom])

    counter = counter + 1
    left -= rightShift
    right -= rightShift
    top -= downShift
    bottom -= downShift

text_file = open(os.path.join('DAVIS', 'Results', 'Segmentations','480p','OSVOS', seq_name,'AreaOfInterests.txt'), "w")
#print("ok")
text_file.writelines(strings)
text_file.close()


name = 0
counter = 0
for r_image in test_imgs:
    r_image_opened = Image.open(r_image)
    copy = r_image_opened.copy()
    draw = ImageDraw.Draw(copy)
    draw.rectangle(array[counter])

    #Image.ImageDraw.Draw.rectangle(array[counter])
    counter+=1
    copy.save('C:/Users/Dennis/Documents/uni/Hiwi/VideoSeg/OSVOS-TensorFlow/DAVIS/ResultsWithRectangles/'+ seq_name + "/"+ str(name) + '.jpg')
    name+=1