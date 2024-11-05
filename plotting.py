import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from torchvision.transforms import v2
import torch
import os
import seaborn as sns

plt.style.use('ggplot')

# --------------------------------------------------------

num = 14

fig, axs = plt.subplots(4, int(num/2))

files = os.listdir("frames/")
files = random.sample(files, num)

for x in range(num):
    im = cv2.imread("frames/"+files[x])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (224,224))
    axs[int(x/int(num/2))][x%int(num/2)].imshow(im)
    axs[int(x/int(num/2))][x%int(num/2)].axis("off")
    axs[int(x/int(num/2))][x%int(num/2)].set_title("r"+str(x+1))


files = os.listdir("frames_sorting_model_subset_99/")
files = random.sample(files, num)

for x in range(num):
    im = cv2.imread("frames_sorting_model_subset_99/"+files[x])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (224,224))
    axs[int(x/int(num/2))+2][x%int(num/2)].imshow(im)
    axs[int(x/int(num/2))+2][x%int(num/2)].axis("off")
    axs[int(x/int(num/2))+2][x%int(num/2)].set_title("s"+str(x+1))


fig.set_size_inches(14, 8)
fig.tight_layout()
plt.savefig('vis/joined_collage.png', dpi=1000, pad_inches=0)
plt.clf()



# --------------------------------------------------------

fig, axs = plt.subplots(1, 1)

confusion_matrix = np.array([[ 2,  0.,   0.,   2,   0.,   0., 0 ], [ 0.,   16, 0.,   0.,   0., 0., 0  ], [ 0., 0.,   17, 5, 0.,   0., 0  ], [ 0.,   0.,   0.,   19, 0.,   0., 0  ], [ 1,   0., 0.,   1,   6, 0., 1], [ 0.,   0.,   0., 0., 0.,   22, 0  ], [ 0, 0, 0, 0, 0, 0, 0 ]])

axs = sns.heatmap(confusion_matrix, annot=True)
axs.set_xlabel("Predicitons", fontsize=13)
axs.set_ylabel('Label', fontsize=13)
axs.set_xticklabels(["1", "2", "3", "4", "5", "6", "unsure"])
axs.set_yticklabels(["1", "2", "3", "4", "5", "6", "unsure"])

plt.savefig('vis/conf_best.png', dpi=1000, pad_inches=0)
plt.clf()

# --------------------------------------------------------

files = os.listdir("duplicates/")
fig, axs = plt.subplots(5, 8)

for x in range(5):
    for y in range(8):
        im = cv2.imread("duplicates/"+files[x*5+y])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (224,224))
        axs[x][y].imshow(im)
        axs[x][y].axis("off")

fig.set_size_inches(8,5, forward=True)
plt.subplots_adjust(hspace=0.1, wspace=0.1)
plt.savefig('vis/duplicates.png', dpi=1000, pad_inches=0)
plt.clf()
# --------------------------------------------------------

fig, axs = plt.subplots(1, 1)

confusion_matrix = np.array([[ 0.9,  0.,   0.,   0.,   0.,   0.01, 0 ], [ 0.,   0.96, 0.,   0.,   0.02, 0.01, 0  ], [ 0.02, 0.,   0.84, 0.01, 0.,   0.01, 0  ], [ 0.,   0.,   0.,   0.64, 0.,   0.01, 0  ], [ 0.,   0.02, 0.,   0.,   0.95, 0., 0], [ 0.,   0.,   0.01, 0.01, 0.,   0.69, 0  ], [ 0.08, 0.02, 0.14, 0.33, 0.03, 0.27, 0 ]])

axs = sns.heatmap(confusion_matrix, annot=True)
axs.set_xlabel("Model Labels", fontsize=13)
axs.set_ylabel('Corrections', fontsize=13)
axs.set_xticklabels(["1", "2", "3", "4", "5", "6", "unsure"])
axs.set_yticklabels(["1", "2", "3", "4", "5", "6", "unsure"])

plt.savefig('vis/conf_s_99_2k.png', dpi=1000, pad_inches=0)
plt.clf()

# --------------------------------------------------------

num = 14

fig, axs = plt.subplots(2, int(num/2))

files = os.listdir("frames_sorting_model_subset_90/")
files = random.sample(files, num)

for x in range(num):
    im = cv2.imread("frames_sorting_model_subset_90/"+files[x])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (224,224))
    axs[int(x/int(num/2))][x%int(num/2)].imshow(im)
    axs[int(x/int(num/2))][x%int(num/2)].axis("off")
    axs[int(x/int(num/2))][x%int(num/2)].set_title(x)

fig.set_size_inches(12, 6)
fig.tight_layout()
plt.savefig('vis/collage_90.png', dpi=1000, pad_inches=0)
plt.clf()

# --------------------------------------------------------


fig, axs = plt.subplots(1, 1)
n = 4000
x = np.arange(0,n)
y1 = np.arange(0,n)
y2 = np.arange(0,n)
y3 = np.arange(170,n)

y1 = (y1/0.17)*1.7
y2 = 600 + 1200 + (y2/0.505)*1.7

y1 /= 60
y2 /= 60

axs.plot(x, y2, color="b")
axs.plot(x, y1, color="r")

fx = x[:271]
fy1 = y1[:271]
fy2 = y2[:271]
axs.fill(np.append(fx, fx[::-1]), np.append(fy1, fy2[::-1]), 'pink')

sx = x[-n+271:]
sy1 = y1[-n+271:]
sy2 = y2[-n+271:]
axs.fill(np.append(sx, sx[::-1]), np.append(sy1, sy2[::-1]), 'lightcyan')
axs.set_xlabel("Identifiable Images", fontsize=13)
axs.set_ylabel('Time (min)', fontsize=13)
axs.legend(['With Sorting (Eq: 1)','Raw (Eq: 3)'], loc="upper left")
axs.text(60, 70, '(271,45)', fontsize = 15)
axs.set_xlim(0,3000)

fig.set_size_inches(18, 6)
plt.subplots_adjust(left=0.05, right=0.95)
plt.savefig('vis/scaling_comparison.png', dpi=1000, pad_inches=0)
plt.clf()

# --------------------------------------------------------
from importlib import reload
plt=reload(plt)
fig, axs = plt.subplots(1, 6)

for x in range(1,7):
    im = cv2.imread("icons/"+str(x)+".png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    axs[x-1].imshow(im)
    axs[x-1].axis("off")
    axs[x-1].set_title(x)

fig.tight_layout()
fig.set_size_inches(24, 6)
plt.savefig('vis/collars.png', dpi=1000, pad_inches=0)
plt.clf()

# -------------------------------------------------------------
from importlib import reload
plt=reload(plt)
fig, axs = plt.subplots(1, 6)

for x in range(1,7):
    im = cv2.imread("other_group/"+str(x)+".png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    axs[x-1].imshow(im)
    axs[x-1].axis("off")

fig.tight_layout()
fig.set_size_inches(24, 6)
plt.savefig('vis/other_collars.png', dpi=1000, pad_inches=0)
plt.clf()
# -------------------------------------------------------------


with open("distance_sort_log", 'r') as file:
    lines = file.readlines()

n1 = [ float(x) for i,x in enumerate(lines) if i % 2 == 0]
n2 = [ float(x) for i,x in enumerate(lines) if i % 2 == 1]

fig, axs = plt.subplots(1, 1)
axs2 = axs.twinx()

axs.grid(True)
axs2.grid(False)

axs2.plot(range(len(n2)), n2, color="b")
axs.plot(range(len(n1)), n1, color="r")

fig.set_size_inches(15, 4, forward=True)
axs.set_ylabel('Average \n Distance', color="r", fontsize=15)
axs.set_xlabel("Iteration", fontsize=15)
axs2.set_ylabel('Number of \n frames', color="b", fontsize=15)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2)
plt.savefig('vis/reduction_duplicate.png', dpi=1000, pad_inches=0)
plt.clf()

# -------------------------------------------------------------
raw_scaling = [ 0.717391304347826, 0.6847826086956522, 0.5543478260869565,  0.782608695652174, 0.8369565217391305, 0.75, 0.8369565217391305, 0.8152173913043478, 0.7717391304347826, 0.8043478260869565, 0.8695652173913043, 0.8260869565217391, 0.8043478260869565, 0.7934782608695652, 0.8369565217391305, 0.8586956521739131, 0.7934782608695652, 0.8260869565217391, 0.8043478260869565, 0.8478260869565217, 0.8695652173913043, 0.8695652173913043, 0.8695652173913043, 0.8478260869565217, 0.8152173913043478, 0.8152173913043478, 0.8260869565217391 ]
x_raw_scaling = [250,250,250,500,500,500,750,750,750,1000,1000,1000,1250,1250,1250,1500,1500,1500,1750,1750,1750,2000,2000,2000,2500,2500, 2500]

mean = [ 0.652, 0.79, 0.808,0.833, 0.811, 0.826, 0.84, 0.862, 0.819  ]
x = [250,500,750,1000,1250,1500,1750,2000,2500]


coef_lin = np.polyfit(x_raw_scaling[3:],raw_scaling[3:],1)
coef_exp = np.polyfit(np.log(x_raw_scaling), raw_scaling, 1)

print("lin:", coef_lin)
print("exp:", coef_exp)

poly1d_fn = np.poly1d(coef_lin) 

ticks_x = np.arange(min(x), max(x), 0.1)
ticks_y = np.arange(min(x), max(x), 0.1)
ticks_y = coef_exp[0] * np.log(ticks_y) + coef_exp[1]


plt.plot(x, mean, 'yo', color="blue")
plt.plot(x, poly1d_fn(x), '--k')
plt.plot(ticks_x, ticks_y)

plt.legend(["Models", "Regression: mx+b", "Regression: a ln(x) + b"], fontsize=12, loc="lower right")
plt.xlabel("Number of Images", fontsize=15)
plt.ylabel("Test set accuracy", fontsize=15) 


figure = plt.gcf()
figure.set_size_inches(12, 6)
plt.subplots_adjust(left=0.12, right=0.88)
plt.savefig('vis/scaling_plot.png', dpi=1000, pad_inches=0)
plt.clf()

# -------------------------------------------------------------
# There are not 

plt.style.use('default')
labels_file = "../data/second_try/A_frames_hand_labeled/labels"
file_dir = "../data/second_try/A_frames_hand_labeled/images/"

with open(labels_file, 'r') as file:
    lines = file.readlines()

lines = [ x.replace("\n","").split(",") for x in lines]
lines = [ [str(x[0]), int(x[1]), int(x[2]), 1] for x in lines]
lines = [ x[0]+"_"+str(x[1])+".png" for x in lines if x[2] != 7 ]

lines = random.sample(lines, 5)

transform = v2.Compose([
    v2.RandomResizedCrop(size=(224,224), scale=(0.8,1)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=45),
])

fig, axs = plt.subplots(2, 5)

for i,x in enumerate(lines):
    img = cv2.imread(file_dir+x)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img2 = img.copy()
    img2 = torch.from_numpy(img2)
    img2 = img2.permute((2, 0, 1))

    img2 = transform(img2)

    img2 = img2.permute((1, 2, 0))
    img2 = img2.numpy()
    
    axs[0][i].imshow(img)
    axs[1][i].imshow(img2)

    axs[0][i].axis("off")
    axs[1][i].axis("off")


plt.tight_layout()
fig.set_size_inches(18.5, 10.7)
plt.savefig('vis/show_augmentations.png', dpi=1000, pad_inches=0)
plt.clf()

# -------------------------------------------------------------
# There are not 
plt.style.use('ggplot')
plt.clf()

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1] })

handsort = [ 81, 131, 167, 89, 152, 180 ] 
handsort = np.array(handsort)/sum(handsort)
values = [x+1 for x in range(len(handsort))]

s1 = [17.6]
s2 = [100]

axs[0].bar(values, handsort, color="g")
axs[0].set_xlabel("Classes", fontsize=15)
axs[0].set_ylabel("Count", fontsize=15) 

axs[1].barh([0], s2, align='center', color="r")
axs[1].barh([0], s1, align='center', color="g")
axs[1].legend(['Unsure','Ident'], bbox_to_anchor=(0, 0.94))
axs[1].set_yticks([])
axs[1].set_xticks([0,25,50,75,100])
axs[1].set_xlim(0, 100)
axs[1].set_xlabel("Percentage of identifiable Images", fontsize=15)
fig.set_size_inches(15, 5, forward=True)
fig.tight_layout()
plt.subplots_adjust(left=0.10, right=0.90)
plt.savefig('vis/handsort.png', dpi=1000, pad_inches=0)
plt.clf()

#-----------------------------------------------------------------
fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1] })

handsort = [ 63, 127, 193, 126, 150, 121 ]
handsort = np.array(handsort)/sum(handsort)
values = [x+1 for x in range(len(handsort))]

s1 = [39]
s2 = [100]

axs[0].bar(values, handsort, color="g")
axs[0].set_xlabel("Classes", fontsize=15)
axs[0].set_ylabel("Count", fontsize=15) 

axs[1].barh([0], s2, align='center', color="r")
axs[1].barh([0], s1, align='center', color="g")
axs[1].legend(['Unsure','Ident'], bbox_to_anchor=(0, 0.94))
axs[1].set_yticks([])
axs[1].set_xticks([0,25,50,75,100])
axs[1].set_xlim(0, 100)
axs[1].set_xlabel("Percentage of identifiable Images", fontsize=15)

fig.set_size_inches(15, 5, forward=True)
fig.tight_layout()
plt.subplots_adjust(left=0.10, right=0.90)
plt.savefig('vis/90_label_data.png', dpi=1000, pad_inches=0)
plt.clf()

#-------------------------------------------

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1] })

handsort =  [ 102, 161, 293, 144, 182, 128 ]
handsort = np.array(handsort)/sum(handsort)
values = [x+1 for x in range(len(handsort))]

s1 = [50.5]
s2 = [100]

axs[0].bar(values, handsort, color="g")
axs[0].set_xlabel("Classes", fontsize=15)
axs[0].set_ylabel("Count", fontsize=15) 

axs[1].barh([0], s2, align='center', color="r")
axs[1].barh([0], s1, align='center', color="g")
axs[1].legend(['Unsure','Ident'], bbox_to_anchor=(0, 0.94))
axs[1].set_yticks([])
axs[1].set_xticks([0,25,50,75,100])
axs[1].set_xlim(0, 100)
axs[1].set_xlabel("Percentage of identifiable Images", fontsize=15)

fig.set_size_inches(15, 5, forward=True)
fig.tight_layout()
plt.subplots_adjust(left=0.10, right=0.90)
plt.savefig('vis/99_label_data.png', dpi=1000, pad_inches=0)
plt.clf()


#-------------------------------------------

fig, axs = plt.subplots(1, 1)

handsort = [ 190, 571, 409, 163, 335, 332 ]
handsort = np.array(handsort)/sum(handsort)
values = [x+1 for x in range(len(handsort))]

axs.bar(values, handsort, color="g")
axs.set_xlabel("Classes", fontsize=15)
axs.set_ylabel("Count", fontsize=15) 

fig.set_size_inches(15, 5, forward=True)
plt.subplots_adjust(left=0.10, right=0.90, bottom=0.2)
plt.savefig('vis/with_sort.png', dpi=1000, pad_inches=0)

#-------------------------------------------

fig, axs = plt.subplots(1, 1)

handsort = [ 153,   455,    427,    297,    304,    364, 0 ]
handsort1 = [ 140,   448,    369,    194,    297,    257,  295  ]

handsort = np.array(handsort)/sum(handsort)
handsort1 = np.array(handsort1)/sum(handsort1)

values = np.array(range(len(handsort)))
print(values)

axs.bar(values-0.2, handsort, width = 0.4, color="g", label ='1')
axs.bar(values+0.2, handsort1, width = 0.4, color="b", label ='2')
axs.set_xlabel("Classes", fontsize=15)
axs.set_xticks(values)
axs.set_xticklabels(["1", "2", "3", "4", "5", "6", "unsure"])
axs.set_ylabel("Count", fontsize=15) 
axs.legend(["Pseudo Labels", "Revised"])

fig.set_size_inches(15, 5, forward=True)
plt.subplots_adjust(left=0.10, right=0.90, bottom=0.2)
fig.tight_layout()
plt.savefig('vis/correction_without_sort.png', dpi=1000, pad_inches=0)
