## Model in the loop as an effort-efficient learning technique

We want to identify individual lemurs using a simple ResNet18. The problem is that the lemurs change over time, new lemurs join groups and others leave. As there are 4-5 groups in total, and data from 4 field seasons, this will require many different models.
One option is to select individual images for each lemur manually, but this is very time consuming.

### Sorting model

One issue with tracking lemurs in the wild is that the animals are often with the back to the camera, partly occluded or far away and thus not easily identifiable. However, we only need them to be identifiable on a few frames, as we can propagate the label through the whole track, as described in [PriMAT](https://github.com/ecker-lab/PriMAT-tracking).
If we want to train a model for identification, it is helpful to select frames in which the individual is identifiable, otherwise the model can be distracted by too much noise (in the sense of images that could be any individual).
To make the selection of images with which to train the sorting model, use label_tool_from_scratch.py. 


Samples that are not useful (To visualize samples from each class, use dataset_review.py.):
![](assets/sample2_class_0.png)
Samples that are useful:
![](assets/sample2_class_1.png)

The sorting model helps to select frames that we consider useful. It is a simple ResNet18 that has been trained on a binary classification task of usefulness for identification. To train the sorting model, use train_sorting_model.py. 

![](assets/quality_examples/1%20=%20<20%25.png)
![](assets/quality_examples/2%20=%20<40%25.png)
![](assets/quality_examples/3%20=%20<60%25.png)
![](assets/quality_examples/4%20=%20<80%25.png)
![](assets/quality_examples/5%20=%20<99%25.png)
![](assets/quality_examples/6%20=%20Top%20Scores.png)



### Identification model

We have many possible occurences of each individual. As the classes are heavily imbalanced, e.g.
Unsure (7)    217096
Ata (3)       140153
Zemlya (4)    114344
Gerald (6)     98539
Yaya (1)       57064
Novaya (2)     43452
Tiwi (0)       12243
Croker (5)      6361

We think that 1000 occurences of one individual should be enough to learn to identify it, however a large portion of the images is actually not easily identifiable.
Therefore we first cluster the bbox location of each occurence of each individual with a very high number of clusters (e.g. 2700). Then we apply the sorting model to one representative image of each cluster, and take the top 1000 scores.

This is what a selection of this process looks like:
![](assets/R1_old/3.png)
