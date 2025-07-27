# 📘 Machine Learning and Deep Learning Daily

## 📌 Introduction

A few months ago, I completed the **Machine Learning Specialization** course. Now, I’m revisiting and revising those machine learning concepts, while also diving deeper into **deep learning**. I’ll be sharing my daily progress here as I go along.

---

## 📚 Resources & Progress

| Title                                                                                      | Progress       |
|--------------------------------------------------------------------------------------------|----------------|
| [Machine Learning – Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning) | ✅ Completed   |
| [Deep Learning Specialization – Andrew Ng (Coursera)](https://www.coursera.org/specializations/deep-learning) | ⏳ In Progress |
| [CampusX: 100 Days of Machine Learning](https://www.youtube.com/watch?v=ZftI2fEz0Fw&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&index=1) | ⏳ In Progress |
| [FastAI: Practical Deep Learning for Coders](https://course.fast.ai/)                      | 🕒 Not Started |

---

# Day 1

## Machine Learning Revision

### Power Transformer

I learned about power transformer class in Scikit-learn. I basically got to know that in order to train machine learning models our data must have normal distribution so in order to make our data normal, one of the most efficient way it power transformation. I used it practically on a concrete dataset from kaggle. Most of the data were not normally distributed, one of those can be seen below through distplot and QQ-Plot below.

<img width="1150" height="393" alt="image" src="https://github.com/user-attachments/assets/bb4a407e-4b59-4ff6-b41b-df220a05bf64" /> 

I used linear regression model without any transformation first and got the accuracy of ~62% and after using box-coz transformation I got 80% accuracy, however on crossvalidating it was around 46%.
Later on using 'Yeo-Johnson' my results were better than previous I got 81% accuracy and ~60% on cross-validationg. 

One example before and after the Yeo-Johnson transformation: 

<img width="1662" height="545" alt="image" src="https://github.com/user-attachments/assets/885e85be-d880-4229-ae7e-28b1309891ef" />

Next, I learned about Binarizer class in scikit-learn. Sometimes we might neet to map a value greater than a threshold to 1 and lower than that would be zero, in that case we can use this class in order to transform the features. 

### Convolution Neural Network

Object Localization

I learned how objection localization is done. We need bounding boxes which will help the model understand where exactly is the object located in the image.

<img width="1893" height="1079" alt="image" src="https://github.com/user-attachments/assets/a970bccf-1dd7-42cf-8239-28cb477824d5" />

We need output like y = [
                        Pc (probablity)
                        bx (x coordinate of the center of the bounding box)
                        by (y coordinate of the center of the bounding box)
                        bh (height of the bounding box)
                        bw (width of the bounding box)
                        c1 - label 1
                        c2 - label 2
                        c3 - label 3
                        ]



# Day 2

## Machine Learning Revision

### Handeling Mixed Variables

I started with handeling mixed variables today, where I took a toy data with mainly 2 kinds of problem with mixed variables. Firstly, one with the numeric and caterogical values in same column. For that we can use to_numeric() function in pandas to extract numeric data and make a new column, and for the remaining categorical data we can fillup with the categorical data from the orignal data ( for columns which has NaN values in the new numeric column). Here's an example: 

<img width="1070" height="615" alt="image" src="https://github.com/user-attachments/assets/e1bb9410-4be7-4085-b9ba-a9edf383fd26" />  <img width="1082" height="383" alt="image" src="https://github.com/user-attachments/assets/d2f14baf-2a90-49ba-b20a-b5c627eebff6" />

Next, for the columns where numeric and categorical data we have to seperate both numeric and categorical and put them in a seperate columns. 

<img width="1399" height="645" alt="image" src="https://github.com/user-attachments/assets/fe356990-e980-4109-b885-c62920271dee" />

Also learned about how to handel date and time, I learned to use a pandas function pd.to_datetime() which allows us to change the object datatype to datetime datatype. It helps us to perform other important operations and extract datas. 

<img width="558" height="596" alt="image" src="https://github.com/user-attachments/assets/6a636ff3-ada9-40c7-85d1-c29a703bc05c" />     <img width="985" height="621" alt="image" src="https://github.com/user-attachments/assets/b205781f-5d8c-4fe7-a400-813c0fab1f1f" />

## Convolution Neural Network

So today I went deep into understanding how object localization is done, first landmark detection where we use CNN to locate where a certain object is present in the image. And next, about bounding boxes and how we spicify where exactly is the object located. For that, first train model with bounding boxex and with the stride and label it as 1 or 0, and bounding box is created whereever it is true, first the center of the bounding box bx,by and the width and hight of the bounding box. 

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/661af6b9-f130-403e-a589-7532f9e462b0" />


We use convolution instead of FC because it gives the prediction for all the regions at once. 



# Day 3

## Machine Learning Revision

### Handeling Missing data

I learned how we can deal with the missing data, there are several ways to deal with them such as Removing (CCA - Complete Case Analysis), and Imputation. There are other sub-parts of imputation like univariate and multivariate imputation which I will be studying tommorow. 

Today however I learned about a least important topic still useful sometimes, in this method we complete remove the whole row if one value in any of the column is missing and we call it CCA. In order to use this method we need to make sure that the values that are missing are 'completely at random' we cannot do it if top 50 or bottom 50 values are missing. And the other condition is that we generally use this only if upto 5% data is missing (at max). 

<img width="774" height="695" alt="image" src="https://github.com/user-attachments/assets/89db1e4b-c8ad-4bef-8826-4322c76c1bf6" />

Here I used a dataset and performed CCA on one of the column and the red is the orignal data ( slightly on the top ) and blue is after performing CCA. This shows that the data were missing completely at random.


# Day 4

Today I learned about Insertion over Union (IoU) in object detection. So here I got to know that we have the actual bounding box and the predicted bounding box and the total area of both of the boxes including where they overlap is the Union. And the area where they intersect is intersection. When we divide the area of the intersection with the total area of the bounding box, we get certain number and if greater than 0.5, we consider that true prediction and false otherwise. 

<img width="1899" height="1012" alt="image" src="https://github.com/user-attachments/assets/13a23123-71c4-4005-9054-7a29f8d4af62" />


# Day 5

## Non-Max Supression

Non-Maximum Suppression (NMS) is a post-processing step in object detection that eliminates redundant bounding boxes for the same object. Object detectors often predict multiple boxes with high overlap—NMS helps by keeping only the most confident one.

<img width="1731" height="968" alt="image" src="https://github.com/user-attachments/assets/b4cfda9b-39b5-43a2-a125-24822b5102b0" />

Steps:

Sort boxes by confidence score.

Select the highest score box.

Remove boxes with high IoU (e.g., > 0.5) with the selected box.

Repeat until no boxes remain.



# Day 6

Anchor boxes are predefined bounding boxes with different sizes and aspect ratios used in object detection models like YOLO, SSD, and Faster R-CNN.

They help detect multiple objects of different shapes at the same location.


<img width="1746" height="974" alt="image" src="https://github.com/user-attachments/assets/e9626686-ea0c-423c-bc4c-af3c58462e45" />


### Key Points:
- Each grid cell predicts offsets for several anchor boxes.
- Anchor boxes have fixed sizes (e.g., 1:1, 2:1, 1:2).
- The model learns to adjust these boxes to match real objects.
- Useful for handling overlapping and varied-shaped objects.

**Example:**  
If a grid cell has 3 anchor boxes, it can predict 3 different object candidates at once.



# Day 7


## YOLO (You Only Look Once)


<img width="1102" height="625" alt="image" src="https://github.com/user-attachments/assets/d636da91-4a7d-403b-860b-719032626df9" />


YOLO (You Only Look Once) is a real-time object detection algorithm that frames detection as a single regression problem, directly predicting bounding boxes and class probabilities from an input image in one evaluation. It divides the image into a grid, and each grid cell predicts a fixed number of bounding boxes along with confidence scores and class probabilities. Unlike traditional methods that use region proposals followed by classification, YOLO performs both tasks simultaneously, making it extremely fast and suitable for real-time applications. It learns global features of the image, resulting in fewer false positives. However, YOLO can struggle with detecting small or overlapping objects, especially when they fall within the same grid cell. Despite this, its balance of speed and accuracy has made it one of the most widely used object detection algorithms in practice.



# Day 8

## Semantic Segmentation with U-Net

U-Net is a convolutional neural network architecture designed for semantic segmentation, where the goal is to classify each pixel of an image into a category. It follows a U-shaped architecture consisting of a contracting path (encoder) that captures context and a symmetric expanding path (decoder) that enables precise localization. Skip connections between encoder and decoder layers help recover spatial information lost during downsampling. Originally developed for biomedical image segmentation, U-Net performs well even with limited data and has become a popular choice for tasks like medical imaging, satellite imagery analysis, and road segmentation. Each output pixel is assigned a class label, resulting in a full-resolution segmentation map.

<img width="1893" height="1001" alt="image" src="https://github.com/user-attachments/assets/6b573cfc-8cf8-4189-bbda-87b3b40a0b4c" />  <img width="1230" height="620" alt="image" src="https://github.com/user-attachments/assets/4238b670-380c-4d6f-b10a-8dd1712b6cb4" />



# Day 9

## Transpose Convolution in U-Net

<img width="1894" height="996" alt="image" src="https://github.com/user-attachments/assets/c44d2311-c902-4ad6-bc9e-d894c1610636" />    <img width="1916" height="1074" alt="image" src="https://github.com/user-attachments/assets/5dd8c826-b39d-4064-8f93-147a78550d66" />



In U-Net, transpose convolutions (also known as deconvolutions or up-convolutions) are used in the expanding path (decoder) to upsample feature maps and recover spatial resolution lost during downsampling. Unlike simple interpolation methods, transpose convolutions are learnable layers that can learn how to upsample in a task-specific way. This allows the network to reconstruct high-resolution output with better precision. Each transpose convolution is typically followed by concatenation with corresponding encoder features via skip connections, enabling the model to combine semantic and spatial information for accurate pixel-wise predictions in semantic segmentation tasks.


# Day 10 



<img width="1322" height="715" alt="image" src="https://github.com/user-attachments/assets/b84c36ee-26f3-44ce-a067-ee4eab871f95" />
