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






