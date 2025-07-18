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

## Day 1

### Machine Learning Revision

Power Transformer

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







