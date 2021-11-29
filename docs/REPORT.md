# Text-Image Scanner  
Project Report CS301-Group 2  
By  
Johann Winter, Raj Nayak, Rishik Danda  
To  
Prof. Pantelis Monogioudis  
November 28, 2021  

## Abstract:  
The current text-image scanners models are not as accurate and efficient in detecting each character in a text. We will be investigating the accuracy and efficiency of converting images containing characters to text that a machine can recognize. Based on our research, we also plan to develop our own text image reader that can predict each character in the word accurately. Our approach will be to extract each character, plug it into the model and output the layer. We will then evaluate our results by a quantitative metric, focusing on accuracy and efficiency by minimizing our loss function as well as minimizing loading time. The data will be separated into three categories, text from a computer (Word Document), text from everyday objects, and handwriting. All of these categories will have different bases and cannot be analyzed in the same context. In doing this we will be able to measure our algorithm using three different metrics, based on the difficulty to predict correctly thus outputting the text accurately.  

Our model was able to recognize individual characters well, but was unable to recognize words efficiently. We believe that these issues could have been due to overfitting or other issues with our model.  

## Introduction:  
As students in CS301 we were assigned to propose a real-world problem and apply Machine Learning (ML) to solve it, by working on it as a team of three members. All the algorithms and experiments are to be discussed along with the approach. The problem we came across is that most text-image scanners have low accuracy and high loss function.  We are trying to solve this by creating an accurate and efficient text-image scanner model, which recognizes characters/text from objects, documents and handwriting.  
    
Since we are advancing in technology, many softwares tools are available as open source. One of them being Tesseract, which is an open source optical character recognition (OCR) platform. For our project we will be working with this software along with some others such as OpenCV, Tensorflow and more to investigate the accuracy, efficiency and proper implementation of these tools. Understanding the implementation of OCR along with these other tools can aid vision impaired users that are unable to read the text from books, documents or images by allowing them to scan the object and use voiceover utility to read it out to them. Furthermore, it can also convert large documents into an electronic file of searchable text, eliminating the need to retype things and allowing functions such as copy/paste to work.  
    
From the results, we found that in each category the accuracy of individual characters increased by training the model from the data set.  As a result, our algorithm was able to predict all characters accurately from different types of text, thus minimizing the loss function as well as the loading time.   
 
## Related Work:  
The material will be referenced from Nanonets and other online resources for the concept, algorithms and datasets. For our project we require multiple datasets to train and test the program. The work published on the website Nanosets talks about the network architecture, which consists of three parts: convolutional layers, recurrent layers and transcription layers. This architecture will be used to extract a feature sequence from an image, predict a label distribution for each frame and translate the frame predictions into the final label sequence. However, the website only uses PyTesseract OCR to recognize text from objects and not handwriting. We will be implementing multiple datasets along with libraries such as Tensorflow and Keras to train the program in recognizing text from objects as well as handwritten text.  

## Data:  
The data we used is a dataset of images containing handwritten characters, annotated with the character data itself. The roughly 800,000 images in our dataset were split into categories based on the character featured in the image. In order to increase the accuracy of the text recognition and train the model, OpenCV was used to modulate these images before inputting them into the model. This, in turn, improved the contrast of the images and the results of the scanner. The data values will be the percentage of accurate characters analyzed by the text scanner. For example, if the scanner reads the word excellent as “exce11lenT”, the data value will be .8 (uppercase and lowercase will not affect the data value).  

## Methods:  
We attempted to create a program that would incorporate all forms of text image recognition but we soon realized that this approach would be complex. We instead decided to split our program into multiple categories, each of which was created to operate on different types of text in images. The categories are handwritten, printed and unstructured objects.  

We began researching the topic and found that many others took similar paths. We started by using OpenCV to modify the images. This was the best approach because it allowed us to clear up originally noisy images into easier to scan, denoised text. After grayscaling, we created a method to crop the images into multiple images, each containing a single character, to scan and identify them.  

![](https://miro.medium.com/max/640/1*P4UW-wqOMSpi82KIcq11Pw.png)

## Experiments:    
The experiments conducted prior to deep learning used OCR for text in the form of handwriting. However, it was ineffective at two classical techniques: character extraction and feature extraction. Since, it required the knowledge of unexpected characters/symbols. For our project we used neural networks to learn features from analyzing a dataset of images, and classifying an image based on weights. The concepts we used were based on how features are extracted in convolutional layers, basically extracting particular features from the image. Furthermore, we wanted to train the model, so using classical methods would have not been optimal as with neural networks, more advanced parameters are learned during the process. This makes it more precise and less likely to change with certain handwriting styles. The most common data set used by many published works is MNIST, which is a data set with handwritten digits for training a model.   

We conducted our testing on a large data set of handwritten characters. The data set that we used included handwritten images of each letter, digit and symbol. The goal was to train the model on individual characters and evaluate words by splitting them into a list of characters. We decided to crop each letter from the text image and test it for the model accuracy. The way we tested for accuracy was using the existing concept of normalization, training and then getting an average of the tested characters. For example: Taking data set of letter B, using the algorithm to test the accuracy and then taking the mean of the recognized letter or character from the cropped image. Similarly, to the image displayed below with the digit 4.  

The main issue we were facing was merging the text in a sorted order from the list of cropped images containing characters. As when the image was cropped the characters were recognized by the model accurately, however storing them into an array resulted in mismatched array dimensional errors.  

After fixing previous errors, the model was relatively functional, albeit inaccurate. We attempted to increase the model accuracy by changing various parameters during training. We believe the issue could be attributed to overfitting, and retrained multiple models to test their results. Since the dataset was significant in size, retraining took many hours and would often have to run overnight. Through this, we were not able to find a model that would accurately predict words, even though our validation accuracy was upwards of 90%.  

## Conclusion:  
Ultimately, through our experiments and research we found out that the best way to interpret a scanned image text is to split the image into individual characters and use the algorithm to predict the character from the cropped image. This way we were able to collect all the individual characters from the cropped images into an array. Later, sorting it and merging it to efficiently and accurately display the whole text.  

Through this project, we have learned a lot about the challenges of machine learning by experiencing them first-hand. Creating a functional model can be very difficult, even for simpler problems such as text recognition. Machine learning can be crucial to certain problems and understanding the functional aspect of them is important.  

To improve upon our project, we would need to address the inaccuracies within our model and make changes based on our findings. We would also try to extend our project to work on printed and unstructured text, which we were originally planning for but ran into time constraints. Even though our models were not that effective in predicting text from images, the structure of our approach can be used as a framework for future projects attempting to solve this problem.  
