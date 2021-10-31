# CS301-Group2

## What is the problem that you will be investigating? Why is it interesting?

We will be investigating the accuracy and efficiency of converting images containing printable characters to text that a machine can recognize. This is interesting because there is text all around us in many different forms and fonts and being able to convert this into a digital medium would enable us to use computers to further process it, possibly into other mediums.

## What reading will you examine to provide context and background? 
https://nanonets.com/blog/deep-learning-ocr/  
We will be referencing the above link to provide context of machine learning and computer text reading. The link includes sample code and examples of text reading that would be invaluable in obtaining data.

## What data will you use? If you are collecting new data, how will you do it?
The data we will use will be a dataset of images containing text, annotated with the text data itself.

## What method or algorithm are you proposing? If there are existing implementations, will you use them and how? How do you plan to improve or modify such implementations? You don’t have to have an exact answer at this point, but you should have a general sense of how you will approach the problem you are working on.

Based on the above reading and existing documentation, we plan to develop our own unique text image reader. Existing python program that can be used as instructions and advice to create our own algorithm. In order to improve the implementations, we plan to test different algorithms and data structures to improve efficiency. 

## How will you evaluate your results? Qualitatively, what kind of results do you expect (e.g. plots or figures)? Quantitatively, what kind of analysis will you use to evaluate and/or compare your results (e.g. what performance metrics or statistical tests)?
We will evaluate our results by a quantitative metric. We will be focusing on accuracy and efficiency by minimizing our loss function as well as minimizing loading time. We will be separating our data into three categories, text from a computer (Word Document), text from everyday objects, and handwriting. All of these categories will have different bases and cannot be analyzed in the same context. In doing this we will be able to measure our algorithm using three different metrics, based on the difficulty to predict correctly.

