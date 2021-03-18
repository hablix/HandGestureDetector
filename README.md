Name: Hannes Bischoff
Academic year: 2020/2021
Course: intelligent Systems

# Hand gesture detector on android phone with pre trained TensorFlow Light Model.

a project at University of Rijeka





## MOTIVATION
The original idea behind this project was to develop and android app which uses the camera to detect different hand gesures. This could assist people who dont understand the finger or hand alphabet to 'read' or understand the what a person shows in sign language. This is a very complicated and complex process. I was thinking about how to reduce complexity to create a arcvievable and realistic project for intelligent systems course. This project should more or less be a proof of conecpt that it is possible to detect different hand gestures on an android phone by using the convolutional neuronal network (cnn) with a pre trained TensorFlow Light Model for image classifying. Furthermore this small report will highlight the limitations and challenges i experienced in this process.


# TECHNOLOGY

The artifical intelligence part will be coverd by the convolutional neuronal network. I decided to use TensorFlow becaus it was used in the exercises during the semester so i already used it and had experience in it and because the light version of TensorFlow models could be run in an android app, which makes it perfect for mobile use in the described use case.

Especally for the the transfer learning for image classification i used TensorFlow Lite model maker library with Keras ub layer as base of the model.


# GLOSSARY
TensorFlow
Is an open source library from google for deep learning. which can be used for training of neuronal networks and is implemented in python and c++;

TensorFlow light
Version of Tensorflow especally designed for moblie devices. It can be used to execute models, but it is not possible to train models. Modelc can be run on android for ecample for image classification.

TensorFlow (Light) model
Model is a cluster of Tensors and layers wich processed data in the way it was trained for. Light version of this is cn be run on android app.

Hand gestures. The 'german' hand alphabet knows 31 'letters'. From there, 5 signals are represented with a motion component, but this image classifier can only detect static pictures so there would not be  possible to detect all letters. 

![iamge](https://github.com/hablix/HandGestureDetector/blob/main/.graphics/Folie10.PNG)


To reduce the compleety i decidet to train the model with 5 hand gestures which are easy to recognise. (thmub up, fist, palm, letter y, letter f)

![iamge](https://github.com/hablix/HandGestureDetector/blob/main/.graphics/Folie11.PNG)



Perspectives from which the hand gesture can be viewd. I definded 3 categorys which i refer to.
* 'best angle'
* 'non perfect angle'
* 'indistingusable angle'


![iamge](https://github.com/hablix/HandGestureDetector/blob/main/.graphics/Folie9.PNG)


## PROJECT

The project conists of the following main parts.

1. Image library (Data set)
2. Training of the model (Model maker)
3. Android app (App Gesture Detector)

![iamge](https://github.com/hablix/HandGestureDetector/blob/main/.graphics/Folie4.PNG)



## 1. Image library (Dataset)
I decided to use my own image libary because of the strict quality input limitations for the training process of the model. Second, this provides full control of what gestures are represented and how the gestures are showen. This offers the posibility to take the pictures with the same camera which is used for training and afterwards for recognising. During the process I gained a lot of experience about which pictures are usefull for the training process. -> see limitations problems. 
I had to make decicion which approach i want to use: if i want to recognise gestures from every perspective or to recognise from the best angle.

Every perspective needs a lot of data. In the experiment i used ~150 pictures for each gesture from multiple angles. This did not lead to an useful model. and it resulted in disaster - > regognision on the phone was as low as 10% even for showing the gesture in 'best angle' the reason for this was there are multiple images in the training process where it is impossible to distinguish betwen the gestures (thumb up, letter y). I assume this problem could be fixed by using a very much larger data set. This works in different models and different sources state that this is not a mayor problem for AIs. But due to my limitations in efforts and unsecure outcome i decided to train the model with images from best angle perspective.

The second mayor issue it the background. Humans can be taught easily wht the object is by presenting it on a plene background. And the person will regognise this object on any background, beause it knows the background does not matter. The imag clasifier on the other hand does not get any further 'explination' so it learns and trains itself by finding the similaitieies between the images. The images which have some consistent parts will be classified as one class, and so represent the same showen object. So it does not distinguist between background and object and therefore it can not ignore the background. This lead to very low sucess rate afer the first experiment on the phone. The data set contained pictures with more or less the same background. In the experiemnt the background did not matched the one from the training images at all and results were very low. So the best way to train the model is indeed by changing the background as much as possible. So the consistent part in the images becomes the object which will be the decisive part and background is insignificant.

With this knowledge i took pictures with of handgestures with different backgrounds from best angle and non perfect angle. But i removed pictures from indistinguisable angle from the dataset. this proved to lead to the best results. in the end i generated more than 210 pictures for each gesture.

Anothe huge improvement in recognision was data argumentation befor feeding this into the model. Rotation, zoom and brightnes adjustment helped a lot.

The pictues were taken in defalt phone quality and compressed to 180px x 180px for further procedures and storing the data set.

During the whole project the image library was changed and improved all the times after each experiment where i gaind new experiences.




## 2. Training of the model (Model maker)

### Training

For training the model I used Google Colaboratory. This so called web IDE for Python enabels to run python code or jupiter notebooks without setting things up on local computer. It provides acess to Googles Computing power and GPUs for faster execution in building the model. The advantages are, there is no need to install all the project related libraries on local computer. During the developping there is no danger from executing and debugging unknowen code from different sources on the internet. Faster execution in building the model. But it also made it necessary to store the dataset in Google drive for easier access. So I reduced the images resolution to 180px x 180px. 

I used the TensorFlow Lite Model Maker library.(https://github.com/tensorflow/examples/tree/83a8b6edfa03fca856b8817c29a06c9d93d4f34b/tensorflow_examples/lite/model_maker)
This Library simplifies training of a TF lite model with custom dataset. It privides transfer learning, data argumentation and different custom settings. With heplp of this library the amount of training data and the training time can be reduced.

During the project i recently changed the used dataset and the configuration of the model. (https://www.tensorflow.org/lite/tutorials/model_maker_image_classification)

Some importand variables here are:

* Dataset
There has to be found a tradeoff between training data, validation data and test data. The best match was:


* training and validation data
Data which is used for training and validation. It is considerd as good tradeoff to use a split of 80/20.


* Dropout
Technique to randomly remove given amount of neurons in each layer to avoid overfitting.


* Epochs
One epoch is the full iteration by the learning algorithem over the entire training set. Increasing the number of epoches can achieve better accuracy until a certain level. Too many epochs lead to overfitting.

* batch size
number of samples which is used in a training step.


* Data argumentation(https://github.com/tensorflow/examples/blob/83a8b6edfa03fca856b8817c29a06c9d93d4f34b/tensorflow_examples/lite/model_maker/core/task/image_preprocessing.py#L186)
input images are modified a bit to represent a larger vayrity of images. This consists of croping the image, rezising the image and flipping the image. 



### Evaluation

After creating and training the model, the model can be evaluated and if satisfying saved as a TF lite model. This TF light model can then be integrated in the android app.


To decide the quality of the created model I used different methods.

* The loss and accuracy values of the model.evaluate() method. The lower the loss value and the higher the accuracy value the better the modell is. This gives a quantitative overview.

* For a qualitative impression i plotted a map of 100 random images with the predicted label and the real label. This made it visible in which actual cases the prediction is not working. Are these fails ambiguous images, where the perspective does not allow any distinction or are there other reasons? This evaluation had huge impact of how i decided to take the next pictures for the dataset.

* Test on the device. After including the TF lite model on the phone the final tests under real circumstances can done. I tested the recognision capabilities under different backgrounds lightning and perspetives. The results were changing a lot and sometimes contradictios to the quantitative and qualitative evaluation. But with large enough datasets and the gaind knowledge this issues could be sloved.

There was a constand feedback loop between taking pictures for dataset, modifying pictures, customising the model and evaluating the new model. I repeated this with different experiments and settings multiple times. The gained knowledge I used to improve the settings and circumstances but it also lead me to define certain limitations. see leimitations.

![iamge](https://github.com/hablix/HandGestureDetector/blob/main/.graphics/Folie13.PNG)



## 3. Android app (App Gesture Detector)

Konstantly updated with the newly built model.

First i wanted to create an java android app from scratch, which i have done earlier before. But during the process i found out that there is a number of existing tutorials and example apps. After few hours of coding I frankly decided to take an open source example app from (keras) which already makes use of devices camera and sets the basics for Tensor flow light model integration. I decided to do this beacuse the app uses a steady stream of images from the camera and has a good reailability while following the android design and coding prnciples. Doing this on my own would have take me a lot of time and effort by not even reacing the same quality. To fulfill this projects requirements i developed on base of the example app. I chaged the integration features, behavior and added extra features.




![iamge](https://github.com/hablix/HandGestureDetector/blob/main/.graphics/Folie12.PNG)




# RESULTS and LIMITATIONS
* adding a no gesture is showen class
* some letters are only working with movement -> not able to detect with continuesly but static picture classifyer
* limited to the gesture alphabet, sign language is more about actions words and needs a whole body movement detector, this is not part of this project.

* ambiguity of some hand gestures from some angles.
* recognising hand gestures from different angles is compliated-> from some perspective they look the same (thumb up, letter y)
* deciding if image classyfier should work with hand gestures from all perspectives OR from the 'best angle'
* limited to the left hand.
* only from best angle
* different backgrounds can cause problems

# Rsuts


# SORCES

https://hoerbehindert.ch/information/kommunikation/fingeralphabet

https://www.tensorflow.org/lite/tutorials/model_maker_image_classification

https://www.tensorflow.org/

https://github.com/tensorflow/examples/blob/83a8b6edfa03fca856b8817c29a06c9d93d4f34b/tensorflow_examples/lite/model_maker