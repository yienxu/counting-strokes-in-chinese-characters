## Counting Strokes in Chinese Characters

Authors: 

Yien Xu (yien.xu@wisc.edu)

Yuqi Lin (ylin273@wisc.edu)

Scott Lai (qlai5@wisc.edu)

#### Abstract

We use machine learning algorithms to count stroke number of Chinese characters that are in image form. As Chinese international students who are proud of our language, we hope to find a good method that helps more people learn Chinese. Hence we make a stroke number recognition system for Chinese character that if we input a 28 x 28 pixels image of a Chinese character, we are able to tell how many stroke(s) it has. To unify the definition of stroke features, we use the definition in Unicode as one of our databases, which includes 36 stroke features in total. We also divide our dataset into 90% training and 10% testing sets. The models we use include k-Nearest Neighbors algorithm (kNN), Logistic Regression and Convolutional Neural Network (CNN), combined with k-Fold Cross Validation (k-Fold CV). We use Euclidean Distance for distance calculation and choose k = 15 for the Cross Validation in kNN algorithm, and find the test accuracy is about 20\%. We also try Logistic Regression model by first inverting the image into black background, and then applying L1 regularization and using the optimizer SAGA to train. We obtain a test accuracy lower than kNN. By adding Bagging to Logistic Regression, we improve the accuracy a little bit. Logistic Regression with Max-Pooling gives the lowest test accuracy among all the models we use, and the highest test accuracy we gain is about 50% by using CNN.

#### Contribution

Yuqi Lin found the Unihan Database that contains stroke count information of Chinese characters. In addition, she coded the complete version of the kNN model, using PCA and `GridSearchCV`. She also wrote code for the Logistic Regression model. Yien Xu processed the original dataset and turned it into a single csv file. He also coded the rest of the models, including Logistic Regression with Bagging and Max-Pooling, and the CNN. Both Yien and Yuqi worked together and finished the project report. Finally, Scott Lai did minor contribution to the project. He found the original dataset from the web that contains 15 million images. He also coded the kNN model at a very superficial level.

#### Report

Please find our project proposal [here](Proposal.pdf) and project final report [here](Report.pdf).
