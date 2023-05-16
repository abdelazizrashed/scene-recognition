# Scene Recognition

## Introduction
This is a Computer Vision (CV) project.
The project aims to build a scene recognition model that can classify images of different scenes using three different approaches.

### Approaches used
- Tiny images representation and nearest neighbor classifier.
- Bag of words representation and nearest neighbor classifier.
- Bag of words representation and linear SVM classifier.

## Results

After tuning the parameters, we achieved the following results:

1. Pixel per cell = 8, cells per block = 2
2. K Means with the following parameters:
   `MiniBatchKMeans(n_clusters=vocab_size, random_state=0, max_iter=300, batch_size=1500).fit(All_features).cluster_centers_`
3. Linear SVC with c=1.0 (a lot of regularization values between .1:5000)
4. K=1 is the best one worked for us after trying k=1,2,3
5. Vocab size = 200 best one out of 8 different sizes as below:

### Performance

#### 1. Tiny Images with KNN

Accuracy: 23.200%
Comment: Not surprising as we lost most of the information when resizing each image to a small size

Confusion Matrix:
![confusion_matrix_tiny_knn](code/tiny+knn/confusion_matrix.png)

#### 2. Bag of Words with 1NN

Accuracy: 46.800%
Comment: Much better than tiny image, which is not surprising as with bag of words, we have a more sophisticated and representative representation.

Confusion Matrix:
![confusion_matrix_bow_knn](code/bag+knn/confusion_matrix.png)

#### 3. Bag of Words with Multiclass LinearSVM

Accuracy: 52.800%
Comment: Linear SVM gives the best accuracy between all three approaches.

Confusion Matrix:
![confusion_matrix_bow_svm](code/bag+svm/confusion_matrix.png)

