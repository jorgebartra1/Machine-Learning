# Machine Learning - Convolutional Neural Networks
> Select the URL below for more details
https://jorgebartra1.github.io/Machine-Learning/


![imagecnn](https://user-images.githubusercontent.com/14510359/39091327-e85a2fe4-45bf-11e8-9fe6-a094c33bf505.png)

[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]

## Project Overview
Welcome to the Convolutional Neural Networks (CNN) project in the Machine Learning Engineer Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images. Given an image of a dog, your algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

This page is a guide to developers who want to use ANACONDA to perform Machine Learning Models with CNN on their local computers .

![](header.png)

## Development setup

Windows:

Install Anaconda - Python 2.7
```
https://www.anaconda.com/download/
```
Create a new environment with Python 2.7
```
conda create -n myenv python=2.7
```
Install Theano
```
conda install theano
```
Install Tensorflow
```
conda install tensorflow
```
Install Keras
```
conda install keras
```
Install OpenCV3 package
``` 
conda install -c conda-forge opencv 
```
Install PILLOW package (PIL does not work with OPENCV3)
```
conda install pillow 
```
tqdm Package
```
conda install -c conda-forge tqdm
```
Install Scikit-Learn Package
```
conda install -c anaconda scikit-learn
```
Install Matplotlib Package
```
conda install -c conda-forge matplotlib
```


## Usage example

Find if the image used in the model belongs to a HUMAN person. If the image is a human then find the face(s), eyes and smile.

```
## face detection algorithm on the LFW dataset
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')

#print (human_files[10])

# load color (BGR) image
img = cv2.imread(human_files[11])


# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# find faces in image
faces = face_cascade.detectMultiScale(gray,1.04, 3)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    #for (ex,ey,ew,eh) in eyes:
    for (ex,ey,ew,eh) in eyes:    
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        smiles = smile_cascade.detectMultiScale(roi_gray)
        for (x2, y2, w2, h2) in smiles:
            cv2.rectangle(roi_color, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 2)
        
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

_For more examples and usage, please refer to the [Wiki][wiki]._

## Release History

* 0.2.1
    * CHANGE: Update docs (module code remains unchanged)
* 0.2.0
    * CHANGE: Remove `setDefaultXYZ()`
    * ADD: Add `init()`
* 0.1.1
    * FIX: Crash when calling `baz()` (Thanks @GenerousContributorName!)
* 0.1.0
    * The first proper release
    * CHANGE: Rename `foo()` to `bar()`
* 0.0.1
    * Work in progress

## Meta

Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

Distributed under the XYZ license. See ``LICENSE`` for more information.

[https://github.com/yourname/github-link](https://github.com/dbader/)

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki

