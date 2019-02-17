# Face Recognition and more with KERAS

#### Keras implementation of the paper: [FaceNet: A Unified Embedding for Face Recognition and Clusterin](https://arxiv.org/abs/1503.03832)

![alt text](https://github.com/Golbstein/keras-face-recognition/blob/master/assets/face_reco.JPG)


* ## Dataset: 
  - **[CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)**

* ## Dependencies
  - **Keras 2 (tensorflow backend)**
  - **open-cv**
  - **tqdm**
  - **pandas**
  
* ## Model
  - Feature extractor model: Xception
  - Embedding model: FaceNet
  
  ![alt text](https://github.com/Golbstein/keras-face-recognition/blob/master/assets/openface.jpg)

  
- [x] Recognize celebrities with trained FaceNet model
- [x] Find out your Doppelgänger
- [ ] Beauty test

* ## Loss
  **[Triplet Loss](https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24)** with cosine similarity
  
  ![alt text](https://github.com/Golbstein/keras-face-recognition/blob/master/assets/obama.png)
  ![alt text](https://github.com/Golbstein/keras-face-recognition/blob/master/assets/loss.JPG)
