# Mechanical-Fault-Diagnosis-Based-on-Deep-Learning
CNN for mechanical fault diagnosis  

These codes serve for two papers: 'Rolling Element Bearings Fault Intelligent Diagnosis Based on Convolutional Neural Networks Using Raw Sensing Signal'(paper_1) and 'Bearings Fault Diagnosis Based on Convolutional Neural Networks with 2-D Representation of Vibration Signals as Input'(paper_2).   

Prerequisite, Matlab 2013a, Python 2.7.11, Tensorflow (better in ubuntu14.04).   

To start with, you should run image_matrix.m to prepare your own data.   

second, disorder_images.py and input_bear_data.py should be used to tansform your data into the input format of tensorflow.   

last, you may choose to run mnist_b.py or mnist_c.py for paper_1 or mnist_2d.py for paper_2.   

如果大家对以上研究感兴趣，可以进一步参考我的这两篇论文：A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals与A deep convolutional neural network with new training methods for bearing fault diagnosis under noisy environment and different working load，更详细的请看我的硕士毕业论文：基于卷积神经网络的轴承故障诊断算法研究。若使用，请引用，谢谢。
