# Deep Learning for Video Stabilization

Problem Statement *
Mention your problem statement as precisely as you can. This should clearly identify the input and output. Provide (clearly focused) links to images to your external document. We are expecting to see 4-6 statements. Don't forget to mention the supervision needed.

Videographers make use of camera stabilization gimbals to prevent physical disturbances from deteriorating their recorded footage. However, these equipment are not perfect in the sense that low frequency disturbances from actions such as walking, running or rolling on bumpy surfaces are not filtered out by such stabilization hardware.
Due to these limitations of optical stabilization tools, and due to the fact that access to such tools is limited to professionals, software based video stabilization is desirable. 
In particular, our project, in collaboration with SysCon ARMS (Autonomous Robots and Multi-robot Systems) Lab at IITB, aims to stabilize the output video stream from a mobile spherical robot - Picture of robot - https://www.sc.iitb.ac.in/embeddedLab.html.

The aim of the project is to compare the following strategies for video stabilization
1. Use of L1 optimal camera paths - Grundmann et al. 2011
2. Subspace video stabilization - Liu et al. 2011
3. Deep online video stabilization - Wang et al. 2018


The first two methods employ mathematical techniques to obtain camera trajectories from a cinematographically pleasing family of paths. These methods are completely unsupervised and do not have a training component to them. Both the algorithms are widely used and the L1 optimal path method has been employed in YouTube video stabilization. 
The third method describes a Deep Learning architecture - StabNet - That outputs a Homography matrix which is used to perform a transformation over an entire frame. It does so while by training the network with a loss that trades-off between stabilization from frame to frame, and temporal coherence.
We will be implementing the above three algorithms in python. The inputs to our system will be shaky, unprocessed videos and the outputs will be their stabilized versions. 
Initially we will be testing the efficacy of the methods on a standard video stabilization dataset and later on the output stream from the spherical robot.

Relevant papers and prior work *
Mention GitHub links and pdf of paper separately on a new line.

[1] Auto directed video stabilization with Robust L1 optimal camera paths
https://smartech.gatech.edu/bitstream/handle/1853/44611/2011-Grundmann-AVSWROCP.pdf?sequence=1&isAllowed=y
[2] SubSpace video stabilization
https://dl.acm.org/doi/pdf/10.1145/1899404.1899408
[3] Deep Onlive Video Stabilization (CNN Based)
https://arxiv.org/abs/1802.08091

Datasets *
What data will you use? If you are collecting new data, how will you do it? For existing datasets, please provide relevant links.
The dataset we would be working with initially is the SIGGRAPH 2013 video stabilization dataset.
http://liushuaicheng.org/SIGGRAPH2013/database.html
Later, if time (and the pandemic situation) permits, we would be evaluating our models on data collected from the ARMS lab spherical robot. 

Extensions/modifications to the original paper
Mention the extensions or modifications to the original paper and briefly justify them.

We plan to deliver the following novel contributions-
1. As mentioned in the problem statement we will be extending the understanding of the algorithms to solve the stabilization problem for a spherical robot where the camera undergoes wavy second-order damped motion.
2. The Deep Learning approach to this problem is quite recent and not much follow up work has been done. Based on our reading of the literature, we plan to add two things to the DL approach. Firstly, in [3], the StabNet architecture only takes historical frames as input to its model for the purpose of predicting a transformation as the output. The authors restrict themselves to prior frames only since they want their application to be online and real time. However, in our application we are willing to allow for a short delay between video recording and rendering and hence we would like to modify the architecture to include a window of frames into the future as well.
Secondly, a more ambitious extension to the reference [3] is to employ GANs to the video stabilization problem. [3] dismisses the use of generative networks on the stabilization problem since as per them there is “severe vibration” in the input video content. We are not quite convinced by this and would like to try for ourselves.

Frameworks and Libraries *
What framework/libraries will you be using for the project. Ex- tensorflow, pytorch, opencv, matlab libraries, etc.
We plan to stick to python in google colab for the deep learning parts of the project and 
Opencv - for data reading, augmentation, rendering etc.
Pytorch - for Deep Learning 
Python PuLP - As a linear programming solver in the implementation of [1]

Link to existing code or non-trivial APIs *
If you are using existing code as your baseline, or as sub-module of your project, mention their repository links. Do not duplicate if this has already been provided in an earlier question.

We might refer to the below links for implementation of reference [1]:
https://github.com/ishit/L1Stabilizer
https://github.com/VAIBHAV-2303/VideoStabilization

Evaluation Metrics *
What kind of analysis will you use to evaluate your results (Example: top-5 accuracy, t-sne plots, etc.)
The metrics that we will be using are :
Cropping Ratio: i.e. area of the remaining content after stabilization
Distortion value: it  evaluates the distortion degree introduced by stabilization
Stability Score which uses frequency-domain analysis of camera path to estimate the stability
The computation time of each algorithm
Finally stabilization results on wavy second-order damped motion video.

How are you planning to divide your work among the team? *
Mention estimated contributions of each group member.

We will separately implement the algorithms in references [1] and [2], thus both of us shall implement one of the papers. For the reference [3], which is based on DL, we would work together to implement, train and test the model.

What systems and training regimes are you planning to use? *
Mention how you estimated the size of the data and the estimated training size. Is your system capable of sufficient training time?
The total size of the SIGGRAPH 2013 data set is about 2 GB, as the first two algorithms do not require any training they can be either tested on our own systems i.e. laptops but to address the portability issues we will stick to Colab for the implementations of all 3 references. 
As per the training information presented in [3], the model should be trainable in about 10 hours of training on a version of the SIGGRAPH 2013 dataset.

What deliverable are you planning for April 3? *
We won't hold you to it, but this April 3 deadline also has marks assigned.
Although the 3rd April deadline was not implemented, an intermediate check-point for our team will be the implementation of references [1] and [2].




