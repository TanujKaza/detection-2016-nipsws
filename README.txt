Object Detection using Deep Reinforcement Learning

                     150050008 - Tanuj Kaza
                     150050054 - Charles Rajan
                     150050057 - Manas Bhargava
                     150050110 - Bhavya Bahl


Project Goal 

In the past few years, object detection has attracted much research attention with the explosion in deep learning methods that enable more complex features to be learnt by the system [1]. Much of this work has focused on using templates and other image features to aid in object identification. We propose a deep reinforcement learning framework that uses Generative Adversarial Networks (GANs) [2] as a reward function for object detection in the image as the object being detected belongs to the dataset distribution. Our initial goals include identification for a fixed set of objects while we hope to extend this to a larger dynamic set contingent on time and resources. 

Introduction 

The idea of using Deep Reinforcement Learning for Object Detection has been explored by other research groups in the past. Miriam Bellver et. al [3] propose focusing on parts of the image that contain richer information (the object) for identification. This is a rather intuitive as it’s similar to the way humans assimilate information in an image. The learning agent stops when the object has been detected, depending on the Intersection Over Union (IoU) score of the zoomed in image with the ground truth image, which is a templated image. We propose using a Generative Adversarial Network to generate the rewards in the MDP on the basis of the distribution to which the identified object belongs to. In the following sections, we further elaborate on the formulation of the problem, datasets which we propose for the conduction of experiments and suitable baselines. 


MDP Formulation
 
We describe the Markov Decision Process that would be employed in the RL formulation below by defining the state, action and reward space. This is quite similar to the MDP formulation employed by [3] barring the reward function which we define below.

States

The state comprises of the given image and a memory vector which stores the past four actions that have been taken to reach the given state. This ensures that the trajectory doesn’t consist of the same action repeatedly being performed. As the agent is learning to refine a bounding box for object detection, a memory vector that encodes the state of this refinement is useful to stabilize the search trajectories.
    

Actions

There are two types of possible actions, namely movement actions that imply a change in the current observed region and the terminal action that indicates that the object has been found and the search has concluded. There is a predefined hierarchy to determine the actions that can be taken from each state. A hierarchy is built by defining five subregions over each observed bounding box : four quarters and a central overlapping region. Thus there are five movement actions, each one associated with one of the regions described above. If the terminal action is selected, there is no movement and the search concludes.

Reward

Our approach differs from the approach followed in [3] in the reward function being employed. Once the terminal action has been taken, we will use a GAN to evaluate how close the image is to the target dataset distribution. This is quite similar to the approach employed by Yuanming Hu et. al, [4] where the negative of the earth mover distance is used as the reward function.


Approach

We would initially start with identifying simple objects in the image like a dot using the formulation given above. On achieving success in this task, we would aim at finding objects present in the dataset, for instance images of cats with a plain background. Our next step would be to find these objects in an image having more complex backgrounds following which we can target different objects present in the dataset.

Dataset

We would be using the Quick Draw Dataset [5] which is a collection of 50 million drawings across 345 categories, contributed by players of the game Quick, Draw! This contains many images of different objects which we would use for object detection.

Baselines

The baselines on which we would evaluate our experiments would be the numbers reported in[3].

References

[1]  Zhong-Qiu Zhao, Peng Zheng, Shou-tao Xu, and Xindong Wu. Object detection with deep learning: A review. CoRR, abs/1807.05511, 2018c. URL http://arxiv.org/abs/1807.05511.

[2]  I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems (NIPS), pages 2672–2680, 2014.

[3]  M. Bellver, X. Giro-i Nieto, F. Marqu ´ es, and J. Torres. Hierarchical Object Detection with Deep Reinforcement Learning. arXiv:1611.03718, 2016

[4] Exposure: A White-Box Photo Post-Processing Framework 
Yuanming Hu, Hao He, Chenxi Xu, Baoyuan Wang, Stephen Lin https://arxiv.org/abs/1709.09602

[5] Quick Draw (https://github.com/googlecreativelab/quickdraw-dataset)
