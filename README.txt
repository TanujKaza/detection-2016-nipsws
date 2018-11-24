Object Detection using Deep Reinforcement Learning

How to run this script - 

1) generate datasets - 

	""""PLEASE perform the following command inside the './data' directory""""

a) To generate the toy dataset of dots in the image, run the command - 
	python3 dot_without_bg.py

b) To generate the datasets corresponding to quickdraw, run the command - 
	python3 generateQuickdrawDataset.py [dataset_name]

c) To generate the dataset to test the functionality of wgan-gp, run the command - 
	python3 convert.py [dataset_name]

Please ensure that you have 128 X 128 images of the quickdraw dataset stored in the 
directory './quickdraw/[dataset_name]/r128'

	""""PLEASE perform the remaining command inside the './scripts' directory""""

2) Scripts for checking the perfromance of gan training - 

	after generating the datasets as mentioned in step one - 
	run the command python3 wgan.py [dataset_name]

3) Script to functionality of our first approach - 
	
	python3 dot_no_bg_train.py

4) Script to functionality of our second approach - 
	
	python3 reinf_gan.py -d [dataset_name]

	and train this command till convergence

5) Script to functionality of our third approach - 
	
	python3 reinf_gan.py -d [dataset_name] -g True

	and train this command till convergence


