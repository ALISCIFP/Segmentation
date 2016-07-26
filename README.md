# Segmentation

dataset:http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Git operation: git status-> git stash-> git pull-> git stash pop-> git add filename ->git commit -m'update fileneme:add new function'-> git push.

For undestanding similarity transformation
https://cseweb.ucsd.edu/classes/wi04/cse291-c/hw4.htm

Paper:
http://arxiv.org/pdf/1411.7766v3.pdf

Data Preprocessing:
1.Get the mean shape A_bar
note: the A_bar is normalized so all the Ai will be normalized and saved to file too.
2.Get the mean image

The i-th image I_i

The i-th annoation A_i

The i-th affine transformation or Thin plate spline matrix X_i

Get X_i from A_bar*X_i = A_i_prime

minimize L2(A_i-A_i_prime)

The i-th image after transormation I_i_prime,its dimension is 224*224

I_i_prime*X_i=I_i to sample the pixel value from I_i to I_i_prime

then mean the I_i_prime to get the mean image I_bar.

ImageNet 2016 Scene Segmentation Task: http://sceneparsing.csail.mit.edu/


http://vision.stanford.edu/teaching/cs231n/slides/caffe_tutorial.pdf

Face Landmark dataset: CeleA
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
