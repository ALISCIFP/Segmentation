# Segmentation

dataset:http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Git operation: git status-> git stash-> git pull-> git stash pop-> git add filename ->git commit -m'update fileneme:add new function'-> git push.

For undestanding similarity transformation
https://cseweb.ucsd.edu/classes/wi04/cse291-c/hw4.htm

Paper:
http://arxiv.org/pdf/1411.7766v3.pdf

Data Preprocessing:
1.Get the mean shape A_bar

2.Get the mean image

The i-th image I_i
The i-th annoation A_i
The i-th similarity transformation X_i
A_i*X_i=A_bar


Get all the X_i and save it in a hdf5 format. the key is i and the value is teh X_i

Then for each image I_i, Do I_i * X_i, then resize I_i to 224 * 224
