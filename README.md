# mldl-101

## Step 1
Welcome to mldl-101 lab!  Please follow these instructions to get started with your lab environment running on the Nimbix Cloud.

  Navigate to https://platform.jarvice.com/
  Type in your provided userid / password
  
![](https://github.com/dustinvanstee/random-public-files/raw/master/ss1.png)

---
## Step 2 
Once you have succesfully logged in, you will be redirected to the Nimbix Jarvice dashboard.  
1.  Click compute on the right hand side of the dashboard
2.  Type mldl_education to find the class application
3.  Click the lock icon in the application tile
![](https://github.com/dustinvanstee/random-public-files/raw/master/ss2.png)

---
## Step 3
Next click on the server button to deploy the server
![](https://github.com/dustinvanstee/random-public-files/raw/master/ss3.png)

---
## Step 4
Now select a Minsky server with one GPU.  
1.  Click the machine type selector, and pick the ***np8g1*** machine
2.  Click Submit
![](https://github.com/dustinvanstee/random-public-files/raw/master/ss4.png)

---
## Step 5
Next your server will be deployed, wait a coupled of moments and you will see the desktop of your server appear.
1. Click anywhere within the desktop
2. This will open a new browser tab 
![](https://github.com/dustinvanstee/random-public-files/raw/master/ss5.png)


---
## Step 6
Next deploy your terminal to run a script that will download our class from github
1. Click the start menu button (lower left corner)
2. Start a terminal 
![](https://github.com/dustinvanstee/random-public-files/raw/master/ss6.png)


---
## Step 7
Next run the script that will setup the class environment and startup the Jupyter notebook
1. type ***sudo /root/bootstrap.sh***
![](https://github.com/dustinvanstee/random-public-files/raw/master/ss7.png)


---
## Step 8 - Final Setup Step.  
1.  Copy the hostname and paste in a new brower window and postpend :5050 to open up your Jupyter Notebook.  
![](https://github.com/dustinvanstee/random-public-files/raw/master/ss8.png)

---
## You are now ready to proceed with the labs!
1. Start with Lab1-GPU=CPU-tensorflow.ipynb in your jupyter notebook

..
