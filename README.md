## Text and Images: Multimodal Learning on SageMaker

Welcome to the Text and Images: [Multimodal Learning on SageMaker workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/47077f46-e2e4-471c-893b-b78345ae72ad/en-US). In this workshop we are going to cover: 

* Download and explore the dataset that contains text, images and tabular data.
* Train a miltimodal autoMM model using a Amazon SageMaker training job
* Perform batch inference using a Amazon SageMaker Processing job

We will use AutoGluon for Multimodal model training and inference. [AutoGluon](https://github.com/awslabs/autogluon) automates machine learning tasks enabling you to easily achieve strong predictive performance in your applications. With just a few lines of code, you can train and deploy high-accuracy deep learning models on tabular, image, and text data. This example shows how to use AutoGluon MultiModal with Amazon SageMaker by using prebuilt AutoGluon containers.

# Getting Started

For this workshop you’ll get access to a temporary AWS Account already pre-configured with Amazon SageMaker Notebook Intances. Follow the steps in this section to login to your AWS Account and download the workshop material.

### 1. To get started navigate to - https://dashboard.eventengine.run/login 

![](./img/setup2.png)

Click on Accept Terms & Login

### 2. Click on Email One-Time OTP (Allow for up to 2 mins to receive the passcode)

![](./img/setup3.png)

### 3. Provide your email address

![](./img/setup4.png)

### 4. Enter your OTP code

![](./img/setup5.png)

### 5. Click on AWS Console

![](./img/setup6.png)

### 6. Click on Open AWS Console, remember to only use 'us-west-2' unless otherwise directed by event operator

![](./img/setup7.png)

### 7. In the AWS Console search for SageMaker and click on the Amazon SageMaker in the Services

![](./img/setup8.png)

### 8. Click on Amazon SageMaker Notebook -> Notebook Instances and then click on Open JupyterLab

![](./img/setup9.png)

### 9. You should now have Amazon SageMaker Notebook Jupyterlab interface open on your browser

![](./img/setup10.png)

### 10. Open a new terminal window

![](./img/setup11.png)

### 11. Clone the workshop content

In the terminal paste the following commands to clone the workshop content repo:

```
 cd SageMaker
 git clone https://github.com/aws-samples/sagemaker-multimodal-workshop.git
```

![](./img/setup12.png)

### 12. Rejoin the presenter for a live walkthrough of the workshop


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

