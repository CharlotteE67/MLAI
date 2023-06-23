# Machine Learning and Artificial Intelligence

## Assignment 3

**ID: 12232418** 		**Name: Jiang Yuchen**


According to the question, we need to use perceptron to learn an OR function. The training data is given as below.

<img src="MLAI-Assignment3.assets/image-20230319181357807.png" alt="image-20230319181357807" style="zoom:33%;" />

**First, we introduce the packages and prepare the data.**

<img src="MLAI-Assignment3.assets/image-20230319181446114.png" alt="image-20230319181446114" style="zoom:50%;" />

**Then, define the Perceptron class and write forward function and training process. It's hard to decide how to update the parameters since the activation function can't give a non-zero gradient. Thus, we just consider the gradient of other parts. We use Mean Square Error as loss function.**

<img src="MLAI-Assignment3.assets/image-20230319181517637.png" alt="image-20230319181517637" style="zoom:50%;" />

 **Last, train the perceptron with given four pieces of data. Max epoch and learning rate are set to 100 and 0.01 respectively. It converges at epoch 50.**

![image-20230319181859612](MLAI-Assignment3.assets/image-20230319181859612.png)

![image-20230319181911168](MLAI-Assignment3.assets/image-20230319181911168.png)

![image-20230319182042874](MLAI-Assignment3.assets/image-20230319182042874.png)