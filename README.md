**Performance analysis of Group Delay MUSIC estimated through Deep Learning**

**ABSTRACT**

The spatial spectrum gives the spatial distribution of the source coming from all directions to the sink. Hence from spatial spectrum DoA can be estimated and hence it is also called DoA estimation.

One such algorithm for DoA estimation is Multiple Signal Classification also known as MUSIC. The MUSIC method is unable to resolve the closely spaced sources with minimal number of sensors. Hence a group delay function is computed from the phase spectrum of MUSIC algorithm. This function can be used to resolve the closely spaced sources. We introduce a deep learning framework for such a function. We design deep convolutional neural network which receives the Music Group Delay Spectrum at the output and Covariance Matrix of the DoA at the input, and it learns the complex relationship between them. We can show that it works faster than the original function as the predicted output comes faster.

Index Terms: Deep Learning, DoA estimation, MUSIC, Group-Delay.

**INTRODUCTION**

Direction of arrival (DoA) estimation is crucial in many fields including Radar, Sonar, communications etc. The method of Multiple signal Classification is used to estimate DoA for this purpose. In Music the magnitude spectrum is used in computing DoA. However, in Group Delay we use negative differential of the unwrapped phase spectrum of MUSIC for DoA estimation.

In this we paper we also discuss why we cannot use MUSIC for closely spaced sources. The examples of closely spaced sources can be taken as meeting room scenarios where the microphones are placed closely. When using the model-based approaches the performance of DoA estimation lies on the accuracy of the input data. To remove this drawback, we use learning-based approach where the mapping between the non linear input output pairs is done by the neural networks. The deep learning is able to learn and predict the complex relationships between input and output in data/signals and can hence achieve better performance. We use the multi-layer neural network architecture to resolve the two sources.

We assume that number of targets are small as the complexity generating of data set increases by increase in targets. We introduce a deep learning framework for Group Delay Music and hence the name DeepGDM. The neural network is fed with Covariance Matrix and output to the network is the Unwrapped Group Delay Music phase spectra.  We show that the proposed approach is faster, data-independent and more accurate than the conventional methods. 

**DoA Estimation using MUSIC**

In Music we use direct relationship between the Direction of Arrival of received signal and the corresponding steering vector and hence we estimate the direction of signals from received signals.

If the M number of signals are detected by N sensors (N>M), the MUSIC spectrum is defined as:			
			**PMUSIC(ᴓ) =   1 / (SH(ᴓ) Qn QHn S(ᴓ))**
			
Where S = [ s(ᴓ1) s(ᴓ2) s(ᴓ3) s(ᴓ4) s(ᴓ5) ……………. s(ᴓM)] is a N x M matrix of steering vector,
Qn is the N x (N-M) noise eigenvector matrix. 
The denominator tends to zero when ᴓ is signal DoA. Therefore, we have M peaks corresponding to M signals. However, when the speech sources are spaced close to each other, MUSIC magnitude spectrum fails to produce results with minimal number of sensors.

In the below figure the MUSIC fails to resolve the DoA placed at 54֯ and 60֯ with 5 sensors but it resolves the two sources when number of sensors are increased to 15. The plots are for Magnitude Spectrum.

![image](https://user-images.githubusercontent.com/46759171/130826048-c085244e-5930-4dbb-9ccc-e41e3b4ec58b.png)
![image](https://user-images.githubusercontent.com/46759171/130826142-f029f18f-3e19-41fe-b8e8-fddc259494b9.png)


              
             Figure: Music Magnitude Plot for 54֯ and 60֯ with 5 sensors(left) , 15 sensors (right).
	     
**DoA Estimation using Group Delay of the Music Spectrum:**

We propose the use of Group Delay function of the Music phase spectrum for resolving closely spaced sources without the increase in number of sensors. The phase information in the noise eigenvalues of the Music spectrum is used. Let Fn(ᴓ) denote the phase information.

				Fn(ᴓ)=arg(QHn S(ᴓ))
	Where arg(.) is the instantaneous phase.
	
However, computing the group delay of the music spectrum can result in sharp spectral changes even when the computed angle is not the DoA. Hence abrupt peaks are formed. Hence to supress these peaks products of Music spectral magnitude and the group delay of the Music is used so that spurious peaks are withheld and only actual peaks gets amplified.
The Group Delay can be defined as:
				PGDM(ᴓ) = dF’n(ᴓ) x PMUSIC (ᴓ)
					       dᴓ    

This example illustrates the property of Music Group Delay to resolve spatially contagious sources.

  ![image](https://user-images.githubusercontent.com/46759171/130827015-80a50a65-4a2f-48de-b18d-8f7a31e7f67e.png)
![image](https://user-images.githubusercontent.com/46759171/130827080-039309a0-7e6c-4779-bc58-2f35789eba5c.png)

                    Figure:  Group delay(left) and Music magnitude(right) plots for 50֯ and 55֯ DoAs. 
                                                        No. of sensors = 10, snr=20 db.

**DoA estimation via Deep Learning **


The proposed deep music network is fed with array covariance matrix and yields the Group Delay Music phase spectrum at the output. We first make the inputs and the labels, and then discuss the network architecture and the training.
**Input Data and Labels**
We use 10,000 samples of DoA and corresponding spectra for training. For the DoA of each sample we find out the Group Delay Music Phase spectrum.

As input for the deep learning network we find out the Covariance Matrix from the array steering vector of the DoA. We use the real, imaginary and angle part of the Covariance Matrix Ry as the input to the convolutional layer of the neural network.

Let X be M x M x 3 matrix then first channel of X for a single sample will be [[X]:,:,1] = Re{[Ry]} for that sample. Similarly [[X]:,:,2] = Img{[Ry]} and [[X]:,:,3] = Angle{[Ry]}. Where angle obtained is of imaginary and real parts of Ry. This is a better method for feature extraction as it helps in input-output mapping.

Using the same covariance matrix, we can find the Group Delay Music phase spectra and use it as label for the entire region of angular spectrum for that sample. Hence, we make input-outputs pairs for 10000 samples.

**Network Architecture and Training**
	
The deep neural network is composed of 17 layers including input and output layers. The nonlinear mapping function can be represented by R(M*M*3)R(L). Where M=no. of antenna elements and L= No. of angles in the region. The network can be represented by:

NN Architecture=f(17) (f(16)(……f(1)(X).))
The Fully connected layer is f(14) which maps the input(x) from layer 13 to the output(y) by using the weights W. The output y can be given by the inner product (W.T*x) summed over entire training set N. The f(i){2,5,8,11} represents the convolutional layers with 256 filters of  kernel size of 5x5 and 3x3 for the first two and second two layers respectively. The arithmetic operation of a single filter of a convolutional layer can be defined as  Y =∑▒〖<W,X>〗 . Where W is the weight of the convolutional kernel and X is the input to the filter.
f(i){i=3,6,9,12} represents normalization layers and f(i){i=4,7,10,13}  represents rectified linear units ReLU layers which can be represented by ReLU(x)=max(0,x). f(15) is a dropout layer with dropout=0.5. f(16) is a softmax layer with output =   exp(x). f(17) is a regression layer with size 
Lx1. L=range of angular spectrum.                          ∑_i▒〖exp⁡(x)〗
 
The proposed network is trained on Google Colab with GPU accelerator. We used Adam optimizer with the learning rate of 0.01 and beta_1=0.9, beta_2=0.999 with epsilon= 1e-08. We take a batch size of 32 samples and train for 100 epochs. The length of Validation dataset is = 20%. The angular range taken is [-90֯,90֯] with 360 angles. We take 400 snapshots for every sample in our simulation. 
                                                                  
                                        Figure: Training and Validation Loss	





	Predictions and Results.

   
	                                                                                      2.
   
                                             3.                                                                                          4. 
Figure 1. & 2. Prediction from Deep Group Delay Music and Group Delay Music at DoAs 50֯ and 55֯ respectively and SNR 20 dB. Figure 3. & 4. Predicted output of DeepGDM and Group Delay functions for DoAs -45֯ and 60֯ at snr 20dB. (6 sensors)                                                                                                                                    
 
Figure RMSE vs SNR
We see that DeepGDM is able to resolve spatially contagious sources with limited amount of information. The Deep learning framework nearly approximates the actual DoAs and once trained thus becomes independent of input data which might sometimes get corrupted due to presence of noise. This is the major advantage of using Deep Learning framework for direction estimation.
The proposed approach is also faster as the time taken to calculate predicted output from DeepGDM is 0.0324 seconds and 0.1305 seconds by using Group Delay function. By using convolutional layers there is a better feature extraction of hidden input data which could be attributed for the performance of the model.
The performance deteriorates at low snr due to noisy data as the resolving power of the model increases with increase with snr during training. Once the model is trained it becomes independent of the input data due to good generalisation of neural networks.
Conclusions 

We implemented a Deep Group Delay Music phase function which is able to resolve spatially contagious sources with less computational time and provides a good approximation of the group delay function thus making it independent of input data after training. It can work for multiple targets thus increasing its significance.


