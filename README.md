**Performance analysis of Group Delay MUSIC estimated through Deep Learning**

**ABSTRACT**

The spatial spectrum gives the spatial distribution of the source coming from all directions to
the sink. Hence from spatial spectrum DoA can be estimated and hence it is also called DoA
estimation.

One such algorithm for DoA estimation is Multiple Signal Classification also known as MUSIC.
The MUSIC method is unable to resolve the closely spaced sources with minimal number of
sensors. Hence a group delay function is computed from the phase spectrum of MUSIC
algorithm. This function can be used to resolve the closely spaced sources. We introduce a
deep learning framework for such a function. We design deep convolutional neural network
which receives the Music Group Delay Spectrum at the output and Covariance Matrix of the
DoA at the input, and it learns the complex relationship between them. We can show that it
works faster than the original function as the predicted output comes faster.

Index Terms: Deep Learning, DoA estimation, MUSIC, Group-Delay.

**INTRODUCTION**

Direction of arrival (DoA) estimation is crucial in many fields including Radar, Sonar,
communications etc. The method of Multiple signal Classification is used to estimate DoA for
this purpose. In Music the magnitude spectrum is used in computing DoA. However, in Group
Delay we use negative differential of the unwrapped phase spectrum of MUSIC for DoA
estimation.

In this we paper we also discuss why we cannot use MUSIC for closely spaced sources. The
examples of closely spaced sources can be taken as meeting room scenarios where the
microphones are placed closely. When using the model-based approaches the performance
of DoA estimation lies on the accuracy of the input data. To remove this drawback, we use
learning-based approach where the mapping between the non linear input output pairs is
done by the neural networks. The deep learning is able to learn and predict the complex
relationships between input and output in data/signals and can hence achieve better
performance. We use the multi-layer neural network architecture to resolve the two sources.

We assume that number of targets are small as the complexity generating of data set
increases by increase in targets. We introduce a deep learning framework for Group Delay
Music and hence the name DeepGDM. The neural network is fed with Covariance Matrix and
output to the network is the Unwrapped Group Delay Music phase spectra. We show that 
the proposed approach is faster, data-independent and more accurate than the conventional
methods.

**DoA Estimation using MUSIC**

In Music we use direct relationship between the Direction of Arrival of received signal and the
corresponding steering vector and hence we estimate the direction of signals from received
signals.

If the M number of signals are detected by N sensors (N>M), the MUSIC spectrum is defined
as:

  PMUSIC(ᴓ) = **1/(SH(ᴓ) Qn QHn S(ᴓ))**
Where S = [ s(ᴓ1) s(ᴓ2) s(ᴓ3) s(ᴓ4) s(ᴓ5) ……………. s(ᴓM)] is a N x M matrix of steering vector,
Qn is the N x (N-M) noise eigenvector matrix. The denominator tends to zero when ᴓ is signal
DoA. Therefore, we have M peaks corresponding to M signals. However, when the speech
sources are spaced close to each other, MUSIC magnitude spectrum fails to produce results
with minimal number of sensors.
In the below figure the MUSIC fails to resolve the DoA placed at 54֯and 60֯with 5 sensors but
it resolves the two sources when number of sensors are increased to 15. The plots are for
Magnitude Spectrum.
