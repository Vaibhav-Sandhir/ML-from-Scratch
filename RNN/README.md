## Sin Wave Regression

### Dataset
1.) Sin Wave of size $S$ \
2.) Time Steps of length $T$ \
3.) $X$ tensor of dimension $(S - T \times T \times 1)$ \
4.) $Y$ tensor of dimension $(S - T \times 1)$ 

### Neural Network Architecture
![](architecture.png)

### Hyperparameters
1.) $\alpha = 0.01 \space (Learning \space Rate)$ \
2.) $h = 50 \space (Hidden \space Units)$ \
3.) $epochs = 50$ 

### Parameters
1.) $W_x$ tensor of dimension $(h \times 1)$ \
2.) $W_h$ tensor of dimension $(h \times h)$ \
3.) $W_y$ tensor of dimension $(h \times 1)$ \
4.) $B$ tensor of dimension $(h \times 1)$ \
5.) $B_y$ tensor of dimension $(1 \times 1)$

### Forward Propagation
1.) $X^{(i)<t>}$ refers to $i^{th}$ training example at time step $t$ \
2.) $Z^{<t>} = W_a \times A^{<t - 1>} + W_x \times X^{(i)<t>} + B$ \
3.) $A^{<t>} = \tanh{Z^{<t>}}$
4.) $\hat{Y} = W_y^{T} \times A^{<T>} + B_y$

### Loss Function

$MSE:  $
