# Introduction-to-the-Fully-Recurrent-Network


## Exercise 1: Numerical stability of the binary cross-entropy loss function

<img width="1577" alt="image" src="https://github.com/user-attachments/assets/26f7dae6-bb46-47f0-a43c-e5adb058474a">


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Exercise 2: Derivative of the loss

Calculate the derivative of the binary cross-entropy loss function $L(z, y)$ with respect to the logit $z$.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Exercise 3: Initializing the network

<img width="1610" alt="image" src="https://github.com/user-attachments/assets/8d05e51f-857e-40e3-aef4-baf41858ac71">


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Exercise 4: The forward pass
Implement the forward pass for the fully recurrent network for sequence classification (many-to-one mapping). To this end, write a function `forward` that takes a `model`, a sequence of input vectors `x`, and a label `y` as arguments. The inputs will be represented as a `numpy` array of shape `(T, D)`. It should execute the behavior of the fully recurrent network and evaluate the (numerically stabilized) binary cross-entropy loss at the end of the sequence and return the resulting loss value. Store the sequence of hidden activations $(a(t))_{t=1}^T$ and the logit $z(T)$ into `model.a` and `model.z`, respectively.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Exercise 5: The computational graph

Visualize the computational graph of the fully recurrent network unfolded in time. The graph should show the functional dependencies of the nodes $x(t), a(t), z(t), L(z(t), y(t))$ for $t \in \{1, 2, 3\}$. Use the package `networkx` in combination with `matplotlib` to draw a directed graph with labelled nodes and edges. If you need help take a look at [this guide](https://networkx.guide/visualization/basics/). Make sure to arrange the nodes in a meaningful way.
