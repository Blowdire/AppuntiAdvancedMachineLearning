# Domande

1. In a neural network non linearity causes the most interesting loss function become non convex: True, the non linearity makes the function non convex, it has a lot of minima.

2. The loss function produces a numerical score that also depends on the set of parameters: True

3. Regularization functions are added to the loss functions to reduce their training error: False, because regularization is used to improve the generalization of the network, to reduce the error on the test set.

4. The gradient can be estimated through an iterative procedure that uses at each interation only a sample of training examples: True, this tecnique is used to estimate an approximation of the gradient in order to reduce computation time.

5. Multilayer networks need to specify the kernel function? False, they don't need the kernel function because they automatically learn the function.

6. Kernel functions first maps features in a different space and then evaluate the inner product? False, kernel functions used in machine learning avoid this computation.

7. In general kernel machines soffer high computational cost of training when the dataset is large? True, because thei complexity is linear, and the kernel function require stron computational power in order to be processed. You can use kernel trick to speedup the processing. They use linear combination of training examples. SVM can speedup this process because it considers only training examples close to the boundary regions.

8. To apply an iterative numerical optimization procedure for learning the weight of a FFN: The cost function may be a function that we cannot evaluate analytically and it is enough to have some way of approximating the gradient.

9. For training a multilayer NN: We can train one layer at the time using the error made in reproducing its own input.

10. It is always possible to train a neural network by solving a system of equations? Ambigua: True because you can effectively can do it, False because it is computational expensive and in practice you can't do it with complex networks.

11. A SVM can be trained by solving a system of equations while neural network can be always trained by using convex optimization? Partially true: You can train an SVM by solving a system of equations but you can't always train a neural network with simple comvex optimization.

12. True sentences:

    - The gradient descend may not converge if the learning rate is too big
    - The gradient descent algorithm can be applied to solve macimization.
    - The gradient descent may converge to a local optimum

13. The Sigmoid function: Saturates for large argument values, has sensitive gradient when z is close to zero.

14. Relu is proposed to speed up the learning convergence? True, maybe, because the derivative is very easy to compute and so the gradient is very easy to compute
15. Advantages of relu functions are: Are computationally simple, Reduced likelihood of the gradient to vanish, the gradient is costant for z>0.
16. Leaky relus: tend to blow up activation when z is large.
17. Maxout: can approximate any convex function, has as special cases Relu and Leaky relu, Does not have the problem of saturation, can approximate any convex function.
18. Weights in a network must be initialized: randomly
19. Sentences that are true: Simpler models generalize better, multiple hypothesis(ensemble) models generalize better
20. Regularrizing estimators: Reduce the gap between training error and validation error, can reduce the complexity of large models(Some methods pushes the weights to zero), introduce bias
21. False sentences: If the weight in the penalization term is too high it may imply overfitting, regularization of bias parameters can introduce overfitting.
22. With parameter norm penalties: The newtork is more stable(for example with explicit constraints), the learning process is encouraged towards small weights( for example with L2 norm)
23. Consider Norm penalizations: L2 results in more sparse weights than L1(false), the addition of the L2 term modifies the learning rule by shrinking the weight vector by a costant factor on each parameter update, squared weights penalize large values more, sum of absolute weights penalizes small weights more, L2 rescales the ewights along the aces defined by the eigenvectors of the Hessian matrix.
24. True sentences: Explicit constraints do not necessarily encourage the weights to aproach the origin, Explicit constraints by re-proj only have effect when the weights become more large and attempt to leave the constraint region, Regularizing operators can be seen as soft contraints of the learning optimization problem
25. Dataset augmentation:creates fake data and adds it to the training set, Injecting noise in the input to a neural network can also be seen as a form of data augmentation
