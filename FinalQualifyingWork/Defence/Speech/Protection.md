# Defense Speech

**Topic:** Pricing American Options with a Neural Network Based on a Binomial Tree  
**Student:** Artem Petrov  
**Supervisor:** Sergey Shorokhov

## The 1st slide

Good afternoon, dear members of the examination committee.

My name is Artem Petrov. Today I am going to present my final qualifying work. The topic of my work is "Pricing American Options with a Neural Network Based on a Binomial Tree".

The work is connected with option pricing, numerical methods, and neural networks. The main object of the work is option pricing methods. The subject is the BTNet architecture for pricing an American put option and for computing its sensitivities to market parameters.

## The 2nd slide

First, I would like to describe the problem.

Options are important financial instruments. They are used for hedging, speculation, and portfolio management. A put option gives the right to sell an asset at a fixed strike price.

An American put option is more difficult than a European option because it can be exercised before maturity. At each moment the holder has to choose between two actions: to exercise the option now or to continue holding it. Therefore, the pricing problem becomes an optimal stopping problem.

For an American put option there is no simple closed analytical formula in the general case. For this reason, numerical methods are used. In my work, QuantLib CRR with 500 steps is used as a practical numerical reference, not as an exact analytical solution.

## The 3rd slide

The goal of my work is to implement and numerically verify BTNet for pricing American put options and to analyze option Greeks using automatic differentiation.

To achieve this goal, I solved several tasks.

First, I studied the Black-Scholes model, the Cox-Ross-Rubinstein binomial model, and the BTNet equivalence.

Second, I implemented the software package `btnn_bs` in Python and PyTorch.

Third, I verified European and American put prices against analytical formulas and QuantLib.

Fourth, I computed Delta, Gamma, Vega, and Theta using automatic differentiation.

And finally, I studied the transfer of weights from the European BTNet model to the American BTNet model.

## The 4th slide

It is important to correctly define my own contribution.

I do not claim that I created a new BTNet theory. The theoretical basis of BTNet was proposed by Sergey Shorokhov. In that approach, the backward induction of a binomial tree is represented as a forward pass of a special neural network.

My contribution is practical and experimental. I implemented the package `btnn_bs`, applied BTNetAmerican to American put option pricing, performed numerical verification against QuantLib CRR, computed and analyzed Greeks through automatic differentiation, and performed the weight transfer experiment.

An important separate result of the work is the identification of the Gamma limitation in the piecewise-linear BTNet architecture.

## The 5th slide

Now I will explain why BTNet is useful in this work.

The classical CRR model is transparent and has a clear financial interpretation. The weights are connected with the risk-neutral measure. The early exercise feature of an American option is also naturally represented by the maximum between continuation and exercise.

An ordinary neural network can be fast after training, but its weights are usually difficult to interpret. BTNet is different. It is not just a black-box neural network. It represents the CRR backward induction as a neural network with interpretable layers.

This means that BTNet keeps the financial logic of the binomial tree and also allows us to use tools from neural network libraries, including automatic differentiation.

## The 6th slide

This slide shows the correspondence between the CRR model and BTNet.

The terminal payoff `max(K - S, 0)` is represented by the first layer. The discounted risk-neutral expectation is represented by a convolution-like layer with the filter `W`. The backward induction is represented by a sequence of layers.

For the American option, the most important part is early exercise. It is represented by the maxout layer. This layer compares the continuation value with the immediate exercise value and chooses the maximum. In other words, maxout is a neural-network form of the Bellman recursion for the optimal stopping problem.

## The 7th slide

This slide shows the architecture of BTNetAmerican.

The main idea is that each layer corresponds to one step of backward induction in the binomial tree. The network starts from terminal payoffs and then moves step by step to the initial node.

The American feature is implemented through maxout layers. At each node, the model chooses whether it is better to continue or to exercise. Because of this, the architecture has a direct financial interpretation.

## The 8th slide

Now I will describe the experimental setup.

The base experiment uses the following parameters: initial asset price `S0 = 0.5`, maturity `T = 1`, risk-free rate `r = 0.05`, volatility `sigma = 0.25`, and tree depth `n = 9`. Strike prices are taken on the grid from 0.25 to 0.75.

For the European put option, I used the Black-Scholes formula and QuantLib as references. For the American put option, I used QuantLib CRR with 500 steps as a numerical benchmark.

The quality metrics are MAE, RMSE, and maximum absolute error. For Greeks, I compared automatic differentiation with analytical formulas for the European option and with central finite differences for the American option.

## The 9th slide

The first main result is the pricing accuracy of the American put option.

With analytical CRR initialization, BTNetAmerican gives a mean absolute error of `2.84e-4`, RMSE of `4.06e-4`, and maximum absolute error of `1.10e-3` relative to QuantLib CRR with 500 steps.

These values show that the implementation is correct and that even the compact tree with `n = 9` gives a close approximation to a deeper numerical benchmark.

With transferred weights, the result in the base scenario becomes worse. The MAE increases to `4.38e-4`, the RMSE increases to `6.81e-4`, and the maximum error increases to `1.71e-3`.

## The 10th slide

The second main result is connected with the transfer of weights from the European model to the American model.

The initial hypothesis was that weights trained on European Black-Scholes prices could be useful for the American architecture. But the experiments show that this transfer is not stable.

At low volatility, the two initializations are almost the same. At `sigma = 0.60`, transferred weights slightly reduce the price error. But at high volatility, `sigma = 0.90`, the transferred weights become much worse.

Therefore, weight transfer should be treated as an unstable heuristic, not as a reliable initialization method for the American model.

## The 11th slide

This slide explains why the transfer of weights is unstable.

With analytical initialization, the filter `W` has a clear financial meaning. It represents discounted risk-neutral expectation. In the base scenario, the analytical weights are approximately `0.4847` and `0.5097`.

After transfer from the European model, the weights become approximately `0.4889` and `0.5063`. The numerical difference looks small, but it changes the continuation value in the American recursion.

For a European option, such a shift can help approximate the Black-Scholes price. But for an American option, it changes the comparison between continuation and immediate exercise. Also, for Vega and Theta, a fixed transferred filter loses part of the dependence on market parameters. This makes the transferred model less reliable for risk analysis.

## The 12th slide

The third important result concerns Greeks.

In the work, Delta, Gamma, Vega, and Theta were computed using automatic differentiation. For the American option, the results were checked by finite differences.

The main limitation is connected with Gamma. In the implemented BTNet architecture, ReLU and maxout operations are piecewise-linear. Therefore, the option price is also piecewise-linear as a function of the input parameters. Inside each linear region, the second derivative is zero. At breakpoints, the classical second derivative is not defined.

For this reason, Gamma computed by autograd is equal to zero almost everywhere. This is not a programming error. It is a mathematical consequence of the architecture.

This is an important negative result of the work. It means that the current BTNet architecture can be used for reproducible pricing and for analysis of some local sensitivities, but it should not be used as a full risk-management model when reliable Gamma and delta-gamma hedging are required.

## The 13th slide

To conclude, the goal of the work was achieved. I implemented and verified BTNet for American put option pricing and analyzed Greeks through automatic differentiation.

Analytical CRR initialization gave good pricing accuracy: MAE equals 2.84 times 10 to the minus 4 relative to QuantLib CRR with 500 steps. The transfer of weights from European to American BTNet was shown to be unstable and should be treated only as a heuristic.

The main limitation of the architecture is zero Gamma almost everywhere. Therefore, future work should include smooth approximations of ReLU and maxout, study of the tree depth, and extensions to more complex option models.

Thank you for your attention. I am ready to answer your questions.


---

## Approval

Approved by the English language tutor:

Full name: ________________________________________________

Signature: ________________________________________________

Date: "____" ____________________ 2026
