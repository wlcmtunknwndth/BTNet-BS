# Defense Speech

**Topic:** Pricing American Options with a Neural Network Based on a Binomial Tree  
**Student:** Artem Petrov  
**Supervisor:** Sergey Shorokhov

## The 1st slide

Good afternoon, dear members of the examination committee.

My name is Artem Petrov. Today I am going to present my final qualifying work. The topic of my work is "Pricing American Options with a Neural Network Based on a Binomial Tree".

The main idea is to use a neural-network representation of the binomial tree method for pricing American put options.

## The 2nd slide

First, I will explain what an American option is.

A put option gives the right to sell an asset at a fixed strike price. A European option can be exercised only at maturity. An American option can be exercised earlier, before maturity.

At each moment the holder decides whether to exercise the option now or continue holding it. Because of this, American option pricing is an optimal stopping problem.

## The 3rd slide

For many European options, we can use the Black-Scholes formula. But for an American put option there is no simple closed formula in the general case.

So we need numerical methods. One classical method is the Cox-Ross-Rubinstein binomial tree model, or CRR model.

The task of my final work is to implement models of this architecture, train the European model, and verify the American model for put option pricing.

## The 4th slide

This slide shows the idea of a binomial tree.

At each time step the price of the underlying asset can move up or down. The up movement has multiplier `u`, and the down movement has multiplier `d`.

At the final nodes we calculate the payoff. Then we move backwards through the tree.

For an American option, at each node we compare two values: the continuation value and the immediate exercise value.

## The 5th slide

Here are the main CRR formulas.

The time step is `dt = T / n`. The up and down multipliers are calculated from volatility. The risk-neutral probability `p` is used to compute the discounted expected future value.

For an American put option, the value at a node is the maximum of two numbers. The first number is the immediate exercise value, `K - S`. The second number is the continuation value from the next layer of the tree.

This maximum is the mathematical reason why American option pricing is connected with optimal stopping.

## The 6th slide

Now I introduce BTNet.

BTNet means Binomial Tree Neural Network. It is an architecture whose forward pass repeats the backward induction of the CRR tree.

The terminal payoff is the first layer. The discounted expectation is represented by a linear filter `W`. The backward steps are represented by a sequence of layers.

For the American option, early exercise is represented by a maxout operation. This operation chooses the maximum between continuation and exercise.

## The 7th slide

In the practical part, I implemented the package `btnn_bs` in Python and PyTorch.

It contains two main models. `BTNetEuropean` is used for European put options. `BTNetAmerican` is used for American put options and includes maxout layers.

I also implemented tools for verification and for computing sensitivities: Delta, Gamma, Vega, and Theta.

## The 8th slide

This slide shows the key feature of the BTNetAmerican architecture.

The number of values decreases after each layer, like in a binomial tree. After all steps only one number remains: the current option price.

Each maxout layer has two branches. One branch calculates the continuation value, and the other branch calculates the immediate exercise value. Then maxout chooses the larger value in each node.

So this architecture is not just a black box. Its structure follows the financial algorithm.

## The 9th slide

Now I will discuss pricing accuracy.

For verification, I used the base parameters `S0 = 0.5`, `T = 1`, `r = 0.05`, `sigma = 0.25`, and tree depth `n = 9`. For American options, QuantLib CRR with 500 steps was used as a numerical reference.

With analytical CRR initialization, the mean absolute error was `2.84e-4`, RMSE was `4.06e-4`, and the maximum error was `1.10e-3`.

With transferred weights from the European model, the errors became worse in the base scenario. This shows that analytical CRR initialization is more reliable.

## The 10th slide

I also studied two important effects.

First, I tested transfer of weights from `BTNetEuropean` to `BTNetAmerican`. The result is not stable. In high volatility the error becomes much worse, so weight transfer should be treated only as a heuristic.

Second, I computed sensitivities using automatic differentiation. The important limitation is Gamma. It is zero almost everywhere because ReLU and maxout are piecewise-linear operations.

This is an important negative result. The current architecture is useful for reproducible pricing experiments, but it is not suitable as a full risk-management model if reliable Gamma is required.

## The 11th slide

To conclude, I will summarize what was done in the work.

I implemented the package `btnn_bs`. I built `BTNetEuropean` and `BTNetAmerican`. I trained the European model and verified the American model. I checked prices against Black-Scholes and QuantLib CRR with 500 steps. I also performed the weight transfer experiment.

The main conclusions are the following.

Analytical CRR initialization gives accurate and interpretable prices. Weight transfer is unstable. The Gamma problem shows that the architecture should be improved with smooth operations for full risk management.

Thank you for your attention. I am ready to answer your questions.


---

## Approval

Approved by the English language tutor:

Full name: ________________________________________________

Signature: ________________________________________________

Date: "____" ____________________ 2026
