## Choice of baseline methods

- We show the results of a BNN alongside the results of a regular DNN fit via SGD as well as regularized DNN fit via SGD + Dropout.

- This allows us to inspect the regularization of weights that occurs via dropout and compare it against bayesian regularization

## Weight distributions in BNN

- The overall weight distribution looks nicely regularized
- Inspecting the weight distribution per layer shows that certain layers have tighter/wider variances
- In particular, the last layer has very tight weight variance
    - We do not have theoretical underpinnings for this, but intuitively this makes sense because the weights in the last layer affect the outputs of the BNN more directly than earlier layers.
    - If the variance in weight values was high, the outputs would vary more widely (less certain).
