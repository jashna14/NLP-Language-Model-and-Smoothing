PROBLEM STATEMENT:
==================
Provided a training corpus corpus.txt. Use it to create an n-gram language model, where n can be provided as a parameter. Perform smoothing on the language model using:
+ Witten Bell Smoothing, and
+ Kneyser Ney Smoothing
  
### HOW TO RUN :
```python
			python3 language_model.py <value of n> <smoothing type> <path to input corpus>
```
+ where n can be between 1 and 3, and smoothing type can be k for Kneyser Ney or w for Witten Bell.

### Comparision:

+ Key Idea behind Kneser-Ney that we can take advantage of interpolation as a sort of backoff model.When the first term is near zero, the second term (the lower-order model) carries more weight. Inversely, when the higher-order model matches strongly, the second lower-order term has little weight.
+ Witten-Bell smoothing is an instance of the recursive interpolation method.The  n-th order smoothed model  are defined recursively as a linear interpolation between the nth order maximum likelyhood model and the (n-1)th order smooth model
+ It is observed that kneser ney gives a slightly higher probabilty output as compared to witten bell for all n-gram model.
+ Witten Bell Smoothing is more conservative when subtracting probability mass and gives good probability estimates.
+ Kneser Ney discounting augments absolute discounting with a more sophisticated way to handle the backoff distribution using linear interpolation and gives even better estimates than the Witten Bell Smoothing in most of the cases.


