$$
MeanSquareError = 1/n \sum_{i = 1}^{n} (Yprediction_i - Y_i)^2
$$
To write in NumPy:

```
error = (1/n) * np.sum(np.square(predictions - labels))
```

