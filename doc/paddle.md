# Record

Record some precautions for using paddlepaddle to reproduce, the paddlepaddle version used is 2.1.3.

- The paddlepaddle cannot handle non-contiguous data slices.
- Creating a new tensor and replacing the values will affect the gradient update of those values.
- The ```paddle.diag``` function does not support gradient backpropagation.

