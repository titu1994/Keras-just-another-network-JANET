# Keras - Chrono LSTM and Just Another Recurrent Neural Network

Keras implementation of the paper [The unreasonable effectiveness of the forget gate](https://arxiv.org/abs/1804.04849).

This model utilizes just 2 gates - forget (f) and context (c) gates out of the 4 gates in a regular LSTM RNN, and with the utilization of `Chrono Initialization` from the paper [Can Recurrent Neural Networks Warp Time?](https://openreview.net/pdf?id=SJcKhk-Ab) to acheive better performance than regular LSTMs while using fewer parameters and less complicated gating structure.

# Usage
Simply import the `janet.py` file into your repo and use the `JANET` layer. 

It is **not** adviseable to use the `JANETCell` directly wrapped around a `RNN` layer, as this will not allow the `max timesteps` calculation that is needed for proper training using the `Chrono Initializer` for the forget gate.

The `chrono_lstm.py` script contains the `ChronoLSTM` model, as it requires minimal modifications to the original `LSTM` layer to use the `ChronoInitializer` for the forget and input gates.

Same restrictions to usage as the `JANET` layer, use the `ChronoLSTM` layer directly instead of the `ChronoLSTMCell` wrapped around a `RNN` layer.

```python
from janet import JANET
from chrono_lstm import ChronoLSTM

...
```

# Experiments
## Addition Task

The `JANET` model perperly gets learns the addition task for T = 100 in approximately 8 epochs, starting from the 5th epoch the loss goes down. This is slower than the paper, where the loss starts dropping rapidly in the 4th epoch and reaches a low enough value by its 6th epoch.

For T = 500 and T = 750, `JANET` loss starts dropping after epoch 12, and goes down steadily and reaches its low enough value around epoch 18. This corresponds to roughly 1200 - 1800 steps, much more than the 900~ steps needed by the paper.

### Notes
Need to study where the difference lies - either in the `ChronoInitializer`, or the initializations of the kernel/recurrent kernel. I used `glorot_uniform` for both since they match what the paper discussed, and found `orthogonal` for the recurrent kernel to provide slightly faster convergence, but still not approaching the paper.

## Sequential MNIST
Perhaps due to its slower convergence, the `JANET` model reaches a max test accuracy of just `98.39` after 100 epochs, far lower than the second standard deviation of the 10 fold mean-std performance in the paper. Will have to wait for their implementation to check what is the difference.
