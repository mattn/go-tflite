# export

Build TensorFlow Lite flatbuffers from models trained in Go, and run them
with go-tflite. Pure Go — the writer does not need the TensorFlow Lite C
library.

```go
m := export.NewModel()
x := m.Input("x", []int{1, 4})
h := m.Tanh(m.FullyConnected(x, w1, b1, export.ActNone))
m.Output(m.Softmax(m.FullyConnected(h, w2, b2, export.ActNone), 1.0))
buf, err := m.Bytes()
```

`export/gorgonia` exports a trained [gorgonia](https://gorgonia.org)
expression graph directly:

```go
buf, err := gorgonia.Export(y, x) // y: output node, x: input node
```

The supported subset is the inference part of a multilayer perceptron:
Mul, Add (as a bias), Tanh, Sigmoid and SoftMax. Weights are baked into the
model as float32 constants.
