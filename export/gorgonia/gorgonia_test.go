package gorgonia_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mattn/go-tflite"
	exportgorgonia "github.com/mattn/go-tflite/export/gorgonia"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func randTensor(r *rand.Rand, shape ...int) *tensor.Dense {
	n := 1
	for _, s := range shape {
		n *= s
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(r.NormFloat64())
	}
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(data))
}

func TestExportMLP(t *testing.T) {
	r := rand.New(rand.NewSource(1))

	g := G.NewGraph()
	x := G.NewMatrix(g, tensor.Float32, G.WithShape(1, 4), G.WithName("x"))
	w1 := G.NewMatrix(g, tensor.Float32, G.WithShape(4, 8), G.WithName("w1"), G.WithValue(randTensor(r, 4, 8)))
	b1 := G.NewMatrix(g, tensor.Float32, G.WithShape(1, 8), G.WithName("b1"), G.WithValue(randTensor(r, 1, 8)))
	w2 := G.NewMatrix(g, tensor.Float32, G.WithShape(8, 3), G.WithName("w2"), G.WithValue(randTensor(r, 8, 3)))
	b2 := G.NewMatrix(g, tensor.Float32, G.WithShape(1, 3), G.WithName("b2"), G.WithValue(randTensor(r, 1, 3)))

	h := G.Must(G.Tanh(G.Must(G.Add(G.Must(G.Mul(x, w1)), b1))))
	y := G.Must(G.SoftMax(G.Must(G.Add(G.Must(G.Mul(h, w2)), b2))))

	// Run the graph with gorgonia to get the expected output.
	in := randTensor(r, 1, 4)
	if err := G.Let(x, in); err != nil {
		t.Fatal(err)
	}
	machine := G.NewTapeMachine(g)
	defer machine.Close()
	if err := machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	want := y.Value().Data().([]float32)

	// Export and run the same computation with the TFLite interpreter.
	buf, err := exportgorgonia.Export(y, x)
	if err != nil {
		t.Fatal(err)
	}

	model := tflite.NewModel(buf)
	if model == nil {
		t.Fatal("cannot load the exported model")
	}
	defer model.Delete()
	interpreter := tflite.NewInterpreter(model, nil)
	if interpreter == nil {
		t.Fatal("cannot create interpreter")
	}
	defer interpreter.Delete()
	if interpreter.AllocateTensors() != tflite.OK {
		t.Fatal("allocate failed")
	}
	copy(interpreter.GetInputTensor(0).Float32s(), in.Data().([]float32))
	if interpreter.Invoke() != tflite.OK {
		t.Fatal("invoke failed")
	}
	got := interpreter.GetOutputTensor(0).Float32s()

	if len(got) != len(want) {
		t.Fatalf("want %d outputs but got %d", len(want), len(got))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("output %d: want %v but got %v", i, want[i], got[i])
		}
	}
}

func TestExportUnsupported(t *testing.T) {
	g := G.NewGraph()
	x := G.NewMatrix(g, tensor.Float32, G.WithShape(1, 2), G.WithName("x"))
	y := G.Must(G.Square(x))
	if _, err := exportgorgonia.Export(y, x); err == nil {
		t.Fatal("expected an error for an unsupported op")
	}

	// A weight leaf without a bound value must be reported.
	g = G.NewGraph()
	x = G.NewMatrix(g, tensor.Float32, G.WithShape(1, 2), G.WithName("x"))
	w := G.NewMatrix(g, tensor.Float32, G.WithShape(2, 2), G.WithName("w"))
	y = G.Must(G.Mul(x, w))
	if _, err := exportgorgonia.Export(y, x); err == nil {
		t.Fatal("expected an error for a weight without a value")
	}
}
