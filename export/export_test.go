package export_test

import (
	"math"
	"testing"

	"github.com/mattn/go-tflite"
	"github.com/mattn/go-tflite/export"
)

func run(t *testing.T, buf []byte, input []float32, want int) []float32 {
	t.Helper()
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
	copy(interpreter.GetInputTensor(0).Float32s(), input)
	if interpreter.Invoke() != tflite.OK {
		t.Fatal("invoke failed")
	}
	got := interpreter.GetOutputTensor(0).Float32s()
	if len(got) != want {
		t.Fatalf("want %d outputs but got %d", want, len(got))
	}
	return got
}

func TestFullyConnected(t *testing.T) {
	m := export.NewModel()
	x := m.Input("x", []int{1, 3})
	// filter is [out, in]: row k holds the weights of output k.
	f := m.Constant("w", []int{2, 3}, []float32{
		1, 2, 3,
		-1, 0.5, 0,
	})
	b := m.Constant("b", []int{2}, []float32{0.5, -0.5})
	m.Output(m.FullyConnected(x, f, b, export.ActNone))
	buf, err := m.Bytes()
	if err != nil {
		t.Fatal(err)
	}

	got := run(t, buf, []float32{1, 2, 3}, 2)
	want := []float32{1*1 + 2*2 + 3*3 + 0.5, -1 + 0.5*2 - 0.5}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("output %d: want %v but got %v", i, want[i], got[i])
		}
	}
}

func TestActivationsAndSoftmax(t *testing.T) {
	m := export.NewModel()
	x := m.Input("x", []int{1, 2})
	f := m.Constant("w", []int{2, 2}, []float32{1, 0, 0, 1})
	b := m.Constant("b", []int{2}, []float32{0, 0})
	h := m.Tanh(m.FullyConnected(x, f, b, export.ActNone))
	m.Output(m.Softmax(h, 1.0))
	buf, err := m.Bytes()
	if err != nil {
		t.Fatal(err)
	}

	got := run(t, buf, []float32{1, -1}, 2)
	h0, h1 := math.Tanh(1), math.Tanh(-1)
	e0, e1 := math.Exp(h0), math.Exp(h1)
	want := []float32{float32(e0 / (e0 + e1)), float32(e1 / (e0 + e1))}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("output %d: want %v but got %v", i, want[i], got[i])
		}
	}
	if math.Abs(float64(got[0]+got[1]-1)) > 1e-6 {
		t.Fatalf("softmax outputs do not sum to 1: %v", got)
	}
}

func TestErrors(t *testing.T) {
	m := export.NewModel()
	if _, err := m.Bytes(); err == nil {
		t.Fatal("expected an error for a model without outputs")
	}

	m = export.NewModel()
	m.Output(m.Constant("w", []int{2, 3}, []float32{1}))
	if _, err := m.Bytes(); err == nil {
		t.Fatal("expected an error for a bad constant")
	}
}
