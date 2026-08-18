package tflite

import (
	"testing"
)

func TestXOR(t *testing.T) {
	model := NewModelFromFile("testdata/xor_model.tflite")
	if model == nil {
		t.Fatal("cannot load model")
	}
	defer model.Delete()

	options := NewInterpreterOptions()
	defer options.Delete()

	interpreter := NewInterpreter(model, options)
	defer interpreter.Delete()

	interpreter.AllocateTensors()

	tests := []struct {
		input []float32
		want  int
	}{
		{input: []float32{0, 0}, want: 0},
		{input: []float32{0, 1}, want: 1},
		{input: []float32{1, 0}, want: 1},
		{input: []float32{1, 1}, want: 0},
	}

	for _, test := range tests {
		input := interpreter.GetInputTensor(0)
		float32s := input.Float32s()
		float32s[0], float32s[1] = test.input[0], test.input[1]
		interpreter.Invoke()

		output := interpreter.GetOutputTensor(0)
		float32s = output.Float32s()
		got := int(float32s[0] + 0.5)

		if got != test.want {
			t.Fatalf("want %v but got %v", test.want, got)
		}
	}
}

func TestTensorIndices(t *testing.T) {
	model := NewModelFromFile("testdata/xor_model.tflite")
	if model == nil {
		t.Fatal("cannot load model")
	}
	defer model.Delete()

	interpreter := NewInterpreter(model, nil)
	defer interpreter.Delete()

	if interpreter.AllocateTensors() != OK {
		t.Fatal("allocate failed")
	}

	inputs := interpreter.InputTensorIndices()
	if len(inputs) != interpreter.GetInputTensorCount() {
		t.Fatalf("unexpected input indices: %v", inputs)
	}
	if inputs[0] != interpreter.GetInputTensorIndex(0) {
		t.Fatalf("input index mismatch: %v vs %v", inputs[0], interpreter.GetInputTensorIndex(0))
	}
	if got, want := interpreter.GetTensor(inputs[0]).Name(), interpreter.GetInputTensor(0).Name(); got != want {
		t.Fatalf("want %q but got %q", want, got)
	}

	outputs := interpreter.OutputTensorIndices()
	if len(outputs) != interpreter.GetOutputTensorCount() {
		t.Fatalf("unexpected output indices: %v", outputs)
	}
	if outputs[0] != interpreter.GetOutputTensorIndex(0) {
		t.Fatalf("output index mismatch: %v vs %v", outputs[0], interpreter.GetOutputTensorIndex(0))
	}
	if got, want := interpreter.GetTensor(outputs[0]).Name(), interpreter.GetOutputTensor(0).Name(); got != want {
		t.Fatalf("want %q but got %q", want, got)
	}
}
