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

func TestSignatureRunner(t *testing.T) {
	model := NewModelFromFile("testdata/xor_model_sig.tflite")
	if model == nil {
		t.Fatal("cannot load model")
	}
	defer model.Delete()

	options := NewInterpreterOptions()
	defer options.Delete()

	interpreter := NewInterpreter(model, options)
	defer interpreter.Delete()

	if interpreter.GetSignatureCount() != 1 {
		t.Fatalf("unexpected signature count: %v", interpreter.GetSignatureCount())
	}

	key := interpreter.GetSignatureKey(0)
	runner := interpreter.GetSignatureRunner(key)
	if runner == nil {
		t.Fatalf("cannot get signature runner for %q", key)
	}
	defer runner.Delete()

	if runner.AllocateTensors() != OK {
		t.Fatal("allocate failed")
	}
	if runner.GetInputCount() != 1 || runner.GetOutputCount() != 1 {
		t.Fatalf("unexpected signature inputs/outputs: %d/%d", runner.GetInputCount(), runner.GetOutputCount())
	}

	input := runner.GetInputTensor(runner.GetInputName(0))
	if input == nil {
		t.Fatal("cannot get input tensor")
	}

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
		float32s := input.Float32s()
		float32s[0], float32s[1] = test.input[0], test.input[1]
		if runner.Invoke() != OK {
			t.Fatal("invoke failed")
		}

		output := runner.GetOutputTensor(runner.GetOutputName(0))
		if output == nil {
			t.Fatal("cannot get output tensor")
		}
		got := int(output.Float32s()[0] + 0.5)

		if got != test.want {
			t.Fatalf("want %v but got %v", test.want, got)
		}
	}
}
