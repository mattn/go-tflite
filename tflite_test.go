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

func TestVersion(t *testing.T) {
	if Version() == "" {
		t.Fatal("version is empty")
	}
	if SchemaVersion() <= 0 {
		t.Fatalf("unexpected schema version: %v", SchemaVersion())
	}
	t.Logf("version=%s schema=%d extension_apis=%s", Version(), SchemaVersion(), ExtensionApisVersion())
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
