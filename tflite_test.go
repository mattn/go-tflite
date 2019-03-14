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
