package xnnpack_test

import (
	"testing"

	"github.com/mattn/go-tflite"
	"github.com/mattn/go-tflite/delegates/xnnpack"
)

func invokeXOR(t *testing.T, interpreter *tflite.Interpreter) {
	t.Helper()

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
		float32s := interpreter.GetInputTensor(0).Float32s()
		float32s[0], float32s[1] = test.input[0], test.input[1]
		if interpreter.Invoke() != tflite.OK {
			t.Fatal("invoke failed")
		}

		float32s = interpreter.GetOutputTensor(0).Float32s()
		got := int(float32s[0] + 0.5)

		if got != test.want {
			t.Fatalf("want %v but got %v", test.want, got)
		}
	}
}

func TestWeightsCache(t *testing.T) {
	model := tflite.NewModelFromFile("../../testdata/xor_model.tflite")
	if model == nil {
		t.Fatal("cannot load model")
	}
	defer model.Delete()

	cache := xnnpack.NewWeightsCache()
	if cache == nil {
		t.Fatal("cannot create weights cache")
	}
	defer cache.Delete()

	var interpreters []*tflite.Interpreter
	for i := 0; i < 2; i++ {
		options := tflite.NewInterpreterOptions()
		defer options.Delete()
		delegate := xnnpack.New(xnnpack.DelegateOptions{NumThreads: 1, WeightsCache: cache})
		if delegate == nil {
			t.Fatal("cannot create delegate")
		}
		defer delegate.Delete()
		options.AddDelegate(delegate)

		interpreter := tflite.NewInterpreter(model, options)
		if interpreter == nil {
			t.Fatal("cannot create interpreter")
		}
		defer interpreter.Delete()

		if interpreter.AllocateTensors() != tflite.OK {
			t.Fatal("allocate failed")
		}
		interpreters = append(interpreters, interpreter)
	}

	if !cache.FinalizeHard() {
		t.Fatal("cannot finalize weights cache")
	}

	for _, interpreter := range interpreters {
		invokeXOR(t, interpreter)
	}
}
