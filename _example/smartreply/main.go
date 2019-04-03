package main

import (
	"bufio"
	"flag"
	"fmt"
	_ "image/jpeg"
	"log"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/mattn/go-tflite"
	customOps "github.com/mattn/go-tflite/_example/smartreply/ops"
	builtinOps "github.com/mattn/go-tflite/ops"
)

type result struct {
	msg        string
	confidence float32
}

func split(s string) []string {
	s = regexp.MustCompile(`([?.!,])+`).ReplaceAllString(s, "$1")
	s = regexp.MustCompile(`([?.!,])+\s+`).ReplaceAllString(s, "$1\t")
	s = regexp.MustCompile(`[ ]+`).ReplaceAllString(s, " ")
	s = regexp.MustCompile(`\t+$`).ReplaceAllString(s, " ")
	return strings.Split(s, "\t")
}

func smartreply(interpreter *tflite.Interpreter, s string) []result {
	var dbuf = new(tflite.DynamicBuffer)
	words := split(s)
	for _, word := range words {
		dbuf.AddString(word)
	}
	dbuf.WriteToTensorAsVector(interpreter.GetInputTensor(0))

	var status tflite.Status

	status = interpreter.AllocateTensors()
	if status != tflite.OK {
		log.Fatal("allocate failed")
	}

	status = interpreter.Invoke()
	if status != tflite.OK {
		log.Fatal("invoke failed")
	}

	messages := interpreter.GetOutputTensor(0)
	confidence := interpreter.GetOutputTensor(1)
	count := confidence.Dim(0)

	results := []result{}
	for i := 0; i < count; i++ {
		msg := messages.GetString(i)
		if msg != "" {
			results = append(results, result{
				msg:        msg,
				confidence: confidence.Float32s()[i],
			})
		}
	}
	return results
}

func main() {
	var model_path string
	flag.StringVar(&model_path, "model", "smartreply.tflite", "path to model file")
	flag.Parse()

	model := tflite.NewModelFromFile(model_path)
	if model == nil {
		log.Fatal("cannot load model")
	}
	defer model.Delete()

	options := tflite.NewInterpreterOptions()
	options.SetNumThread(4)
	options.ExpAddBuiltinOp(tflite.BuiltinOperator_SKIP_GRAM, builtinOps.Register_SKIP_GRAM(), 1, 1)
	options.ExpAddBuiltinOp(tflite.BuiltinOperator_LSH_PROJECTION, builtinOps.Register_LSH_PROJECTION(), 1, 1)
	options.ExpAddBuiltinOp(tflite.BuiltinOperator_HASHTABLE_LOOKUP, builtinOps.Register_HASHTABLE_LOOKUP(), 1, 1)

	options.ExpAddCustomOp("Normalize", customOps.Register_NORMALIZE(), 1, 1)
	options.ExpAddCustomOp("ExtractFeatures", customOps.Register_EXTRACT_FEATURES(), 1, 1)
	options.ExpAddCustomOp("Predict", customOps.Register_PREDICT(), 1, 1)
	defer options.Delete()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		interpreter := tflite.NewInterpreter(model, options)
		if interpreter == nil {
			log.Fatal("cannot create interpreter")
		}

		results := smartreply(interpreter, scanner.Text())
		if len(results) == 0 {
			interpreter.Delete()
			continue
		}
		sort.Slice(results, func(i, j int) bool {
			return results[i].confidence > results[j].confidence
		})
		fmt.Println(results[0].msg)

		interpreter.Delete()
	}
	if scanner.Err() != nil {
		log.Fatal(scanner.Err())
	}
}
