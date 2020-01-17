package main

//#include <stdlib.h>
import "C"
import (
	"bufio"
	"log"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unsafe"
	"encoding/json"

	"github.com/mattn/go-tflite"
	gopointer "github.com/mattn/go-pointer"
)

// Classifier ...
type Classifier struct {
	dictionary  map[string]int
	labels      []string
	interpreter *tflite.Interpreter
}

const (
	START   = "<START>"
	PAD     = "<PAD>"
	UNKNOWN = "<UNKNOWN>"
)

const (
	SENTENCE_LEN = 256
)

//Build ... Builder method of Application and return the index
//export Build
func Build() unsafe.Pointer {
	dic, err := loadDictionary("vocab.txt")
	if err != nil {
		log.Fatal(err)
	}

	labels, err := loadLabels("labels.txt")
	if err != nil {
		log.Fatal(err)
	}

	model := tflite.NewModelFromFile("text_classification.tflite")
	if model == nil {
		log.Fatal("cannot load model")
	}

	interpreter := tflite.NewInterpreter(model, nil)

	classifier := Classifier{dictionary: dic, labels: labels, interpreter: interpreter}

	p := gopointer.Save(classifier)

	return p
}

//Classify ... Classify function
//export Classify
func Classify(appPointer unsafe.Pointer, word *C.char) *C.char {
	goWord := C.GoString(word)

	classifier := gopointer.Restore(appPointer)
	if classifier != nil {
		c := classifier.(*Classifier)
		result := c.classify(goWord)
		return C.CString(result)
	}
	return C.CString("Error Occurred")
}

//Close ... Close function
//export Close
func Close(appPointer unsafe.Pointer) {
	c := gopointer.Restore(appPointer).(*Classifier)
	if c != nil {
		defer c.interpreter.Delete()
	}
	defer gopointer.Unref(appPointer)
}

func loadDictionary(fname string) (map[string]int, error) {
	f, err := os.Open("vocab.txt")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	dic := make(map[string]int)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.Split(scanner.Text(), " ")
		if len(line) < 2 {
			continue
		}
		n, err := strconv.Atoi(line[1])
		if err != nil {
			continue
		}
		dic[line[0]] = n
	}
	return dic, nil
}

func loadLabels(fname string) ([]string, error) {
	f, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var labels []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	return labels, nil
}

func(c *Classifier) classify(word string) string{
	re := regexp.MustCompile(" |\\,|\\.|\\!|\\?|\n")
	tokens := re.Split(strings.TrimSpace(word), -1)
		index := 0
		tmp := make([]float32, SENTENCE_LEN)
		if n, ok := c.dictionary[START]; ok {
			tmp[index] = float32(n)
			index++
		}
		for _, word := range tokens {
			if index >= SENTENCE_LEN {
				break
			}

			if v, ok := c.dictionary[word]; ok {
				tmp[index] = float32(v)
			} else {
				tmp[index] = float32(c.dictionary[UNKNOWN])
			}
			index++
		}

		for i := index; i < SENTENCE_LEN; i++ {
			tmp[i] = float32(c.dictionary[PAD])
		}

		c.interpreter.AllocateTensors()

		copy(c.interpreter.GetInputTensor(0).Float32s(), tmp)

		c.interpreter.Invoke()

		type rank struct {
			label string
			poll  float32
		}
		ranks := []rank{}
		for i, v := range c.interpreter.GetOutputTensor(0).Float32s() {
			ranks = append(ranks, rank{
				label: c.labels[i],
				poll:  v,
			})
		}
		sort.Slice(ranks, func(i, j int) bool {
			return ranks[i].poll < ranks[j].poll
		})

		strResponse, _ := json.Marshal(ranks)
		return string(strResponse)
}

func main() {
}