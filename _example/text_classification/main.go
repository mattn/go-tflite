package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/mattn/go-tflite"
)

const (
	START   = "<START>"
	PAD     = "<PAD>"
	UNKNOWN = "<UNKNOWN>"
)

const (
	SENTENCE_LEN = 256
)

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

func main() {
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
	defer model.Delete()

	interpreter := tflite.NewInterpreter(model, nil)
	defer interpreter.Delete()

	re := regexp.MustCompile(" |\\,|\\.|\\!|\\?|\n")

	r := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ")
		b, _, err := r.ReadLine()
		if err != nil {
			break
		}
		text := string(b)

		tokens := re.Split(strings.TrimSpace(text), -1)
		index := 0
		tmp := make([]float32, SENTENCE_LEN)
		if n, ok := dic[START]; ok {
			tmp[index] = float32(n)
			index++
		}
		for _, word := range tokens {
			if index >= SENTENCE_LEN {
				break
			}

			if v, ok := dic[word]; ok {
				tmp[index] = float32(v)
			} else {
				tmp[index] = float32(dic[UNKNOWN])
			}
			index++
		}

		for i := index; i < SENTENCE_LEN; i++ {
			tmp[i] = float32(dic[PAD])
		}

		interpreter.AllocateTensors()

		copy(interpreter.GetInputTensor(0).Float32s(), tmp)

		interpreter.Invoke()

		type rank struct {
			label string
			poll  float32
		}
		ranks := []rank{}
		for i, v := range interpreter.GetOutputTensor(0).Float32s() {
			ranks = append(ranks, rank{
				label: labels[i],
				poll:  v,
			})
		}
		sort.Slice(ranks, func(i, j int) bool {
			return ranks[i].poll < ranks[j].poll
		})
		for _, rank := range ranks {
			fmt.Printf("  %s: %v\n", rank.label, rank.poll)
		}
	}
}
