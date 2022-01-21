package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/mattn/go-tflite"
)

var doLowerCase = true

const (
	UNKNOWN_TOKEN           = "[UNK]" // For unknown words.
	MAX_INPUTCHARS_PER_WORD = 200
	MAX_SEQ_LEN             = 384
	PREDICT_ANS_NUM         = 5
	MAX_ANS_LEN             = 32
)

type Feature struct {
	inputIds       []int32
	inputMask      []int32
	segmentIds     []int32
	origTokens     []string
	tokenToOrigMap map[int]int
}

type Pos struct {
	start int
	end   int
	logi  float32
}

type QaAnswer struct {
	text string
	pos  Pos
}

func cleanText(text string) string {
	var buf bytes.Buffer
	for _, r := range text {
		// Skip the characters that cannot be used.
		if !utf8.ValidRune(r) || unicode.IsControl(r) {
			continue
		}
		if unicode.IsSpace(r) {
			buf.WriteRune(' ')
		} else {
			buf.WriteRune(r)
		}
	}
	return buf.String()
}

func whitespaceTokenize(text string) []string {
	return strings.Split(text, " ")
}

func fullTokenize(dic map[string]int32, text string) []string {
	splitTokens := []string{}
	for _, token := range basicTokenize(text) {
		for _, subToken := range wordpieceTokenize(dic, token) {
			splitTokens = append(splitTokens, subToken)
		}
	}
	return splitTokens

}

func basicTokenize(text string) []string {
	cleanedText := cleanText(text)
	origTokens := whitespaceTokenize(cleanedText)

	var buf bytes.Buffer
	for _, token := range origTokens {
		if doLowerCase {
			token = strings.ToLower(token)
		}
		list := runSplitOnPunc(token)
		for _, subToken := range list {
			buf.WriteString(subToken)
			buf.WriteRune(' ')
		}
	}
	return whitespaceTokenize(buf.String())
}

func runSplitOnPunc(text string) []string {
	tokens := []string{}
	startNewWord := true
	for _, r := range text {
		if unicode.IsPunct(r) {
			tokens = append(tokens, fmt.Sprintf("%c", r))
			startNewWord = true
		} else {
			if startNewWord {
				tokens = append(tokens, "")
				startNewWord = false
			}
			tokens[len(tokens)-1] = tokens[len(tokens)-1] + fmt.Sprintf("%c", r)
		}
	}

	return tokens
}

func wordpieceTokenize(dic map[string]int32, text string) []string {
	outputTokens := []string{}

	for _, token := range whitespaceTokenize(text) {
		if len(token) > MAX_INPUTCHARS_PER_WORD {
			outputTokens = append(outputTokens, UNKNOWN_TOKEN)
			continue
		}

		isBad := false // Mark if a word cannot be tokenized into known subwords.
		start := 0
		subTokens := []string{}

		for start < len(token) {
			curSubStr := ""

			end := len(token) // Longer substring matches first.
			for start < end {
				var subStr string
				if start == 0 {
					subStr = token[start:end]
				} else {
					subStr = "##" + token[start:end]
				}
				if _, ok := dic[subStr]; ok {
					curSubStr = subStr
					break
				}
				end--
			}

			// The word doesn't contain any known subwords.
			if curSubStr == "" {
				isBad = true
				break
			}

			// curSubStr is the longeset subword that can be found.
			subTokens = append(subTokens, curSubStr)

			// Proceed to tokenize the resident string.
			start = end
		}

		if isBad {
			outputTokens = append(outputTokens, UNKNOWN_TOKEN)
		} else {
			for _, token := range subTokens {
				outputTokens = append(outputTokens, token)
			}
		}
	}

	return outputTokens
}

func convertTokensToIds(dic map[string]int32, tokens []string) []int32 {
	ret := []int32{}
	for _, token := range tokens {
		ret = append(ret, dic[token])
	}
	return ret
}

func convert(dic map[string]int32, query string, content string) *Feature {
	queryTokens := fullTokenize(dic, query)
	if len(queryTokens) > 200 {
		queryTokens = queryTokens[:200]
	}

	origTokens := regexp.MustCompile(`\s+`).Split(strings.TrimSpace(content), -1)
	tokenToOrigIndex := []int{}
	allDocTokens := []string{}
	for i := 0; i < len(origTokens); i++ {
		token := origTokens[i]
		subTokens := fullTokenize(dic, token)
		for _, subToken := range subTokens {
			tokenToOrigIndex = append(tokenToOrigIndex, i)
			allDocTokens = append(allDocTokens, subToken)
		}
	}

	// -3 accounts for [CLS], [SEP] and [SEP].
	maxContextLen := MAX_SEQ_LEN - len(queryTokens) - 3
	if len(allDocTokens) > maxContextLen {
		allDocTokens = allDocTokens[0:maxContextLen]
	}

	tokens := []string{}
	segmentIds := []int32{}

	// Map token index to original index (in feature.origTokens).
	tokenToOrigMap := map[int]int{}

	// Start of generating the features.
	tokens = append(tokens, "[CLS]")
	segmentIds = append(segmentIds, 0)

	// For query input.
	for _, queryToken := range queryTokens {
		tokens = append(tokens, queryToken)
		segmentIds = append(segmentIds, 0)
	}

	// For Separation.
	tokens = append(tokens, "[SEP]")
	segmentIds = append(segmentIds, 0)

	// For Text Input.
	for i, docToken := range allDocTokens {
		tokens = append(tokens, docToken)
		segmentIds = append(segmentIds, 1)
		tokenToOrigMap[len(tokens)] = tokenToOrigIndex[i]
	}

	// For ending mark.
	tokens = append(tokens, "[SEP]")
	segmentIds = append(segmentIds, 1)

	inputIds := convertTokensToIds(dic, tokens)
	inputMask := make([]int32, len(inputIds))

	for len(inputIds) < MAX_SEQ_LEN {
		inputIds = append(inputIds, 0)
		inputMask = append(inputMask, 0)
		segmentIds = append(segmentIds, 0)
	}

	return &Feature{
		inputIds:       inputIds,
		inputMask:      inputMask,
		segmentIds:     segmentIds,
		origTokens:     origTokens,
		tokenToOrigMap: tokenToOrigMap,
	}
}

func loadDictionaryFile(filename string) map[string]int32 {
	dic := map[string]int32{}
	f, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	index := 0
	for scanner.Scan() {
		key := scanner.Text()
		dic[key] = int32(index)
		index++
	}
	return dic
}

func getBestIndex(logits []float32) []int {
	tmpList := []Pos{}
	for i := 0; i < MAX_SEQ_LEN; i++ {
		tmpList = append(tmpList, Pos{
			start: i,
			end:   i,
			logi:  logits[i],
		})
	}
	sort.Slice(tmpList, func(i, j int) bool {
		return tmpList[i].logi > tmpList[j].logi
	})

	indexes := make([]int, PREDICT_ANS_NUM)
	for i := 0; i < PREDICT_ANS_NUM; i++ {
		indexes[i] = tmpList[i].start
	}

	return indexes
}

func printResults(feature *Feature, startLogits, endLogits []float32) {
	startIndexes := getBestIndex(startLogits)
	endIndexes := getBestIndex(endLogits)
	origResults := []Pos{}
	for _, start := range startIndexes {
		for _, end := range endIndexes {
			if _, ok := feature.tokenToOrigMap[start+1]; !ok {
				continue
			}
			if _, ok := feature.tokenToOrigMap[end+1]; !ok {
				continue
			}
			if end < start {
				continue
			}
			if end-start > MAX_ANS_LEN {
				continue
			}
			origResults = append(origResults, Pos{
				start: start,
				end:   end,
				logi:  startLogits[start] + endLogits[end],
			})
		}
	}
	sort.Slice(origResults, func(i, j int) bool {
		return origResults[i].logi > origResults[j].logi
	})
	for _, result := range origResults {
		startIndex := feature.tokenToOrigMap[result.start+1]
		endIndex := feature.tokenToOrigMap[result.end+1]
		if startIndex > 0 {
			fmt.Println(strings.Join(feature.origTokens[startIndex:endIndex+1], " "))
			break
		}
	}
}

var content = `
Google LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, search engine, cloud computing, software, and hardware. It is considered one of the Big Four technology companies, alongside Amazon, Apple, and Facebook.

Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet's leading subsidiary and will continue to be the umbrella company for Alphabet's Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet.
`

func main() {
	var query, basefile, modelfile string
	flag.StringVar(&basefile, "f", "", "base content file")
	flag.StringVar(&query, "q", "", "query")
	flag.StringVar(&modelfile, "m", "mobilebert_float_20191023.tflite", "model")
	flag.Parse()

	if basefile != "" {
		b, err := ioutil.ReadFile(basefile)
		if err != nil {
			log.Println("cannot load base file")
			return
		}
		content = string(b)
	}

	model := tflite.NewModelFromFile(modelfile)
	if model == nil {
		log.Println("cannot load model")
		return
	}
	//defer model.Delete()

	interpreter := tflite.NewInterpreter(model, nil)
	defer interpreter.Delete()

	interpreter.AllocateTensors()

	dic := loadDictionaryFile("vocab.txt")

	if query != "" {
		feature := convert(dic, query, content)
		copy(interpreter.GetInputTensor(0).Int32s(), feature.inputIds)
		copy(interpreter.GetInputTensor(1).Int32s(), feature.inputMask)
		copy(interpreter.GetInputTensor(2).Int32s(), feature.segmentIds)
		interpreter.Invoke()
		startLogits := interpreter.GetOutputTensor(1).Float32s()
		endLogits := interpreter.GetOutputTensor(0).Float32s()
		printResults(feature, startLogits, endLogits)
	} else {
		r := bufio.NewReader(os.Stdin)
		for {
			fmt.Print("> ")
			b, _, err := r.ReadLine()
			if err != nil {
				break
			}
			feature := convert(dic, string(b), content)
			copy(interpreter.GetInputTensor(0).Int32s(), feature.inputIds)
			copy(interpreter.GetInputTensor(1).Int32s(), feature.inputMask)
			copy(interpreter.GetInputTensor(2).Int32s(), feature.segmentIds)
			interpreter.Invoke()
			startLogits := interpreter.GetOutputTensor(1).Float32s()
			endLogits := interpreter.GetOutputTensor(0).Float32s()
			printResults(feature, startLogits, endLogits)
		}
	}
}
