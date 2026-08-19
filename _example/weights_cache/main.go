package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/mattn/go-tflite"
	"github.com/mattn/go-tflite/delegates/xnnpack"
)

// rss returns the resident set size of the process in bytes, or 0 when
// /proc is not available.
func rss() int64 {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return 0
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) >= 2 && fields[0] == "VmRSS:" {
			kb, err := strconv.ParseInt(fields[1], 10, 64)
			if err != nil {
				return 0
			}
			return kb * 1024
		}
	}
	return 0
}

func main() {
	var model_path string
	var n, cache_size int
	var share bool
	flag.StringVar(&model_path, "model", "../label_image/mobilenet_quant_v1_224.tflite", "path to model file")
	flag.IntVar(&n, "n", 4, "number of interpreters")
	flag.BoolVar(&share, "share", true, "share packed weights between the interpreters")
	flag.IntVar(&cache_size, "cache-size", 64*1024*1024, "size of the weights cache in bytes")
	flag.Parse()

	model := tflite.NewModelFromFile(model_path)
	if model == nil {
		log.Fatal("cannot load model")
	}
	defer model.Delete()

	// The cache must be created large enough for all packed weights up
	// front: when it grows while interpreters are being created, pointers
	// held by the delegates created so far become invalid and inference
	// crashes.
	var cache *xnnpack.WeightsCache
	if share {
		cache = xnnpack.NewWeightsCacheWithSize(cache_size)
		if cache == nil {
			log.Fatal("cannot create weights cache")
		}
		defer cache.Delete()
	}

	before := rss()
	start := time.Now()

	interpreters := make([]*tflite.Interpreter, n)
	for i := range interpreters {
		options := tflite.NewInterpreterOptions()
		defer options.Delete()
		delegate := xnnpack.New(xnnpack.DelegateOptions{NumThreads: 1, WeightsCache: cache})
		if delegate == nil {
			log.Fatal("cannot create delegate")
		}
		defer delegate.Delete()
		options.AddDelegate(delegate)

		interpreters[i] = tflite.NewInterpreter(model, options)
		if interpreters[i] == nil {
			log.Fatal("cannot create interpreter")
		}
		defer interpreters[i].Delete()

		if interpreters[i].AllocateTensors() != tflite.OK {
			log.Fatal("allocate failed")
		}
	}

	if cache != nil {
		if !cache.FinalizeHard() {
			log.Fatal("cannot finalize weights cache")
		}
	}

	fmt.Printf("created %d interpreters in %v (share=%v)\n", n, time.Since(start), share)
	if before > 0 {
		fmt.Printf("resident set size: %.1f MB -> %.1f MB\n",
			float64(before)/1024/1024, float64(rss())/1024/1024)
	}

	// Run all interpreters in parallel to show they work concurrently.
	start = time.Now()
	var wg sync.WaitGroup
	for _, interpreter := range interpreters {
		wg.Add(1)
		go func(interpreter *tflite.Interpreter) {
			defer wg.Done()
			if interpreter.Invoke() != tflite.OK {
				log.Println("invoke failed")
			}
		}(interpreter)
	}
	wg.Wait()
	fmt.Printf("ran %d invocations in parallel in %v\n", n, time.Since(start))
}
