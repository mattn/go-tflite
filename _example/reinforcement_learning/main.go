package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"

	"github.com/mattn/go-tflite"
)

const (
	boardSize  = 8
	planeCells = 8
)

const (
	cellUntried = 0
	cellHit     = 1
	cellMiss    = -1
)

// placePlane hides the 8-cell plane on the board the same way the official
// PlaneStrike example does: a random orientation, a 5-cell cross around the
// core, and a 3-cell tail.
func placePlane(rnd *rand.Rand) [boardSize][boardSize]bool {
	var hidden [boardSize][boardSize]bool
	var x, y int
	switch rnd.Intn(4) {
	case 0: // heading right
		x = rnd.Intn(boardSize-2) + 1
		y = rnd.Intn(boardSize-3) + 2
		hidden[x][y-2] = true
		hidden[x-1][y-2] = true
		hidden[x+1][y-2] = true
	case 1: // heading up
		x = rnd.Intn(boardSize-3) + 1
		y = rnd.Intn(boardSize-2) + 1
		hidden[x+2][y] = true
		hidden[x+2][y+1] = true
		hidden[x+2][y-1] = true
	case 2: // heading left
		x = rnd.Intn(boardSize-2) + 1
		y = rnd.Intn(boardSize-3) + 1
		hidden[x][y+2] = true
		hidden[x-1][y+2] = true
		hidden[x+1][y+2] = true
	default: // heading down
		x = rnd.Intn(boardSize-3) + 2
		y = rnd.Intn(boardSize-2) + 1
		hidden[x-2][y] = true
		hidden[x-2][y+1] = true
		hidden[x-2][y-1] = true
	}
	hidden[x][y] = true
	hidden[x+1][y] = true
	hidden[x-1][y] = true
	hidden[x][y+1] = true
	hidden[x][y-1] = true
	return hidden
}

func printBoard(board *[boardSize][boardSize]float32) {
	for x := 0; x < boardSize; x++ {
		for y := 0; y < boardSize; y++ {
			switch board[x][y] {
			case cellHit:
				fmt.Print(" *")
			case cellMiss:
				fmt.Print(" .")
			default:
				fmt.Print(" -")
			}
		}
		fmt.Println()
	}
}

func main() {
	var model_path string
	var seed int64
	flag.StringVar(&model_path, "model", "planestrike.tflite", "path to model file")
	flag.Int64Var(&seed, "seed", 0, "random seed for the plane placement (0 picks one)")
	flag.Parse()

	if seed == 0 {
		seed = rand.Int63()
	}
	rnd := rand.New(rand.NewSource(seed))

	model := tflite.NewModelFromFile(model_path)
	if model == nil {
		log.Fatal("cannot load model")
	}
	defer model.Delete()

	options := tflite.NewInterpreterOptions()
	options.SetNumThread(4)
	defer options.Delete()

	interpreter := tflite.NewInterpreter(model, options)
	if interpreter == nil {
		log.Fatal("cannot create interpreter")
	}
	defer interpreter.Delete()

	if interpreter.AllocateTensors() != tflite.OK {
		log.Fatal("allocate failed")
	}

	hidden := placePlane(rnd)
	var board [boardSize][boardSize]float32

	// The agent strikes the hidden plane until all 8 cells are hit. Each turn
	// the model sees the board (1=hit, -1=miss, 0=untried) and returns 64
	// logits; the best untried cell is the next strike.
	hits, moves := 0, 0
	for hits < planeCells {
		input := interpreter.GetInputTensor(0).Float32s()
		for x := 0; x < boardSize; x++ {
			for y := 0; y < boardSize; y++ {
				input[x*boardSize+y] = board[x][y]
			}
		}
		if interpreter.Invoke() != tflite.OK {
			log.Fatal("invoke failed")
		}
		logits := interpreter.GetOutputTensor(0).Float32s()

		best := -1
		for i, v := range logits {
			if board[i/boardSize][i%boardSize] != cellUntried {
				continue
			}
			if best < 0 || v > logits[best] {
				best = i
			}
		}
		if best < 0 {
			log.Fatal("no cell left to strike")
		}

		x, y := best/boardSize, best%boardSize
		moves++
		if hidden[x][y] {
			board[x][y] = cellHit
			hits++
			fmt.Printf("move %2d: strike (%d,%d) hit! (%d/%d)\n", moves, x, y, hits, planeCells)
		} else {
			board[x][y] = cellMiss
			fmt.Printf("move %2d: strike (%d,%d) miss\n", moves, x, y)
		}
	}

	fmt.Printf("\ndestroyed the plane in %d moves (seed=%d)\n", moves, seed)
	printBoard(&board)
}
