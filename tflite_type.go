package tflite

/*
#ifndef GO_TFLITE_H
#include "tflite.go.h"
#endif
*/
import "C"

func (t *Tensor) Int32s() []int32 {
	if t.Type() != Int32 {
		return nil
	}
	ptr := C.TFL_TensorData(t.t)
	if ptr != nil {
		num := t.NumDims()
		return (*((*[1<<31 - 1]int32)(ptr)))[:num]
	}
	return nil
}

func (t *Tensor) Float32s() []float32 {
	if t.Type() != Float32 {
		return nil
	}
	ptr := C.TFL_TensorData(t.t)
	if ptr != nil {
		num := t.NumDims()
		return (*((*[1<<31 - 1]float32)(ptr)))[:num]
	}
	return nil
}

func (t *Tensor) Uint8s() []uint8 {
	if t.Type() != UInt8 {
		return nil
	}
	ptr := C.TFL_TensorData(t.t)
	if ptr != nil {
		num := t.NumDims()
		return (*((*[1<<31 - 1]uint8)(ptr)))[:num]
	}
	return nil
}

func (t *Tensor) Int64s() []int64 {
	if t.Type() != Int64 {
		return nil
	}
	ptr := C.TFL_TensorData(t.t)
	if ptr != nil {
		num := t.NumDims()
		return (*((*[1<<31 - 1]int64)(ptr)))[:num]
	}
	return nil
}

func (t *Tensor) Int16s() []int16 {
	if t.Type() != Int64 {
		return nil
	}
	ptr := C.TFL_TensorData(t.t)
	if ptr != nil {
		num := t.NumDims()
		return (*((*[1<<31 - 1]int16)(ptr)))[:num]
	}
	return nil
}

func (t *Tensor) Int8s() []int8 {
	if t.Type() != Int64 {
		return nil
	}
	ptr := C.TFL_TensorData(t.t)
	if ptr != nil {
		num := t.NumDims()
		return (*((*[1<<31 - 1]int8)(ptr)))[:num]
	}
	return nil
}
