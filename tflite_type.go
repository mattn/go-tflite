package tflite

/*
#ifndef GO_TFLITE_H
#include "tflite.go.h"
#endif
*/
import "C"
import "errors"

var (
	// ErrTypeMismatch is type mismatch.
	ErrTypeMismatch = errors.New("type mismatch")
	// ErrBadTensor is bad tensor.
	ErrBadTensor = errors.New("bad tensor")
)

// SetInt32s sets int32s.
func (t *Tensor) SetInt32s(v []int32) error {
	if t.Type() != Int32 {
		return ErrTypeMismatch
	}
	ptr := C.TfLiteTensorData(t.t)
	if ptr == nil {
		return ErrBadTensor
	}
	n := t.ByteSize() / 4
	to := (*((*[1<<29 - 1]int32)(ptr)))[:n]
	copy(to, v)
	return nil
}

// Int32s returns int32s.
func (t *Tensor) Int32s() []int32 {
	if t.Type() != Int32 {
		return nil
	}
	ptr := C.TfLiteTensorData(t.t)
	if ptr == nil {
		return nil
	}
	n := t.ByteSize() / 4
	return (*((*[1<<29 - 1]int32)(ptr)))[:n]
}

// SetFloat32s sets float32s.
func (t *Tensor) SetFloat32s(v []float32) error {
	if t.Type() != Float32 {
		return ErrTypeMismatch
	}
	ptr := C.TfLiteTensorData(t.t)
	if ptr == nil {
		return ErrBadTensor
	}
	n := t.ByteSize() / 4
	to := (*((*[1<<29 - 1]float32)(ptr)))[:n]
	copy(to, v)
	return nil
}

// Float32s returns float32s.
func (t *Tensor) Float32s() []float32 {
	if t.Type() != Float32 {
		return nil
	}
	ptr := C.TfLiteTensorData(t.t)
	if ptr == nil {
		return nil
	}
	n := t.ByteSize() / 4
	return (*((*[1<<29 - 1]float32)(ptr)))[:n]
}

// Float32At returns float32 value located in the dimension.
func (t *Tensor) Float32At(at ...int) float32 {
	pos := 0
	for i := 0; i < t.NumDims(); i++ {
		pos = pos*t.Dim(i) + at[i]
	}
	return t.Float32s()[pos]
}

// SetUint8s sets uint8s.
func (t *Tensor) SetUint8s(v []uint8) error {
	if t.Type() != UInt8 {
		return ErrTypeMismatch
	}
	ptr := C.TfLiteTensorData(t.t)
	if ptr == nil {
		return ErrBadTensor
	}
	n := t.ByteSize()
	to := (*((*[1<<29 - 1]uint8)(ptr)))[:n]
	copy(to, v)
	return nil
}

// UInt8s returns uint8s.
func (t *Tensor) UInt8s() []uint8 {
	if t.Type() != UInt8 {
		return nil
	}
	ptr := C.TfLiteTensorData(t.t)
	if ptr == nil {
		return nil
	}
	n := t.ByteSize()
	return (*((*[1<<29 - 1]uint8)(ptr)))[:n]
}

// SetInt64s sets int64s.
func (t *Tensor) SetInt64s(v []int64) error {
	if t.Type() != Int64 {
		return ErrTypeMismatch
	}
	ptr := C.TfLiteTensorData(t.t)
	if ptr == nil {
		return ErrBadTensor
	}
	n := t.ByteSize() / 8
	to := (*((*[1<<28 - 1]int64)(ptr)))[:n]
	copy(to, v)
	return nil
}

// Int64s returns int64s.
func (t *Tensor) Int64s() []int64 {
	if t.Type() != Int64 {
		return nil
	}
	ptr := C.TfLiteTensorData(t.t)
	if ptr == nil {
		return nil
	}
	n := t.ByteSize() / 8
	return (*((*[1<<28 - 1]int64)(ptr)))[:n]
}

// SetInt16s sets int16s.
func (t *Tensor) SetInt16s(v []int16) error {
	if t.Type() != Int16 {
		return ErrTypeMismatch
	}
	ptr := C.TfLiteTensorData(t.t)
	if ptr == nil {
		return ErrBadTensor
	}
	n := t.ByteSize() / 2
	to := (*((*[1<<29 - 1]int16)(ptr)))[:n]
	copy(to, v)
	return nil
}

// Int16s returns int16s.
func (t *Tensor) Int16s() []int16 {
	if t.Type() != Int16 {
		return nil
	}
	ptr := C.TfLiteTensorData(t.t)
	if ptr == nil {
		return nil
	}
	n := t.ByteSize() / 2
	return (*((*[1<<29 - 1]int16)(ptr)))[:n]
}

// SetInt8s sets int8s.
func (t *Tensor) SetInt8s(v []int8) error {
	if t.Type() != Int8 {
		return ErrTypeMismatch
	}
	ptr := C.TfLiteTensorData(t.t)
	if ptr == nil {
		return ErrBadTensor
	}
	n := t.ByteSize()
	to := (*((*[1<<29 - 1]int8)(ptr)))[:n]
	copy(to, v)
	return nil
}

// Int8s returns int8s.
func (t *Tensor) Int8s() []int8 {
	if t.Type() != Int8 {
		return nil
	}
	ptr := C.TfLiteTensorData(t.t)
	if ptr == nil {
		return nil
	}
	n := t.ByteSize()
	return (*((*[1<<29 - 1]int8)(ptr)))[:n]
}

// String returns name of tensor.
func (t *Tensor) String() string {
	return t.Name()
}
