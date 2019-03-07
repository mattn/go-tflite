package tflite

import "C"
import (
	"unsafe"

	"github.com/mattn/go-pointer"
)

type callbackInfo struct {
	user_data interface{}
	f         func(msg string, user_data interface{})
}

//export _go_error_reporter
func _go_error_reporter(user_data unsafe.Pointer, msg *C.char) {
	cb := pointer.Restore(user_data).(*callbackInfo)
	cb.f(C.GoString(msg), cb.user_data)
}
