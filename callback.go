package tflite

import "C"
import "unsafe"

//export _go_error_reporter
func _go_error_reporter(user_data unsafe.Pointer, msg unsafe.Pointer) {
	println(msg, user_data)
}
