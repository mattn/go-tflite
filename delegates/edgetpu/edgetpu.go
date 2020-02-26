package edgetpu

// +build !windows

/*
#ifndef GO_EDGETPU_H
#include "edgetpu.go.h"
#include <edgetpu_c.h>
#endif
#cgo LDFLAGS: -ledgetpu

*/
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/mattn/go-tflite/delegates"
)

const (
	// The Device Types
	TypeApexPCI DeviceType = C.EDGETPU_APEX_PCI
	TypeApexUSB DeviceType = C.EDGETPU_APEX_USB
)

type DeviceType uint32

type Device struct {
	Type DeviceType
	Path string
}

// There are no options
type DelegateOptions struct {
}

// Delegate is the tflite delegate
type Delegate struct {
	d *C.TfLiteDelegate
}

func New(device Device) delegates.Delegater {
	var d *C.TfLiteDelegate
	d = C.edgetpu_create_delegate(uint32(device.Type), C.CString(device.Path), nil, 0)
	if d == nil {
		return nil
	}
	return &Delegate{
		d: d,
	}
}

// Delete the delegate
func (d *Delegate) Delete() {
	C.edgetpu_free_delegate(d.d)
}

// Return a pointer
func (d *Delegate) Ptr() unsafe.Pointer {
	return unsafe.Pointer(d.d)
}

// Version fetches the EdgeTPU runtime version information
func Version() (string, error) {
	version := C.edgetpu_version()
	if version == nil {
		return "", fmt.Errorf("could not get version")
	}
	return C.GoString(version), nil
}

// Verbosity sets the edgetpu verbosity
func Verbosity(v int) {
	C.edgetpu_verbosity(C.int(v))
}

// DeviceList fetches a list of devices
func DeviceList() ([]Device, error) {
	// Fetch the list of devices
	var numDevices C.size_t
	cDevices := C.edgetpu_list_devices(&numDevices)

	if cDevices == nil {
		return nil, nil
	}

	// Cast the result to a Go slice
	deviceSlice := (*[1024]C.struct_edgetpu_device)(unsafe.Pointer(cDevices))[:numDevices:numDevices]

	// Convert the list to go struct
	var devices []Device
	for i := C.size_t(0); i < numDevices; i++ {
		devices = append(devices, Device{
			Type: DeviceType(deviceSlice[i]._type),
			Path: C.GoString(deviceSlice[i].path),
		})
	}

	// Free the list
	C.edgetpu_free_devices(cDevices)

	return devices, nil
}
