package delegates

import (
	"unsafe"
)

type ModifyGraphWithDelegater interface {
	ModifyGraphWithDelegate(Delegater)
}

type Delegater interface {
	Delete()
	Ptr() unsafe.Pointer
}
