package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// readWAV decodes a 16-bit PCM mono WAV file and returns samples normalized
// to [-1, 1] and the sample rate.
func readWAV(filename string) ([]float32, int, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	var riff struct {
		ID   [4]byte
		Size uint32
		Form [4]byte
	}
	if err := binary.Read(f, binary.LittleEndian, &riff); err != nil {
		return nil, 0, err
	}
	if string(riff.ID[:]) != "RIFF" || string(riff.Form[:]) != "WAVE" {
		return nil, 0, fmt.Errorf("%s: not a WAV file", filename)
	}

	var sampleRate, channels, bits, format int
	var data []byte
	for {
		var hdr struct {
			ID   [4]byte
			Size uint32
		}
		if err := binary.Read(f, binary.LittleEndian, &hdr); err == io.EOF {
			break
		} else if err != nil {
			return nil, 0, err
		}
		size := int64(hdr.Size)
		switch string(hdr.ID[:]) {
		case "fmt ":
			buf := make([]byte, size)
			if _, err := io.ReadFull(f, buf); err != nil {
				return nil, 0, err
			}
			format = int(binary.LittleEndian.Uint16(buf[0:2]))
			channels = int(binary.LittleEndian.Uint16(buf[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(buf[4:8]))
			bits = int(binary.LittleEndian.Uint16(buf[14:16]))
		case "data":
			data = make([]byte, size)
			if _, err := io.ReadFull(f, data); err != nil {
				return nil, 0, err
			}
		default:
			if _, err := io.CopyN(io.Discard, f, size); err != nil {
				return nil, 0, err
			}
		}
		// chunks are word-aligned
		if size%2 == 1 {
			io.CopyN(io.Discard, f, 1)
		}
	}
	if format != 1 || bits != 16 {
		return nil, 0, fmt.Errorf("%s: only 16-bit PCM is supported", filename)
	}
	if channels < 1 {
		return nil, 0, fmt.Errorf("%s: no channels", filename)
	}

	// downmix to mono
	frames := len(data) / 2 / channels
	samples := make([]float32, frames)
	for i := 0; i < frames; i++ {
		var sum int
		for c := 0; c < channels; c++ {
			sum += int(int16(binary.LittleEndian.Uint16(data[(i*channels+c)*2:])))
		}
		samples[i] = float32(sum/channels) / 32768
	}
	return samples, sampleRate, nil
}
