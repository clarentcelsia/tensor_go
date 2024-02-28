package main

import (
	t "goutils/tensor"
	"os"
)

func init() {
	os.Setenv("CD..FLAGS...", "VALUE")
}
func main() {
	t.PredictTensor()
}
