package tensor

// #cgo CFLAGS: -I C:/Clarenti/Program/Libtf/libtensorflow-cpu-windows-x86_64-2.15.0/include/tensorflow/c
// #cgo LDFLAGS: -L C:/Clarenti/Program/Libtf/libtensorflow-cpu-windows-x86_64-2.15.0/lib -ltensorflow
// import "C"
import (
	// "C"
	"fmt"
	"image"

	"image/color"
	"image/jpeg"
	"math"
	"os"

	svm "github.com/ewalker544/libsvm-go"
	"github.com/nfnt/resize"

	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"

	// gocr "github.com/otiai10/gosseract/v2"

	// "gocv.io/x/gocv"
	"gonum.org/v1/gonum/mat"
	// "github.com/kornia/kornia"
)

type SVM struct {
	model interface{}
}

// func OCR(c *gin.Context) {
// 	client := gocr.NewClient()
// 	defer client.Close()

// 	path := `C:\Clarenti\Data\Project\GoUtils\GoUtils\assets\1.jpg`
// 	client.SetImage(path)
// 	client.SetLanguage("eng")
// 	t, _ := client.Text()
// 	fmt.Println(t)
// }

func Train() {
	// image.RegisterFormat("jpeg", "jpeg", jpeg.Decode, jpeg.DecodeConfig)
	param := svm.NewParameter()
	param.KernelType = svm.RBF
	param.C = 1.
	param.Degree = 3.
	param.Coef0 = 0.0
	param.Gamma = 1.

	model := svm.NewModel(param)

	// Create a problem specification from the training data and parameter attributes
	problem, err := svm.NewProblem(`C:/Clarenti/Data/Project/Py/ML/SVM-GLCM-CM_train.csv`, param)
	if err != nil {
		print(err)
		return
	}

	model.Train(problem) // Train the model from the problem specification

	model.Dump("train.model")
}

// docker pull tensorflow/tensorflow:2.9.1
func PredictTensor() {
	model := tg.LoadModel(`C:\Clarenti\Data\Project\Go\TensorGO\output\keras`, []string{"serve"}, nil)

	path := `C:\Clarenti\Data\Project\Go\TensorGO\assets\3.jpg`
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	input, _ := tf.NewTensor(data)

	results := model.Exec([]tf.Output{
		model.Op("StatefulPartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		model.Op("serving_default_inputs_input", 0): input,
	})
	predictions := results[0]
	fmt.Println(predictions.Value())
}

func Predict() {
	image.RegisterFormat("jpeg", "jpeg", jpeg.Decode, jpeg.DecodeConfig)

	path := `C:\Clarenti\Data\Project\Go\GoUtils\GoUtils\assets\1.jpg`
	file, err := os.Open(path)
	if err != nil {
		fmt.Println("Error opening image:", err)
		return
	}
	defer file.Close()

	img, format, err := image.Decode(file)
	println("format ", format)
	if err != nil {
		fmt.Println("Error decoding image:", err)
		return
	}

	// RESIZE
	img = resize.Resize(uint(550), uint(350), img, resize.Lanczos3)

	// FEATURE EXTRACTION
	var features []float64

	// Image to byte arr
	// buf := bytes.Buffer{}
	// jpeg.Encode(&buf, img, &jpeg.Options{})

	println("COLOR EXTRACT START")

	fcolor := ColorExtract(img)

	grays := image.NewGray(img.Bounds())
	// Convert each pixel to grayscale
	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			// Get the color of the current pixel
			r, g, b, _ := img.At(x, y).RGBA()

			// Calculate the grayscale value (average of RGB values)
			grayVal := uint8((r>>8 + g>>8 + b>>8) / 3)

			// Set the grayscale value for the current pixel in the new image
			grays.SetGray(x, y, color.Gray{Y: grayVal})

		}
	}

	// Angle GLCM
	dx := []int{1, 1, 0, -1}
	dy := []int{0, 1, 1, 1}

	// Conditional as your extraction on training section
	selected_distance := dx[0]
	selected_angle := dy[0]

	println("GLCM START")
	// Compute GLCM for different directions
	// for i := 0; i < len(dx); i++ {
	graymatrix := graycomatrix(*grays, selected_distance, selected_angle)
	contrast, energy, homogeneity, correlation := graycocrops(graymatrix)
	// }
	println("GLCM DONE")

	features = fcolor
	fmt.Println("COLOR : ", features)
	features = append(features, contrast, energy, homogeneity, correlation)
	fmt.Println("COLOR-GLCM : ", features)

	// fcolor, fglcm = extraction(550, 350, img_mat.get)
	model := svm.NewModelFromFile(`C:\Clarenti\Data\Project\Go\GoUtils\GoUtils/assets/svm_model.json`)
	x := make(map[int]float64, len(features))
	for i := 0; i < len(features); i++ {
		x[i] = features[i]
	}
	fmt.Println("X Features ", x)
	y_pred := model.Predict(x)
	print(y_pred)
	// c.JSON(200, y_pred)
}

func ColorExtract(img image.Image) []float64 {
	bounds := img.Bounds()
	R := mat.NewDense(bounds.Max.X, bounds.Max.Y, nil)
	G := mat.NewDense(bounds.Max.X, bounds.Max.Y, nil)
	B := mat.NewDense(bounds.Max.X, bounds.Max.Y, nil)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			R.Set(x, y, float64(uint8(r)))
			G.Set(x, y, float64(uint8(g)))
			B.Set(x, y, float64(uint8(b)))
		}
	}

	// r, c := img.Dims()

	// R := make([]float64, r*c)
	// G := make([]float64, r*c)
	// B := make([]float64, r*c)

	// matColIdx := 0
	// for j := 0; j < c; j++ {
	// 	for i := 0; i < r; i++ {
	// 		R[matColIdx] = img.At(i, j)
	// 		G[matColIdx] = img.At(i, j+1)
	// 		B[matColIdx] = img.At(i, j+2)
	// 		matColIdx++
	// 	}
	// }

	var sumR, sumG, sumB float64
	var meanR, meanG, meanB float64
	var varianceR, varianceG, varianceB float64
	var differenceR, differenceG, differenceB float64

	// NOTES: DIM OF R/G/B MUST BE SAME
	r, c := R.Dims()
	// R
	for _, val := range R.RawMatrix().Data {
		meanR += val
		diff := val - meanR
		sumR += diff * diff
		differenceR -= math.Pow(val-meanR, 3)
	}
	meanR /= float64(r * c)
	varianceR = sumR / float64(r*c)

	// G
	for _, val := range G.RawMatrix().Data {
		meanG += val
		diff := val - meanG
		sumG += diff * diff
		differenceG -= math.Pow(val-meanG, 3)
	}
	meanG /= float64(r * c)
	varianceG = sumR / float64(r*c)

	// B
	for _, val := range B.RawMatrix().Data {
		meanB += val
		diff := val - meanB
		sumB += diff * diff
		differenceB -= math.Pow(val-meanB, 3)
	}
	meanB /= float64(r * c)
	varianceB = sumB / float64(r*c)

	// ===========================================================

	// RdataFlat := R.RawMatrix().Data
	// for _, val := range RdataFlat {
	// 	diff := val - meanR
	// 	sumR += diff * diff
	// }
	// varianceR = sumR / float64(r*c)

	// // G
	// for _, val := range G {
	// 	meanG += val
	// }
	// meanG /= float64(r * c)
	// G_ := mat.NewDense(r, c, G)
	// GdataFlat := G_.RawMatrix().Data
	// for _, val := range GdataFlat {
	// 	diff := val - meanR
	// 	sumR += diff * diff
	// }
	// varianceG = sumG / float64(r*c)

	// // B
	// for _, val := range B {
	// 	meanB += val
	// }
	// meanB /= float64(r * c)
	// B_ := mat.NewDense(r, c, B)
	// BdataFlat := B_.RawMatrix().Data
	// for _, val := range BdataFlat {
	// 	diff := val - meanR
	// 	sumR += diff * diff
	// }
	// varianceB = sumB / float64(r*c)

	N := float64(r * c)

	// for _, val := range R {
	// 	differenceR -= math.Pow(val-meanR, 3)
	// }

	// for _, val := range G {
	// 	differenceG -= math.Pow(val-meanG, 3)
	// }

	// for _, val := range B {
	// 	differenceB -= math.Pow(val-meanB, 3)
	// }

	skewnessR := math.Cbrt(differenceR / N)
	skewnessG := math.Cbrt(differenceG / N)
	skewnessB := math.Cbrt(differenceB / N)

	return []float64{meanR, meanG, meanB, varianceR, varianceG, varianceB, skewnessR, skewnessG, skewnessB}
}

func graycomatrix(img image.Gray, dx, dy int) [][]float64 {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	grayLevels := 256
	glcm := make([][]float64, grayLevels)
	for i := range glcm {
		glcm[i] = make([]float64, grayLevels)
	}

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// Get current pixel intensity
			pixel1 := img.GrayAt(x, y)

			// Calculate coordinates for the neighbor pixel
			neighborX, neighborY := x+dx, y+dy

			// Check if neighbor pixel is within image boundaries
			if neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height {
				// Get neighbor pixel intensity
				pixel2 := img.GrayAt(neighborX, neighborY)

				// Update GLCM
				glcm[pixel1.Y][pixel2.Y]++
			}
		}
	}

	// Normalize GLCM
	totalPixels := float64(width * height)
	for i := range glcm {
		for j := range glcm[i] {
			glcm[i][j] /= totalPixels
		}
	}

	return glcm
}

func graycocrops(graycomatrix [][]float64) (contrast, energy, homogeneity, correlation float64) {
	n := len(graycomatrix)
	var muX, muY, sigmaX, sigmaY, covariance float64

	// Calculate mean and standard deviation
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			muX += float64(i) * graycomatrix[i][j]
			muY += float64(j) * graycomatrix[i][j]
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			sigmaX += math.Pow(float64(i)-muX, 2) * graycomatrix[i][j]
			sigmaY += math.Pow(float64(j)-muY, 2) * graycomatrix[i][j]
			covariance += (float64(i) - muX) * (float64(j) - muY) * graycomatrix[i][j]
		}
	}
	sigmaX = math.Sqrt(sigmaX)
	sigmaY = math.Sqrt(sigmaY)

	// Compute GLCM features
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			contrast += math.Pow(float64(i)-float64(j), 2) * graycomatrix[i][j]
			energy += math.Pow(graycomatrix[i][j], 2)
			homogeneity += graycomatrix[i][j] / (1 + math.Abs(float64(i)-float64(j)))
		}
	}
	correlation = covariance / (sigmaX * sigmaY)

	return contrast, energy, homogeneity, correlation
}
