# converter_mnist.py

def convert(img_file, label_file, txt_file, n_images):
  lbl_f = open(label_file, "rb")   # MNIST has labels (digits)
  img_f = open(img_file, "rb")     # and pixel vals separate
  txt_f = open(txt_file, "w")      # output file to write to

  img_f.read(16)   # discard header info
  lbl_f.read(8)    # discard header info

  for i in range(n_images):   # number images requested 
    lbl = ord(lbl_f.read(1))  # get label (unicode, one byte) 
    txt_f.write("digit " + str(lbl) + "\n")
    txt_f.write("pixls ")
    for j in range(784):  # get 784 vals from the image file
      val = ord(img_f.read(1))
      txt_f.write(str(val) + " ")  # will leave a trailing space 
    txt_f.write("\n")  # next image

  img_f.close(); txt_f.close(); lbl_f.close()

def main():
  convert("C:/Users/allen/Desktop/DigitRecognition/t10k-images.idx3-ubyte",
          "C:/Users/allen/Desktop/DigitRecognition/t10k-labels.idx1-ubyte",
          "C:/Users/allen/Desktop/DigitRecognition/mnist_check.txt", 10000)

if __name__ == "__main__":
  main()