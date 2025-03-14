def convert(imgf, labelf, outimg, outlbl,n):
    f = open(imgf, "rb")
    l = open(labelf, "rb")
    oi = open(outimg, "w")
    ol = open(outlbl, "w")

    f.read(16)
    l.read(8)
    images = []
    labels = []

    for i in range(n):
        image = []
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for i in range(n):
        label = []
        label.append(ord(l.read(1)))
        labels.append(label)

    for image in images:
        oi.write(" ".join(str(pix) for pix in image) + "\n")

    for label in labels:
        for i in range(10):
            if label[0] == i:
                ol.write(str(1))
            else:
                ol.write(str(0))
            if i != 9: ol.write(" ")
        ol.write("\n")
    f.close()
    l.close()
    oi.close()
    ol.close()

def convert2(imgf, labelf, outimg, outlbl,n):
    f = open(imgf, "rb")
    l = open(labelf, "rb")
    oi = open(outimg, "w")
    ol = open(outlbl, "w")

    f.read(16)
    l.read(8)
    images = []
    labels = []

    for i in range(n):
        image = []
        for j in range(28 * 28):
            oi.write(f.read(1))
        oi.write("\n")

    for i in range(n):
        labels.append(l.read(1))

    for image in images:
        oi.write((pix for pix in image) + "\n")

    for label in labels:
        ol.write((l for l in label) + "\n")
    f.close()
    l.close()
    oi.close()
    ol.close()
def read_txt():
    data = []
    with open("MNISTvINTEL_trainImages.txt", "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            line = line.split()
            data.append(line)
        print(data[0])

read_txt()
# convert("../data/mnist/raw/train-images-idx3-ubyte", "../data/mnist/raw/train-labels-idx1-ubyte",
#         "MNISTvINTEL_trainImages.txt", "MNISTvINTEL_trainLabels.txt" ,60000)
# convert("../data/mnist/raw/t10k-images-idx3-ubyte", "../data/mnist/raw/t10k-labels-idx1-ubyte",
#         "MNISTvINTEL_testImages.txt", "MNISTvINTEL_testLabels.txt", 10000)

print("Convert Finished!")