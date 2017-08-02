import numpy as np
import glob
import cv2
import sys
np.set_printoptions(threshold=np.inf)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
window_size = 64
block_size = 16 #only 16,16 is supported by openCV
stride_size = 8
cell_size = 8 
bin_size = 9

def process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray= cv2.resize(gray, (window_size, window_size))
    gray = clahe.apply(gray) 
    return gray

if __name__ == '__main__':
    import getopt
    optlist, args = getopt.getopt(sys.argv[1:], '', ['positive_data_dir=', 'negative_data_dir='])
    args = dict(optlist)
    positive_dir = args.get('--positive_data_dir', "/Volumes/Mac/png/frontal_positives")
    negative_dir = args.get('--negative_data_dir', "/Volumes/Mac/png/frontal_negatives")
    print("Training using " + positive_dir + " and " + negative_dir)
    extension = args.get('--extension', 'png') 
     
    positive_data_files = glob.glob(positive_dir + '/*.' + extension )
    negative_data_files = glob.glob(negative_dir + '/*.' + extension ) 
    file_length = len(positive_data_files) + len(negative_data_files)
    feature_size =  int(((window_size - block_size) / stride_size + 1) ** 2 * (block_size / cell_size) ** 2 * bin_size) 

    #Data
    HOG_features = np.empty((feature_size, file_length), dtype=np.float32)
    labels = np.ones((1, file_length), dtype=np.int32 )
    labels[0, len(positive_data_files):file_length - 1] = 0 
    hog = cv2.HOGDescriptor((window_size, window_size), (block_size, block_size), (stride_size, stride_size), (cell_size, cell_size), bin_size)
    hog.save('hog.xml')
    index = 0

    for files in [positive_data_files, negative_data_files]:
        for file in files:
            image = cv2.imread(file)    
            gray = process(image)
            HOG_features[:, index] = hog.compute(gray)[:, 0]
            index += 1

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1000, 1.e-07))
    svm.train(HOG_features, cv2.ml.COL_SAMPLE, labels)
    svm.save('output.xml') 
   
    samples = np.reshape(HOG_features, (file_length, feature_size))
    results = svm.predict(samples)[1].ravel()
    unique, counts = np.unique(results - labels, return_counts=True)
    counts = dict(zip(unique, counts))
    print(counts)
    print( 'false positive rate: %.4f' % (float(counts[1]) / len(negative_data_files)))
    #print(np.histogram(results - labels, bins=[0,1,-1]) )
