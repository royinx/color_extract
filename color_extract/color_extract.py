import cv2
from skimage.color import rgb2hsv, hsv2rgb
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from sklearn.cluster import KMeans , DBSCAN
import numpy as np 
import json

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

class ColorFilter(object):
    def __init__(self, color_fmt):
        self.read_json(color_fmt)

    def read_json(self, class_fmt):
        with open('config', 'r') as infile:
            data = json.load(infile)
            for key, value in data[class_fmt].items():
                setattr(self, key, value)

    def color_filter(self, image, colors): # Inference 
        color_percent = []
        for color in colors:
            color_pixel_array = np.where(np.all([np.logical_and(image[:,:,0]>=color[0]-self.color_range[0], image[:,:,0]<=color[0]+self.color_range[0]),
                                                 np.logical_and(image[:,:,1]>=color[1]-self.color_range[1], image[:,:,1]<=color[1]+self.color_range[1]),
                                                 np.logical_and(image[:,:,2]>=color[2]-self.color_range[2], image[:,:,2]<=color[2]+self.color_range[2])],
                                                 axis=0))
            color_percent.append(len(image[color_pixel_array])/(image.size/3)*100)
        print("inference",color_percent)
        return color_percent

    def color_match(self, img, colors: np.array , target_percent: np.array): 
        # image and pixel color matching , color_threshold: percent_std_dev, color_range: RGB +- 15
        color_percent = self.color_filter(img, colors)
        bl = np.logical_and(color_percent >= target_percent - self.color_threshold , color_percent <= target_percent + self.color_threshold)

        return np.all(bl) # match or not

    def dominantColors_Kmeans(self, image, plot, k=5):  # k = no_class
        #reshaping to a list of pixels
        img = image.reshape((image.shape[0] * image.shape[1], 3))
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = k)
        idx = kmeans.fit_predict(img)

        #the cluster centers are our dominant colors.
        colors = kmeans.cluster_centers_
        colors.astype(int) if self.fmt == 'rgb' else colors
        #save labels
        labels = kmeans.labels_


        percent = np.array([ np.divide(sum(idx==class_)*100,len(idx)) for class_ in range(k)],dtype=np.int8)
        
        if plot:
            self.plot(img, idx)
        #returning after converting to integer from float
        return colors, percent

    # Caution: size of image < 250*250,3 as possible
    def dominantColors_DBSCAN(self, image, plot):
        #reshaping to a list of pixels
        img = image.reshape((image.shape[0] * image.shape[1], 3))
        dbs = DBSCAN(self.eps ,self.min_sample)
        idx = dbs.fit_predict(img)
        no_class = len(set(idx))


        if self.fmt == 'hsv':
            colors = np.array([np.average(img[idx==class_],axis = 0) for class_ in range(no_class)])
        elif self.fmt == 'rgb':
            colors = np.array([np.average(img[idx==class_],axis = 0) for class_ in range(no_class)],dtype=np.int16)
        else:
            raise('Invalid color format')

        percent = np.array([ np.divide(sum(idx==class_)*100,len(idx)) for class_ in range(no_class)],dtype=np.int8)

        if plot:
            self.plot(img, idx)

        return colors, percent

    def plot(self, img, idx):
        fig = plt.figure()
        fig.set_size_inches(20, 15)
        ax = fig.add_subplot(111, projection='3d')

        xs = img[:,0]
        ys = img[:,1]
        zs = img[:,2]
        ax.scatter(xs, ys, zs, marker='o', c=idx)

        ax.set_xlabel('{} Label'.format(self.label[0]))
        ax.set_ylabel('{} Label'.format(self.label[1]))
        ax.set_zlabel('{} Label'.format(self.label[2]))
        plt.show()

    def rescale(self, img):
        return rgb_to_hsv(img/self.scale) if self.fmt == 'hsv' else img

    def resize(self, image,size=224):
        scale = image.shape[1]/image.shape[0]  # W/H
        inp = cv2.resize(image,(int(size*scale),size), interpolation = cv2.INTER_LINEAR) #(resize shape = (w,h) )
        inp = inp.astype(np.uint8)
        inp = self.rescale(inp)
        return inp

    def extraction(self, image, plot):
        color_list, percent = self.dominantColors_Kmeans(image,plot)
        # color_list, percent = self.dominantColors_DBSCAN(image, plot)
        return color_list, percent

    # convert color_list into RGB 
    def recolor(self, color_list):
        rgb = []
        for idx,color in enumerate(color_list):
            rgb.append((hsv_to_rgb(color.reshape(1,1,1,3))*255).astype(np.uint8))
        return np.array(rgb).squeeze()
         

def extract_color(img:np.array,plot , color_fmt):
    color_filter = ColorFilter(color_fmt)
    img = color_filter.resize(img,100)
    # plt.imshow(img)
    # plt.show()

    color_list, percent = color_filter.extraction(img,plot)

    if color_fmt == 'hsv':
        print('========= HSV in RGB ========:',color_filter.recolor(color_list))

    bl = color_filter.color_match(img, color_list, percent)
    
    print(color_list,percent,bl)
    return(bl)


if __name__ == '__main__':
    file_name = 'jacket.jpg'
    img = cv2.imread(file_name)
    rgb_img = img[:,:,[2,1,0]]
    extract_color(rgb_img, color_fmt = 'rgb', plot = True)

    ## highly recommend usd HSV instead of RGB. 
    # KMEAN (RGB)
    # ================== KMEAN (RGB) ==================
    # [[253.48762542 253.34876254 253.28749164]
    #  [ 72.14611212  70.09077758  76.21844485]
    #  [ 26.63396861  26.00840807  26.242713  ]
    #  [ 49.23721468  48.28720023  51.38832707]
    #  [107.79509202 101.9006135  104.87116564]] 

    # Percent of image - Clustering
    #  [45                 17                  10                  21                 5] 
    # Percent of image - Inference
    #  [44.24539877300614, 18.877300613496935, 15.380368098159508, 27.60122699386503, 4.920245398773006]


    # ===============================================================================================

    # KMEAN (HSV)
    # ================== KMEAN (HSV) ==================
    # HSV Color 
    # [[0.67881244 0.07842921 0.25975267]
    #  [0.00600736 0.00213144 0.99450272]
    #  [0.07255089 0.09488661 0.18018258]
    #  [0.60250057 0.51682981 0.28225705]
    #  [0.9309358  0.08319914 0.29192116]]
    # In RGB Color Form
    # [[ 61  61  66]
    #  [253 253 253]
    #  [ 45  43  41]
    #  [ 34  49  71]
    #  [ 74  68  70]]

    # Percent of image - Clustering
    #  [16                  45                  17                  4                   16] 
    # Percent of image - Inference
    #  [16.87730061349693,  44.858895705521476, 16.05521472392638,  4.288343558282208,  21.11042944785276]

    # ===============================================================================================

    # DBSCAN (RGB)
    # ================== DBSCAN (RGB) ==================
    # In RGB Color Form
    # [[253 253 253]
    #  [ 57  54  56]
    #  [ 41  55  83]
    #  [  0   0   0]] 

    # Percent of image - Clustering
    # [45                   49                  3                   0] 
    # Percent of image - Inference
    # [44.325153374233125,  28.380368098159508, 3.717791411042945,  2.4417177914110426]

    # ===============================================================================================

    # DBSCAN (HSV)
    # ================== DBSCAN (HSV) ==================
    # HSV Color 
    # [[3.18887352e-05 2.15804919e-05 9.97063976e-01]
    #  [8.17258231e-01 7.84095971e-02 2.68741079e-01]
    #  [6.11918612e-01 5.03692210e-01 3.28533497e-01]
    #  [6.62706294e-02 8.08771816e-02 1.83760526e-01]
    #  [           nan            nan            nan]]
    # In RGB Color Form
    # [[254 254 254]
    #  [ 68  63  68]
    #  [ 41  55  83]
    #  [ 46  44  43]
    #  [  1   0 101]]

    # Percent of image - Clustering
    # [44                  29                 3                  15                  0 ] True
    # Percent of image - Inference
    # [44.852760736196316, 28.14723926380368, 4.177914110429448, 15.969325153374234, 0.0]

    # ===============================================================================================



    # HSV may cluster out more accurate in differenet color in 3D graph.