//standard
#include <iostream>
#include <sstream>
#include <string>
#include <math.h>

//boost
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ANN/ANN.h>                    // ANN declarations

class ImageData{

private:
    int idx;
    std::string path;
    std::vector< double > colorArray;

public:
    ImageData();
    ImageData(int idx, std::string path);
    void setIndex(int idx){this->idx = idx;}
    void setPath(std::string path){this->path = path;}
    void setColorArray(double col){ colorArray.push_back(col); }

    int getIndex(){return idx;}
    std::string getPath(){return path;}
    std::vector<double> getColorArray(){return colorArray;}
    int getColorArraySize(){return colorArray.size();}
    std::string toString();
};

ImageData::ImageData(){
    idx = 0;
}

ImageData::ImageData(int idx, std::string path){
    this->idx = idx;
    this->path = path;
}

std::string ImageData::toString(){

    std::ostringstream oss;

    oss << "[index]: " << idx << std::endl;
    oss << "[path]: " << path << std::endl;

    for(int i = 0; i < colorArray.size() ;i++){
        oss << colorArray.at(i);
        if((i + 1) % 3 == 0 && i != 0){
            oss << std::endl;
        } else {
            oss << ",";
        }
    }

    return oss.str();
}

class NNeigh{

private:
    int k;
    int dim;
    double eps;
    int maxPts;
    int nPts;

    ANNpointArray       dataPts;                // data points
    ANNpoint            queryPt;                // query point
    ANNidxArray         nnIdx;                  // near neighbor indices
    ANNdistArray        dists;                  // near neighbor distances
    ANNkd_tree*         kdTree;                 // search structure

public:
    NNeigh();
    NNeigh(int k, int dim, double eps, int maxPts);
    void init();
    void readPoints(std::vector<ImageData> imgList);
    void structTree();
    int search(std::vector<double> colorQuery);
    void release();

};

NNeigh::NNeigh(){
    k = 1;          // number of nearest neighbors
    dim = 2;        // dimension
    eps = 0;        // error bound
    maxPts = 5000;  // maximum number of data points
    nPts = 0;       // actual number of data points
}

NNeigh::NNeigh(int k, int dim, double eps, int maxPts){
    this->k = k;
    this->dim = dim;
    this->eps = eps;
    this->maxPts = maxPts;
    this->nPts = 0;
}

void NNeigh::init(){
    dataPts = annAllocPts(maxPts, dim);
    queryPt = annAllocPt(dim);
    nnIdx = new ANNidx[k];
    dists = new ANNdist[k];
}

void NNeigh::readPoints(std::vector<ImageData> imgList){

    for(int i = 0; i < imgList.size();i++){
        for(int j = 0; j < dim ;j++){
            dataPts[i][j] = imgList.at(i).getColorArray().at(j);     
        }
    }
    nPts = imgList.size();
}

void NNeigh::structTree(){
    kdTree = new ANNkd_tree(dataPts, nPts, dim);
}

int NNeigh::search(std::vector<double> colorQuery){

    boost::random::random_device seed_gen;
    boost::random::mt19937 gen(seed_gen);
    boost::random::uniform_int_distribution<> dist(0, k - 1);

    for(int i = 0; i < dim; i++){
        queryPt[i] = colorQuery[i];
    }

    kdTree->annkSearch(queryPt, k, nnIdx, dists, eps);

    /*
    for(int j = 0; j < k ;j++){
        std::cout << "[" << j << "]:";
        std::cout << nnIdx[j] << ",";
        std::cout << sqrt(dists[j]) << std::endl;
    }
    */

    return nnIdx[dist(gen)];
}

void NNeigh::release(){
    delete [] nnIdx;
    delete [] dists;
    delete kdTree;
    annClose();
}   

void createFixPhotoMosaic(cv::Mat* img, int gwidth, int gheight, 
                            int gcount, int gdivision, 
                            std::vector<ImageData> dataList,
                            NNeigh* tree){    

    std::vector<double> colors;

    //モサイク画生成
    for(int y = 0; y < gcount ;y++){
        for(int x = 0; x < gcount ;x++){
            cv::Rect roiRect(x * gwidth,
                             y * gheight,
                            gwidth,
                            gheight);
            cv::Mat srcROI = (*img)(roiRect);
            
            for(int yy = 0; yy < gdivision;yy++){
                for(int xx = 0; xx < gdivision;xx++){
                    //std::cout << "[" << xx << "," << yy << "]" << std::endl;
                    int miniWidth = (int)(gwidth / gdivision);
                    int miniHeight = (int)(gheight / gdivision);
                    cv::Rect miniRoi(x * gwidth + xx * miniWidth,
                                     y * gheight + yy * miniHeight,
                                     miniWidth,
                                     miniHeight);
                    //std::cout << cv::mean((*img)(miniRoi)) << std::endl;
                    for(int i = 0; i < 3 ;i++){
                        colors.push_back(cv::mean((*img)(miniRoi))[i]);
                    }
                }
            }

            ImageData imgdata = dataList.at(tree->search(colors));
            cv::Rect imgROIRect(x * gwidth, y * gheight,
                            gwidth, gheight);
            cv::Mat imgROI = (*img)(imgROIRect);
            cv::Mat gridImg = cv::imread(imgdata.getPath());
            cv::resize(gridImg, gridImg, cv::Size(gwidth, gheight), 0, 0, cv::INTER_AREA);
            gridImg.copyTo(imgROI);
            colors.clear();
            

            /*
            cv::rectangle(*img,
                cv::Point(x * gwidth, y * gheight),
                cv::Point((x + 1) * gwidth, (y + 1) * gheight),
                cv::mean(srcROI),
                -1,
                CV_AA);
            cv::rectangle(*img,
                cv::Point(x * gwidth, y * gheight),
                cv::Point((x + 1) * gwidth, (y + 1) * gheight),
                cv::Scalar(0,0,0),
                1,
                CV_AA);
            */
        }
    }
}
