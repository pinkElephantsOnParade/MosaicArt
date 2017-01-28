//standard
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>

//boost
#include <boost/algorithm/string/classification.hpp> // is_any_of
#include <boost/algorithm/string/split.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/optional.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>

//opencv2
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "mosaic.cpp"

namespace p_tree = boost::property_tree;

boost::optional<int> windowWidth;
boost::optional<int> windowHeight;
boost::optional<int> gridWidth;
boost::optional<int> gridHeight;
boost::optional<int> gridDivision;

void getSize(boost::optional<int>* ww,
            boost::optional<int>* wh,
            boost::optional<int>* gw,
            boost::optional<int>* gh,
            boost::optional<int>* gd
            ){

    p_tree::ptree pt;
    p_tree::read_ini("mosaicData.ini", pt);

    *ww = pt.get_optional<int>("mosaic.windowWidth");
    *wh = pt.get_optional<int>("mosaic.windowHeight");
    *gw = pt.get_optional<int>("mosaic.gridWidth");
    *gh = pt.get_optional<int>("mosaic.gridHeight");
    *gd = pt.get_optional<int>("mosaic.gridDivision");

    std::cout << "windowWidth:" << *ww;
    std::cout << " windowHeight:" << *wh;
    std::cout << " gridWidth:" << *gw;
    std::cout << " gridHeight:" << *gh;
    std::cout << " gridDivision" << *gd << std::endl;
}

int main(int argc, char** args){

    std::string filename("imagedata.csv");
    std::ifstream fin;
    fin.open(filename, std::ios::in);
    std::string titleName = "Mosaic art";

    std::string readline;

    getSize(&windowWidth,
        &windowHeight,
        &gridWidth,
        &gridHeight,
        &gridDivision);

    int gridCount = *windowWidth / *gridWidth;
    
    std::vector< std::vector<int> > gridBoard;
    std::vector<ImageData> dataList;

    //画像データ読み込み
    while(!fin.eof()){
        std::vector<std::string> result;
        std::getline(fin, readline);
        boost::algorithm::split(result, readline, boost::is_any_of(",")); // カンマで分割
        ImageData data = ImageData(atoi(result.at(0).c_str()), result.at(1)); 
        for(int i = 0; i < 3 * *gridDivision * *gridDivision ;i++){
            data.setColorArray(atof(result.at(2 + i).c_str()));
        }
        dataList.push_back(data);

        result.clear();
    }

    //std::cout << dataList.at(2000).toString() << std::endl;

    boost::random::random_device seed_gen;
    boost::random::mt19937 gen(seed_gen);
    boost::random::uniform_int_distribution<> dist(0, dataList.size() - 1);

    NNeigh* colorNN = new NNeigh(10, 3 * *gridDivision * *gridDivision, 0, (int)(dataList.size() * 1.5));
    colorNN->init();
    colorNN->readPoints(dataList);
    colorNN->structTree();

    ImageData idata = dataList.at(dist(gen));
    cv::Mat img = cv::imread(idata.getPath());
    cv::Mat dstImg = img.clone();

    std::cout << idata.toString() << std::endl;

    //モザイク画作成
    createFixPhotoMosaic(&dstImg, *gridWidth, *gridHeight, gridCount, *gridDivision, dataList, colorNN);

    colorNN->release();

    cv::resize(img, img, cv::Size(640,360), 0, 0, cv::INTER_AREA);

    while(true){
        cv::namedWindow(titleName);
        cv::imshow(titleName, dstImg);
        cv::namedWindow("source");
        cv::imshow("source", img);
        int key = cv::waitKey(0);
        if(key == 'q'){
            break;
        } else if(key == 's'){
            cv::imwrite("mosaic.png", dstImg);
        }
    }

    return 0;
}