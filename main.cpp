#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
// https://www.jianshu.com/p/077d80422c11
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>
#include <dirent.h>

#include "helper.hpp"
#include "process.hpp"
#include "detect.hpp"

using namespace std;
using namespace cv;


#define TRAIN false     //是否进行训练,true表示重新训练，false表示读取xml文件中的SVM模型
#define CENTRAL_CROP true   //true:训练时，对96*160的INRIA正样本图片剪裁出中间的64*128大小人体
//是否需要裁剪
//需要处理的目标文件
#define TARGET "target.jpg"
#define TARGET_PATH string("./targets/")

//数据来源
#define BASE_PATH string("./")
#define TRAIN_POS_PATH  BASE_PATH + string("pos_all/")
#define TRAIN_NEG_PATH  BASE_PATH + string("neg/")
#define TRAIN_HARD_PATH BASE_PATH + string("hard/")
#define SVM_PATH BASE_PATH

#define PRODUCTION 1
#define DEVELOPMENT 0
#define env PRODUCTION


int main()
{
    
    int mode=1;
    int top = 100;
    int ESize=1;
    
    if(env == PRODUCTION)
    {
        cout<<"请输入处理模式： (1:高清 2:模糊)"<<endl;
        cin>>mode;
        cout<<"请输入高度："<<endl;
        cin>>top;
        cout<<"请输入E大小："<<endl;
        cin>>ESize;
    }
    
    

        
    
    MySVM svm;
    //训练分类器
    if(TRAIN)//训练模式
    {
        train(svm,TRAIN_POS_PATH,TRAIN_NEG_PATH,SVM_PATH+"svm.xml");//正负样本的路径，svm保存路径
        //训练并将结果放在svm对象中   
    }
    else
    {
        svm.load((SVM_PATH+"svm.xml").c_str());//已经训练过
    }
    
    

    
    HOGDescriptor *myHOG = detect(svm);//设置用于检测的svm
    //特征描述子使用svm进行检测
    

    //预处理图片
    //处理目标文件
    vector<string> targets = getAllFiles(TARGET_PATH);//用于检测的图片
    for (int i=0; i<targets.size(); i++) {//逐张读取
        
        
        Mat origin = imread(targets[i]);
        

        
        Mat mask;
        vector<Mat> masks;
        masks.push_back(filterColor(origin));//设置mask，前后两次过滤，第一次预处理图片，第二次过滤识别结果
        masks.push_back(filterCanny(origin));
        
        mask = mergeMasks(masks);//复合mask，即得到最终的图片
        
        
        
        if(mode==1)//清晰图片，对于清晰的图片，进行适度腐蚀膨胀操作，以进一步提高图片中标尺的对比度。
        {
            //腐蚀、膨胀
            int erosion_size = 3;
            Mat element = getStructuringElement( MORPH_RECT,
                                                Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                                Point( erosion_size, erosion_size ) );//腐蚀矩阵
            /// 腐蚀操作
            erode( origin, origin, element );//将origin处理后放回origin，使用element操作图片
            dilate(origin, origin, element);//膨胀
        }
        else if(mode == 2)//模糊图片，对于模糊的图片数据，先进行滤波，再提高对比度
        {
            //创建并初始化滤波模板
            cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));//
            kernel.at<float>(1,1) = 5.0;
            kernel.at<float>(0,1) = -1.0;
            kernel.at<float>(1,0) = -1.0;
            kernel.at<float>(1,2) = -1.0;
            kernel.at<float>(2,1) = -1.0;
            cv::filter2D(origin,origin,origin.depth(),kernel);
            
            int alpha = 1.5;//提高对比度
            int beta = 50;
            for( int y = 0; y < origin.rows; y++ )
            {
                for( int x = 0; x < origin.cols; x++ )
                {
                    for( int c = 0; c < 3; c++ )
                    {
                        origin.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*( origin.at<Vec3b>(y,x)[c] ) + beta );
                    }
                }
            }
            
        }
        
        
        
        Mat src = origin;//处理后的图片
        cvtColor(src, src, CV_RGB2GRAY);//灰度化，加快检测速度
        equalizeHist( src, src );//直方图均衡化
        // 图像的直方图是对图像对比度效果上的一种处理，旨在使得图像整体效果均匀，黑与白之间的各个像素级之间的点更均匀一点。
        vector<Rect> found, found_filtered;//矩形框数组
        //可能是如果min_neighbors，minSize，maxSize和两个scale参数
        //设置svm用于检测的参数，此方法的任务是检测不同大小的对象，并返回矩形的列表。
        myHOG->detectMultiScale(src, found, 0, Size(8,8), Size(16,16), 1.05, 2);//found存取检测到的目标位置
        // 3.hitThreshold (可选)
        // opencv documents的解释是特征到SVM超平面的距离的阈值(Threshold for the distance between features and SVM classifying plane)
        // 所以说这个参数可能是控制HOG特征与SVM最优超平面间的最大距离，当距离小于阈值时则判定为目标。
        // 4.winStride(可选)
        // HoG检测窗口移动时的步长(水平及竖直)。
        // winStride和scale都是比较重要的参数，需要合理的设置。一个合适参数能够大大提升检测精确度，同时也不会使检测时间太长。
        // 5.padding(可选)
        // 在原图外围添加像素，作者在原文中提到，适当的pad可以提高检测的准确率（可能pad后能检测到边角的目标？）
        // 常见的pad size 有(8, 8), (16, 16), (24, 24), (32, 32).
        // 6.scale(可选)
        // 如图是一个图像金字塔，也就是图像的多尺度表示。每层图像都被缩小尺寸并用gaussian平滑。
        // scale参数可以具体控制金字塔的层数，参数越小，层数越多，检测时间也长。 一下分别是1.01  1.5 1.03 时检测到的目标。 
        // 通常scale在1.01-1.5这个区间
        // 7.finalThreshold（可选）
        // 这个参数不太清楚，有人说是为了优化最后的bounding box

        //  参数1：image–待检测图片，一般为灰度图像加快检测速度；
        // 参数2：objects–被检测物体的矩形框向量组；
        // 参数3：scaleFactor–表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
        // **参数4：**minNeighbors–表示构成检测目标的相邻矩形的最小个数(默认为3个)。
        // 如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
        // 如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
        // 这种设定值一般用在用户自定义对检测结果的组合程序上
        // 参数5：flags–要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为
        // CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，
        // 因此这些区域通常不会是人脸所在区域；
        // 参数6、7：minSize和maxSize用来限制得到的目标区域的范围。

        found = filterRect(mask, found);//使用处理后的图片，过滤异常大小的识别矩形后放入图片
        found = filterSinglePeak(found);//使用单峰过滤矩形
        
        
        int finalHeight=0;
        finalHeight = adjustRect(found);//对识别结果进行调整补齐，比如数字和E字母
        finalHeight = fitting(finalHeight);//线性拟合调整函数
        
 
        
        for (size_t i = 0; i < found.size(); i++)//opencv的数据类型size_t
        {
            cv::rectangle(origin, found[i], cv::Scalar(0, 255, 0),2);
        }
        
        imwrite(BASE_PATH+"processed/"+randName()+".png", origin);
    }
    

    return 0;
}
