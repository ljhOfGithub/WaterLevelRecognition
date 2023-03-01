//
//  detect.cpp
//  final_assignment
//
//  Created by william wei on 17/2/22.
//  Copyright © 2017年 simon. All rights reserved.
//
//svm训练，结合训练后的svm使用hog描述子识别结果
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include "detect.hpp"
#include "helper.hpp"
#include "process.hpp"

using namespace cv;

HOGDescriptor* detect(MySVM &svm)//测试，传入已经训练好的svm模型
{
    HOGDescriptor hog(WIN_SIZE,BLOCK_SIZE,BLOCK_STRIDE,CELL_SIZE,BIN);  //HOG检测器，用来计算HOG描述子的
    int DescriptorDim;//HOG描述子的维数

    //特征向量的维数不是用户指定，用户只指定步长，窗口等
    DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数，通过特征向量的维数获得描述子的维数
    int supportVectorNum = svm.get_support_vector_count();//支持向量的个数

    //创建1*supportVectorNum的矩阵
    Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
    Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
    Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果矩阵，矩阵相乘
    //resultMat是1*DescriptorDim的矩阵
    //将支持向量的数据复制到supportVectorMat矩阵中，逐个复制
    for(int i=0; i<supportVectorNum; i++)
    {
        const float * pSVData = svm.get_support_vector(i);//获得支持向量数组
        for(int j=0; j<DescriptorDim; j++)
        {
            supportVectorMat.at<float>(i,j) = pSVData[j];//逐个复制支持向量数组
        }
    }
    
  
    //返回SVM的决策函数中的alpha向量
    double * pAlphaData = svm.get_alpha_vector();
    for(int i=0; i<supportVectorNum; i++)
    {
        alphaMat.at<float>(0,i) = pAlphaData[i];
    }

    resultMat = -1 * alphaMat * supportVectorMat;//计算结果矩阵，*-1，存放分类结果
    //注意因为svm.predict使用的是alpha*sv*another-rho，如果为负的话则认为是正样本，在HOG的检测函数中，使用rho+alpha*sv*another(another为-1)
    //sv是支持向量

    vector<float> myDetector;
    //resultMat是1*DescriptorDim的矩阵
    //将resultMat结果矩阵中的数据复制到数组myDetector中
    for(int i=0; i<DescriptorDim; i++)
    {
        myDetector.push_back(resultMat.at<float>(0,i));//resultMat第一行的所有列，是alpha*sv*another(another为-1)的结果
    }

    myDetector.push_back(svm.get_rho());//获得rho
    //设置hog用于提取特征的参数，比如步长
    HOGDescriptor *myHOG = new HOGDescriptor(WIN_SIZE,BLOCK_SIZE,BLOCK_STRIDE,CELL_SIZE,BIN);

    myHOG->setSVMDetector(myDetector);//设置svm检测器
    // SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
    // 将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个行向量，将该向量前面乘以-1。之后，再该行向量的最后添加一个元素rho。
    // 如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器即使用cv::HOGDescriptor::setSVMDetector()
    return myHOG;
}


void train(MySVM &svm,string posPath,string negPath,string savePath="")//训练,//使用SVM学习
{
    HOGDescriptor hog(WIN_SIZE,BLOCK_SIZE,BLOCK_STRIDE,CELL_SIZE,BIN);//HOG检测器，用来计算HOG描述子的
    unsigned int DescriptorDim = 0;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
    vector<string> posImg = getAllFiles(posPath);//正样本
    vector<string> negImg = getAllFiles(negPath);//负样本
    unsigned long posNum = posImg.size();
    unsigned long negNum = negImg.size()*2;
    string ImgName;//图片名(绝对路径)
    
    
    Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
    Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有尺子，-1表示无尺子
    
    for (int num=0; num<posImg.size(); num++)//正样本图片
    {
        Mat origin = imread(posImg[num]);//读取图片
        Mat src;//应该是写漏了将src转换为origin格式，并保存在src对象中
        
        std::cout<<posImg[num];//输出被读图片路径
        
        vector<float> descriptors;//HOG描述子向量，float数组
        hog.compute(src,descriptors,BLOCK_STRIDE);//计算HOG描述子，设置检测窗口移动步长(8,8)，应该是src的hog
        //一张一张计算描述子，hog描述子复制到样本特征矩阵sampleFeatureMat
        //处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
        if( num == 0 )
        {
            DescriptorDim = descriptors.size();//HOG描述子的维数

            sampleFeatureMat = Mat::zeros(posNum+negNum, DescriptorDim, CV_32FC1);//行：样本数，列：HOG描述子的维数
            sampleLabelMat = Mat::zeros(posNum+negNum, 1, CV_32FC1);//设置默认类
        }
        
        //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
        for(int i=0; i<DescriptorDim; i++)//也复制了第一个样本的hog描述子
            sampleFeatureMat.at<float>(num,i) = descriptors[i];//第num个样本的特征向量中的第i个元素
        
        sampleLabelMat.at<float>(num,0) = 1;//正样本类别为1，有人
    }
    
    
    
    //依次读取负样本图片，生成HOG描述子
    for(int j=0; j<negNum; j++)
    {
        int num = j/2;
        
        Mat origin = imread(negImg[num]);
        Mat src;
        cvtColor( origin, src, CV_BGR2GRAY );//将读入的图片转换为灰度图，将后者src设置为和前者origin一样，保存在src中
        Mat img1 = src(Rect(0,0,64,64));//裁切src
        Mat img2 = src(Rect(0,64,64,64));
        
        vector<float> descriptors;//HOG描述子向量
        hog.compute(img1,descriptors,BLOCK_STRIDE);//计算HOG描述子，检测窗口移动步长(8,8)，计算第一部分的hog

        
        //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
        for(int i=0; i<DescriptorDim; i++)
            sampleFeatureMat.at<float>(j+posNum,i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
        
        sampleLabelMat.at<float>(j+posNum,0) = -1;//负样本类别为-1，无人
    }
    //以上两个for得到所有图片的hog
    
    //训练SVM分类器
    CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
    //注意必须使用线性SVM进行训练，因为HogDescriptor检测函数只支持线性检测！！！
    CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);//设置svm参数

    svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器，使用得到的hog特征和标签，有监督学习
    if(savePath.length()>0)//保存分类器(里面包括了SVM的参数，支持向量,α和rho)
    {
        svm.save(savePath.c_str());//将训练好的SVM模型保存为xml文件
    }
    
    
}


