
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>
using namespace cv;
int main(int argc, char** argv) {
    if (argc != 3 || cuda::getCudaEnabledDeviceCount() < 1) return 2;
    cuda::setDevice(0); cuda::DeviceInfo info(0);
    Mat bgr=imread(argv[1]), input, cpu, gpu;
    if (bgr.empty()) return 3;
    cvtColor(bgr,input,COLOR_BGR2BGRA);
    GaussianBlur(input,cpu,Size(31,31),0,0,BORDER_DEFAULT);
    cuda::GpuMat gi,go; gi.upload(input);
    auto f=cuda::createGaussianFilter(CV_8UC4,CV_8UC4,Size(31,31),0,0,BORDER_DEFAULT);
    f->apply(gi,go); go.download(gpu);
    Mat d; absdiff(cpu,gpu,d); double mx=0; minMaxLoc(d.reshape(1),nullptr,&mx);
    Scalar m=mean(d); double avg=(m[0]+m[1]+m[2]+m[3])/4.0;
    std::vector<Mat> ch; split(d,ch);
    Mat diff_gray; max(ch[0],ch[1],diff_gray); max(diff_gray,ch[2],diff_gray);
    int changed=countNonZero(diff_gray);
    double changed_pct=100.0*changed/diff_gray.total();
    Mat normalized, heatmap;
    if(mx>0) diff_gray.convertTo(normalized,CV_8U,255.0/mx); else normalized=Mat::zeros(diff_gray.size(),CV_8U);
    applyColorMap(normalized,heatmap,COLORMAP_TURBO);
    heatmap.setTo(Scalar(0,0,0),diff_gray==0);
    String dir=argv[2]; imwrite(dir+"/cpu_gaussian.png",cpu);
    imwrite(dir+"/gpu_gaussian.png",gpu); imwrite(dir+"/difference_heatmap.png",heatmap);
    std::cout<<"GPU_NAME="<<info.name()<<std::endl;
    std::cout<<"MAX_ABS_ERROR="<<mx<<std::endl;
    std::cout<<"MEAN_ABS_ERROR="<<avg<<std::endl;
    std::cout<<"PSNR_DB="<<PSNR(cpu,gpu)<<std::endl;
    std::cout<<"CHANGED_PIXELS="<<changed<<std::endl;
    std::cout<<"CHANGED_PERCENT="<<changed_pct<<std::endl;
    return 0;
}
