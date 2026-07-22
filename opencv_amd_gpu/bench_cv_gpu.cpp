#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>
#include <string>

using namespace cv;

static double ms_since(int64 t0) {
    return (getTickCount() - t0) * 1000.0 / getTickFrequency();
}

static void cpu_pipeline(const Mat& gray32, Mat& a, Mat& b, Mat& sx, Mat& sy, Mat& mag, int inner) {
    gray32.copyTo(a);
    for (int k = 0; k < inner; ++k) {
        GaussianBlur(a, b, Size(31, 31), 0, 0);
        Sobel(b, sx, CV_32F, 1, 0, 3);
        Sobel(b, sy, CV_32F, 0, 1, 3);
        magnitude(sx, sy, mag);
        mag.copyTo(a);
    }
}

static double run_cpu(const Mat& img, int iters, int warmup, int inner) {
    Mat gray, gray32, a, b, sx, sy, mag;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray32, CV_32F, 1.0 / 255.0);
    for (int i = 0; i < warmup; ++i) cpu_pipeline(gray32, a, b, sx, sy, mag, inner);
    int64 t0 = getTickCount();
    for (int i = 0; i < iters; ++i) cpu_pipeline(gray32, a, b, sx, sy, mag, inner);
    return ms_since(t0) / iters;
}

static void run_gpu(const Mat& img, int iters, int warmup, int inner, double& compute_ms, double& full_ms) {
    cuda::GpuMat g_bgr, g_gray, g_gray32, gA, gB, gsx, gsy, gmag;
    g_bgr.upload(img);
    cuda::cvtColor(g_bgr, g_gray, COLOR_BGR2GRAY);
    g_gray.convertTo(g_gray32, CV_32F, 1.0 / 255.0);
    auto gauss = cuda::createGaussianFilter(CV_32F, CV_32F, Size(31, 31), 0);
    auto sobelx = cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, 3);
    auto sobely = cuda::createSobelFilter(CV_32F, CV_32F, 0, 1, 3);
    cuda::Stream s;
    auto pipeline = [&](cuda::Stream& st) {
        g_gray32.copyTo(gA, st);
        for (int k = 0; k < inner; ++k) {
            gauss->apply(gA, gB, st);
            sobelx->apply(gB, gsx, st);
            sobely->apply(gB, gsy, st);
            cuda::magnitude(gsx, gsy, gmag, st);
            gmag.copyTo(gA, st);
        }
    };
    for (int i = 0; i < warmup; ++i) pipeline(s);
    s.waitForCompletion();
    int64 t0 = getTickCount();
    for (int i = 0; i < iters; ++i) pipeline(s);
    s.waitForCompletion();
    compute_ms = ms_since(t0) / iters;
    Mat out;
    t0 = getTickCount();
    for (int i = 0; i < iters; ++i) {
        g_bgr.upload(img, s);
        cuda::cvtColor(g_bgr, g_gray, COLOR_BGR2GRAY, 0, s);
        g_gray.convertTo(g_gray32, CV_32F, 1.0 / 255.0, 0.0, s);
        pipeline(s);
        gmag.download(out, s);
    }
    s.waitForCompletion();
    full_ms = ms_since(t0) / iters;
}

int main(int argc, char** argv) {
    int W = 3840, H = 2160, iters = 30, warmup = 5, inner = 4, gpuloop = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--w" && i + 1 < argc) W = std::stoi(argv[++i]);
        else if (a == "--h" && i + 1 < argc) H = std::stoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
        else if (a == "--inner" && i + 1 < argc) inner = std::stoi(argv[++i]);
        else if (a == "--gpuloop" && i + 1 < argc) gpuloop = std::stoi(argv[++i]);
    }
    if (cv::cuda::getCudaEnabledDeviceCount() < 1) {
        std::cerr << "No CUDA/HIP device visible." << std::endl;
        return 2;
    }
    cv::cuda::setDevice(0);
    cv::cuda::DeviceInfo info(0);
    Mat img(H, W, CV_8UC3);
    randu(img, Scalar::all(0), Scalar::all(255));
    if (gpuloop > 0) {
        cuda::GpuMat g_bgr, g_gray, g_gray32, gA, gB, gsx, gsy, gmag;
        g_bgr.upload(img);
        cuda::cvtColor(g_bgr, g_gray, COLOR_BGR2GRAY);
        g_gray.convertTo(g_gray32, CV_32F, 1.0 / 255.0);
        auto gauss = cuda::createGaussianFilter(CV_32F, CV_32F, Size(31, 31), 0);
        auto sx = cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, 3);
        auto sy = cuda::createSobelFilter(CV_32F, CV_32F, 0, 1, 3);
        cuda::Stream s;
        for (int i = 0; i < gpuloop; ++i) {
            g_gray32.copyTo(gA, s);
            for (int k = 0; k < inner; ++k) {
                gauss->apply(gA, gB, s);
                sx->apply(gB, gsx, s);
                sy->apply(gB, gsy, s);
                cuda::magnitude(gsx, gsy, gmag, s);
                gmag.copyTo(gA, s);
            }
            if ((i % 50) == 0) s.waitForCompletion();
        }
        s.waitForCompletion();
        std::cout << "GPULOOP_DONE=" << gpuloop << std::endl;
        return 0;
    }
    double cpu_ms = run_cpu(img, iters, warmup, inner);
    double gpu_compute_ms = 0.0, gpu_full_ms = 0.0;
    run_gpu(img, iters, warmup, inner, gpu_compute_ms, gpu_full_ms);
    std::cout << "GPU_NAME=" << info.name() << std::endl;
    std::cout << "WIDTH=" << W << std::endl;
    std::cout << "HEIGHT=" << H << std::endl;
    std::cout << "ITERS=" << iters << std::endl;
    std::cout << "INNER=" << inner << std::endl;
    std::cout << "CPU_MS=" << cpu_ms << std::endl;
    std::cout << "GPU_COMPUTE_MS=" << gpu_compute_ms << std::endl;
    std::cout << "GPU_FULL_MS=" << gpu_full_ms << std::endl;
    std::cout << "SPEEDUP_COMPUTE=" << (cpu_ms / gpu_compute_ms) << std::endl;
    std::cout << "SPEEDUP_FULL=" << (cpu_ms / gpu_full_ms) << std::endl;
    return 0;
}
