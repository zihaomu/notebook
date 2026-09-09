# Yolo 26 End2End Workshop

用户需要直接使用AMD w7900 来运行整体的流程。

1. 让用户了解step by step的流程，知道难点在哪。
1.1 用户了解yolo26，先从 yolo26的pt文件转换成onnx文件。
1.2 进一步让用户的yolo26 onnx 部署到 ultralytics的migraphx后端上
1.3 单独运行vlm，看看vlm整体运行的速度。展示vlm的输入和输出是什么？
1.4 yolo+vlm一起运行，一段视频来了之后，vlm和yolo怎么去处理同一帧。
1.5 怎么去加速？向用户展示图片在整个链路中搬运的过程 D2H，H2D。
1.6 单独cell展示，h2d和d2h，一张图片搬运的耗时。
1.7 全链路都在 GPU上的pipeline。

稍等：我们算一笔帐，怎么去加速整个流程。
接下来，我希望大家了解每个环节的耗时。

1. yolo的耗时是多少？
2. vlm的耗时是多少？
3. 图片资源运送的耗时？
4. 怎么加速整体的流程？
5.



2. end2end 运行流程