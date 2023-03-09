# engineer_2023
# 一.主要功能
## 1.inference_function
***该文件主要是对推理出来的结果进行相应的算法处理最终达到需要的效果***
### `to_inference`
+ 对推理结果及处理算法的整合
### `station_function`
+ 兑换站 `pitch` 和 `roll`角度解析的各个功能算法， 以及兑换站串口待发信息的初始化
### `mineral_function`
+ 矿石串口待发信息的初始化
### `share_function`
+ 通用功能算法的编写
---
## 2.serial_function
***该文件主要是对串口功能的编写***
### `serial_function`
+ 串口初始化，串口掉线重连
+ 兑换站，矿石数据的收发
---
## 3.video_function
### `video_capture`
+ 相机初始化文件，添加了颜色补偿和录像功能
### `mvsdk`
+ 工业相机驱动配置文件
---
## 4.main
+ 多线程启动功能代码
---
## 5.其他
+ 为`YOLOV5`运行所需文件
---
# 二.环境
+ 所使用都为`YOLOV5 v6.0`的环境，要是提示环境问题大概了就算所装依赖版本太高的问题，运行环境不要与 requirements.txt 的相差太多不然会出现问题
---
# 三.兑换站识别流程
![流程图](./%E5%85%91%E6%8D%A2%E7%AB%99%E6%8E%A8%E7%90%86%E6%B5%81%E7%A8%8B.png)