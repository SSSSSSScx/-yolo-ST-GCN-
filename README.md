# 实验室智能预警系统

基于计算机视觉的实验室安全实时监控与智能预警系统。支持多路摄像头接入、人员检测跟踪、行为识别、危险区域管理、多级告警分发和事件录制回放。

## 功能概览

### 行为识别（10 类）

| ID | 行为 | 识别方式 | 关联规则 | 危险等级 |
|----|------|----------|----------|----------|
| 0 | 站立 | ST-GCN + 姿态几何 | — | — |
| 1 | 正常行走 | ST-GCN + 姿态几何 | — | — |
| 2 | 坐姿 | ST-GCN + 姿态几何 | — | — |
| 3 | 奔跑 | 髋/踝速度分析 | A03 | L1 注意 |
| 4 | 饮食动作 | 肘角 + 手腕距离 + 持续时间（15帧） | A01 | L2 警告 |
| 5 | 摔倒 | 头-髋速度突变 + 宽高比变化 | A05 | L3 紧急 |
| 6 | 倒地不动 | 摔倒后持续低姿态 | A05 | L3 紧急 |
| 7 | 其他操作 | 默认兜底 | — | — |
| 8 | 抽烟 | 肘角 + 手腕距离 + 持续时间 | A04 | L2 警告 |
| 9 | 推搡嬉闹 | 手腕速度爆发检测 + 滑动窗口 | A02 | L1 注意 |

**V3 推理管线**：MediaPipe 关键点 → VisibilityRouter 路由 → 6 个专用检测器 → PriorityArbiter 仲裁 → TemporalSmoother 平滑

### 实时监控

- 多路摄像头 MJPEG 实时推流
- 服务端检测框叠加渲染（PIL/OpenCV 双路径）
- 按行为类型显示不同颜色的检测框和标签
- 危险区域多边形绘制与实时闯入检测
- 人员面板：track ID、当前行为、PPE 状态、危险等级

### 告警中心

- 三级告警体系：L1 注意 / L2 警告 / L3 紧急
- WebSocket 实时推送告警消息
- 桌面 Toast 通知（L2/L3）
- 告警历史查询、筛选、批量删除
- 事件触发自动录像（预录 5s + 后录 5s）

### 数据看板

- 24 小时告警趋势图
- 告警等级分布统计
- 告警类型 TOP5
- 实时 FPS / 延迟 / 检测人数

### 系统管理

- 摄像头热添加/移除
- 危险区域可视化绘制（多边形标注）
- 预警规则启用/禁用（A 行为规则 + O 物体规则）
- 实验室信息管理

## 技术栈

### 推理引擎

| 组件 | 技术 |
|------|------|
| 人员检测 | YOLOv11n (ONNX Runtime DirectML) |
| 多目标跟踪 | RAP-SORT (改进的 SORT 算法) |
| 姿态估计 | YOLOv11n-Pose (ONNX, COCO-17 关键点) |
| 行为识别 | 启发式专用检测器 + VideoMAE + ST-GCN/CTR-GCN |
| 物体检测 | YOLO 通用物体检测 (烟雾/火焰等) |
| 关键点可见性 | VisibilityRouter (全量/受限模式路由) |
| 时序平滑 | TemporalSmoother + PriorityArbiter |

### 后端

| 组件 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 实时通信 | WebSocket (原生) |
| 数据库 | SQLite (事件 + 配置存储) |
| 视频流 | MJPEG (JPEG 帧推送) |
| 配置管理 | PyYAML (规则/摄像头配置) |
| 日志 | Loguru |

### 前端

| 组件 | 技术 |
|------|------|
| 整体架构 | 原生 HTML/CSS/JS (无框架依赖) |
| 样式方案 | CSS Variables 主题系统 + 暗色模式 |
| 视频流 | MJPEG `<img>` 标签直接渲染 |
| 画布交互 | Canvas API (区域绘制、视频回放) |
| 实时更新 | WebSocket 消息驱动 |
| 字体 | 系统中文字体 (SimHei / Microsoft YaHei) |

### 部署与工具

| 组件 | 技术 |
|------|------|
| 运行环境 | Python ≥ 3.10 |
| 推理加速 | ONNX Runtime DirectML (Windows GPU) |
| 视频编码 | OpenCV imencode (JPEG) |
| 视频录制 | 环形缓冲区 + JPEG 帧打包 (LWV1 格式) |
| 摄像头 | USB 摄像头 / 视频文件 / YAML 配置多源 |

## 快速启动

### 环境准备

```bash
pip install -r requirements.txt
```

### 启动系统

```bash
# 完整启动（使用 config.yaml 中配置的摄像头）
python main.py

# 使用 USB 摄像头
python main.py --camera usb --device 0

# 使用视频文件（循环播放）
python main.py --camera file --file test_video.mp4

# 自定义 Web 端口
python main.py --web-port 9090

# 禁用推理（仅显示原始画面）
python main.py --no-inference

# 禁用告警分发（仅推理和标注）
python main.py --no-alert

## 告警体系

| 级别 | 颜色 | 触发条件 | 响应方式 |
|------|------|----------|----------|
| L1 注意 | 橙黄 | 奔跑、推搡嬉闹、限制区域闯入 | 控制台日志 + WebSocket 推送 |
| L2 警告 | 橙色 | 饮食、抽烟、危险区域闯入 | Toast 弹窗 + 录像 + WebSocket 推送 |
| L3 紧急 | 红色 | 摔倒、倒地不动、火焰 | 持续警报 + 弹窗 + 录像 + WebSocket 推送 |

## 项目结构

```
lab-warning-system/
├── main.py                          # 程序入口，主循环
├── requirements.txt                 # Python 依赖
├── app/                             # FastAPI Web 应用
│   ├── main.py                      # 应用工厂 + WebServer
│   ├── config.py                    # 全局配置
│   └── modules/                     # 业务模块
│       ├── stream/                  # MJPEG 视频流
│       ├── alert/                   # 告警 CRUD API
│       ├── camera/                  # 摄像头管理 API
│       ├── auth/                    # 认证模块（已禁用）
│       ├── lab/                     # 实验室管理 API
│       └── dashboard/               # 数据看板 API
├── inference_engine/                # AI 推理引擎
│   ├── pipeline.py                  # InferencePipeline（流程编排）
│   ├── detection.py                 # PersonDetector（YOLOv11n）
│   ├── tracking.py                  # PersonTracker（RAP-SORT）
│   ├── pose.py                      # PoseEstimator（YOLOv11n-Pose）
│   ├── action.py                    # ActionRecognizer（入口）
│   ├── action_v3.py                 # ActionRecognizerV3（核心管线）
│   ├── action_labels.py             # 动作标签 + 颜色映射
│   ├── visibility_router.py         # VisibilityRouter（关键点路由）
│   ├── specialized_detectors.py     # 6 个专用行为检测器
│   ├── priority_arbiter.py          # PriorityArbiter（优先级仲裁）
│   ├── temporal_smoother.py         # TemporalSmoother（时序平滑）
│   ├── videomae_inference.py        # VideoMAE 视频理解模型
│   ├── ctrgc_onnx.py                # CTR-GCN ONNX 推理
│   ├── coco_to_ntu.py               # COCO → NTU 关键点转换
│   ├── temporal_resampler.py        # 时序重采样
│   ├── object_detect.py             # ObjectDetector（通用物体检测）
│   ├── ppe.py                       # PPEDetector（防护装备）
│   ├── overlay.py                   # OverlayRenderer（检测框叠加）
│   └── utils.py                     # 工具函数
├── danger_assessor/                 # 危险评估引擎
│   ├── assessor.py                  # DangerAssessor（评估入口）
│   ├── rules_engine.py              # RulesEngine（A/B/O 规则匹配）
│   ├── zones.py                     # ZoneManager（区域管理）
│   ├── smoothing.py                 # LevelSmoother（等级平滑去重）
│   └── rules/                       # 规则配置文件
│       ├── action_rules.yaml        # 行为规则（A01-A05）
│       ├── object_rules.yaml        # 物体检测规则（O01-O03）
│       └── zones.yaml               # 区域配置
├── danger_assessor/                 # 危险评估引擎
│   ├── assessor.py                  # DangerAssessor（主评估器）
│   ├── rules_engine.py              # RulesEngine（A/B/O 规则）
│   ├── zones.py                     # ZoneManager（电子围栏）
│   ├── smoothing.py                 # LevelSmoother（EMA + 去重）
│   └── rules/                       # 规则 YAML 配置文件
├── alert_service/                   # 告警分发
│   └── dispatcher.py                # AlertDispatcher（L1/L2/L3）
├── recording/                       # 事件录制
│   ├── ring_buffer.py               # 环形缓冲区
│   └── event_recorder.py            # EventRecorder（JPEG 帧打包）
├── storage/                         # 数据存储
│   └── database.py                  # EventDatabase（SQLite）
├── camera_service/                  # 摄像头服务
│   ├── camera_manager.py            # CameraManager（多源管理）
│   └── cameras/
│       ├── usb_camera.py            # USB 摄像头驱动
│       └── file_camera.py           # 视频文件驱动
└── web_api/                         # Web API
    ├── ws_manager.py                # WebSocket 管理器
    └── static/                      # 前端静态资源
        ├── index.html               # 主页面（监控/看板/告警/管理）
        └── monitor.html             # 精简监控页（备用）
```

## 模型文件

需要自行下载或训练以下模型，放置于 `models/` 目录：

| 模型 | 文件 | 用途 |
|------|------|------|
| YOLOv11n | `models/yolo11n.onnx` | 人员检测 |
| YOLOv11n-Pose | `models/yolo11n-pose.onnx` | 姿态估计 |
| VideoMAE | `models/videomae_lab_safety/` | 视频级行为理解 |
| CTR-GCN | `models/stgcn.onnx` | 骨架动作识别 |
