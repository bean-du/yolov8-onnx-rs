// 引入clap库的Parser模块，用于解析命令行参数
use clap::Parser;

// 引入本地模块中的YOLOTask枚举类型
use crate::YOLOTask;

// 定义命令行参数的结构体
#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// ONNX模型路径
    #[arg(long, required = true)]
    pub model: String,

    /// 输入路径
    #[arg(long, required = true)]
    pub source: String,

    /// 设备ID
    #[arg(long, default_value_t = 0)]
    pub device_id: u32,

    /// 是否使用TensorRT执行提供者
    #[arg(long)]
    pub trt: bool,

    /// 是否使用CUDA执行提供者
    #[arg(long)]
    pub cuda: bool,

    /// 输入批处理大小
    #[arg(long, default_value_t = 1)]
    pub batch: u32,

    /// TensorRT输入的最小批处理大小
    #[arg(long, default_value_t = 1)]
    pub batch_min: u32,

    /// TensorRT输入的最大批处理大小
    #[arg(long, default_value_t = 32)]
    pub batch_max: u32,

    /// 是否启用TensorRT的FP16模式
    #[arg(long)]
    pub fp16: bool,

    /// 指定YOLO任务类型
    #[arg(long, value_enum)]
    pub task: Option<YOLOTask>,

    /// 类别数量
    #[arg(long)]
    pub nc: Option<u32>,

    /// 关键点数量
    #[arg(long)]
    pub nk: Option<u32>,

    /// 掩码数量
    #[arg(long)]
    pub nm: Option<u32>,

    /// 输入图像的宽度
    #[arg(long)]
    pub width: Option<u32>,

    /// 输入图像的高度
    #[arg(long)]
    pub height: Option<u32>,

    /// 置信度阈值
    #[arg(long, required = false, default_value_t = 0.3)]
    pub conf: f32,

    /// 非极大值抑制中的交并比阈值
    #[arg(long, required = false, default_value_t = 0.45)]
    pub iou: f32,

    /// 关键点的置信度阈值
    #[arg(long, required = false, default_value_t = 0.55)]
    pub kconf: f32,

    /// 是否绘制推理结果并保存
    #[arg(long)]
    pub plot: bool,

    /// 是否检查每个阶段消耗的时间
    #[arg(long)]
    pub profile: bool,
}