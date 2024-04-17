// 引入clap库的Parser模块，用于解析命令行参数
use clap::Parser;

// 引入本地模块中的Args和YOLOv8
use yolov8_rs::{Args, YOLOv8};

// 主函数，返回一个Result类型，如果有错误，错误类型为Box<dyn std::error::Error>
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 解析命令行参数
    let args = Args::parse();

    // 1. 构建YOLOv8模型
    // 使用Args中的参数新建一个YOLOv8模型实例
    let mut model = YOLOv8::new(args.clone())?;
    // 打印模型的摘要信息
    model.summary();

    if args.source.contains("rtsp://") || args.source.contains("rtmp://") {
        println!("run rtsp or rtmp video inference.");
        // let url = "rtsp://192.168.1.56:554/ch01.264";
        // let url = "rtsp://192.168.2.202:8554/zlm/001";
        let ys = model.run_video(&args.source)?;
    } else {
        // 1. 加载图像
        // 使用image库的io::Reader模块打开图像文件，并尝试猜测图像的格式，然后解码图像
        let x = image::io::Reader::open(&args.source)?
            .with_guessed_format()?
            .decode()?;

        // 2. 模型支持动态批量推理，所以输入应该是一个Vec
        // 创建一个包含图像x的Vec

        let ys = model.run_image(&x)?;
        // 你可以通过这种方式测试 `--batch 2`
        // let xs = vec![x.clone(), x];

        // 打印推理结果
        println!("{:?}", ys);

    }
    Ok(())
}