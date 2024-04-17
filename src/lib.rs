// 允许复杂类型，这是一个编译器的lint设置
#![allow(clippy::type_complexity)]

// 引入标准库中的读写模块
use std::io::{Read, Write};

// 引入本地模块
pub mod cli; // 命令行接口模块
pub mod model; // 模型模块
pub mod ort_backend; // ONNX运行时后端模块
pub mod yolo_result; // YOLO结果模块

// 重新导出一些类型，使得其他模块可以直接使用
pub use crate::cli::Args; // 命令行参数类型
pub use crate::model::YOLOv8; // YOLOv8模型类型
pub use crate::ort_backend::{Batch, OrtBackend, OrtConfig, OrtEP, YOLOTask}; // ONNX运行时相关类型
pub use crate::yolo_result::{Bbox, Embedding, Point2, YOLOResult}; // YOLO结果相关类型

// 非极大值抑制函数，用于去除冗余的边界框
pub fn non_max_suppression(
    xs: &mut Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)>, // 输入的边界框、关键点和概率
    iou_threshold: f32, // 交并比阈值
) {
    // 根据置信度对边界框进行排序
    xs.sort_by(|b1, b2| b2.0.confidence().partial_cmp(&b1.0.confidence()).unwrap());

    let mut current_index = 0;
    for index in 0..xs.len() {
        let mut drop = false;
        for prev_index in 0..current_index {
            // 计算交并比
            let iou = xs[prev_index].0.iou(&xs[index].0);
            // 如果交并比大于阈值，则丢弃当前边界框
            if iou > iou_threshold {
                drop = true;
                break;
            }
        }
        // 如果当前边界框没有被丢弃，则将其移动到前面
        if !drop {
            xs.swap(current_index, index);
            current_index += 1;
        }
    }
    // 删除多余的边界框
    xs.truncate(current_index);
}

// 生成时间字符串的函数
pub fn gen_time_string(delimiter: &str) -> String { // 输入的分隔符
    // 设置时区为北京时间
    let offset = chrono::FixedOffset::east_opt(8 * 60 * 60).unwrap(); // Beijing
    // 获取当前时间
    let t_now = chrono::Utc::now().with_timezone(&offset);
    // 设置时间格式
    let fmt = format!(
        "%Y{}%m{}%d{}%H{}%M{}%S{}%f",
        delimiter, delimiter, delimiter, delimiter, delimiter, delimiter
    );
    // 格式化时间并返回
    t_now.format(&fmt).to_string()
}

// 定义人体骨架的连接关系
pub const SKELETON: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
];

// 检查并加载字体的函数

pub fn check_font(font: &str) -> rusttype::Font<'static> {
    // check then load font

    // ultralytics font path
    let font_path_config = match dirs::config_dir() {
        Some(mut d) => {
            d.push("Ultralytics");
            d.push(font);
            d
        }
        None => panic!("Unsupported operating system. Now support Linux, MacOS, Windows."),
    };

    // current font path
    let font_path_current = std::path::PathBuf::from(font);

    // check font
    let font_path = if font_path_config.exists() {
        font_path_config
    } else if font_path_current.exists() {
        font_path_current
    } else {
        println!("Downloading font...");
        let source_url = "https://ultralytics.com/assets/Arial.ttf";
        let resp = ureq::get(source_url)
            .timeout(std::time::Duration::from_secs(500))
            .call()
            .unwrap_or_else(|err| panic!("> Failed to download font: {source_url}: {err:?}"));

        // read to buffer
        let mut buffer = vec![];
        let total_size = resp
            .header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap();
        let _reader = resp
            .into_reader()
            .take(total_size)
            .read_to_end(&mut buffer)
            .unwrap();

        // save
        let _path = std::fs::File::create(font).unwrap();
        let mut writer = std::io::BufWriter::new(_path);
        writer.write_all(&buffer).unwrap();
        println!("Font saved at: {:?}", font_path_current.display());
        font_path_current
    };

    // load font
    let buffer = std::fs::read(font_path).unwrap();
    rusttype::Font::try_from_vec(buffer).unwrap()
}
