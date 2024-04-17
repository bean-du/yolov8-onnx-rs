// 允许复杂类型，这是一个编译器的lint设置
#![allow(clippy::type_complexity)]

// 引入所需的库和模块
use anyhow::Result;
// 用于处理错误
use image::{DynamicImage, GenericImageView, ImageBuffer};
// 用于处理图像
use ndarray::{s, Array, Axis, IxDyn};
// 用于处理多维数组
use rand::{thread_rng, Rng};
// 用于生成随机数
use std::path::PathBuf; // 用于处理文件路径

use opencv::{videoio, highgui, prelude::*};
use opencv::core::{Size, split, Vec3b};
use opencv::types::VectorOfMat;
use rusttype::Font;

// 引入本地模块
use crate::{
    check_font, // 检查并加载字体的函数
    gen_time_string, // 生成时间字符串的函数
    non_max_suppression, // 非极大值抑制函数
    Args, // 命令行参数类型
    Batch, // 批处理类型
    Bbox, // 边界框类型
    Embedding, // 嵌入类型
    OrtBackend, // ONNXRuntime后端类型
    OrtConfig, // ONNXRuntime配置类型
    OrtEP, // ONNXRuntime执行提供者类型
    Point2, // 二维点类型
    YOLOResult, // YOLO结果类型
    YOLOTask, // YOLO任务类型
    SKELETON, // 人体骨架的连接关系
};

// 定义YOLOv8模型的结构体
pub struct YOLOv8 {
    // YOLOv8模型，适用于所有YOLO任务
    engine: OrtBackend,
    // ONNXRuntime后端
    nc: u32,
    // 类别数量
    nk: u32,
    // 关键点数量
    nm: u32,
    // 掩码数量
    height: u32,
    // 输入图像的高度
    width: u32,
    // 输入图像的宽度
    batch: u32,
    // 批处理大小
    task: YOLOTask,
    // YOLO任务类型
    conf: f32,
    // 置信度阈值
    kconf: f32,
    // 关键点的置信度阈值
    iou: f32,
    // 非极大值抑制中的交并比阈值
    names: Vec<String>,
    // 类别名称
    color_palette: Vec<(u8, u8, u8)>,
    // 颜色调色板
    profile: bool,
    // 是否检查每个阶段消耗的时间
    plot: bool, // 是否绘制推理结果并保存
}

impl YOLOv8 {
    // 定义新建YOLOv8模型的函数
    pub fn new(config: Args) -> Result<Self> {
        // 根据配置选择执行提供者
        let ep = if config.trt {
            OrtEP::Trt(config.device_id) // 如果选择了TensorRT，则使用TensorRT执行提供者
        } else if config.cuda {
            OrtEP::Cuda(config.device_id) // 如果选择了CUDA，则使用CUDA执行提供者
        } else {
            OrtEP::Cpu // 否则，默认使用CPU执行提供者
        };

        // 根据配置设置批处理大小
        let batch = Batch {
            opt: config.batch, // 优化批处理大小
            min: config.batch_min, // 最小批处理大小
            max: config.batch_max, // 最大批处理大小
        };

        // 构建ONNXRuntime引擎
        let ort_args = OrtConfig {
            ep, // 执行提供者
            batch, // 批处理配置
            f: config.model, // ONNX模型文件路径
            task: config.task, // YOLO任务类型
            trt_fp16: config.fp16, // 是否启用TensorRT的FP16模式
            image_size: (config.height, config.width), // 图像大小（高度，宽度）
        };
        let engine = OrtBackend::build(ort_args)?; // 构建ONNXRuntime引擎

        // 获取批处理大小、图像高度、图像宽度、任务类型
        let (batch, height, width, task) = (
            engine.batch(),
            engine.height(),
            engine.width(),
            engine.task(),
        );
        // 获取类别数量，如果获取失败，则抛出错误
        let nc = engine.nc().or(config.nc).unwrap_or_else(|| {
            panic!("Failed to get num_classes, make it explicit with `--nc`");
        });
        // 根据任务类型获取关键点数量或掩码数量
        let (nk, nm) = match task {
            YOLOTask::Pose => {
                let nk = engine.nk().or(config.nk).unwrap_or_else(|| {
                    panic!("Failed to get num_keypoints, make it explicit with `--nk`");
                });
                (nk, 0)
            }
            YOLOTask::Segment => {
                let nm = engine.nm().or(config.nm).unwrap_or_else(|| {
                    panic!("Failed to get num_masks, make it explicit with `--nm`");
                });
                (0, nm)
            }
            _ => (0, 0),
        };

        // 获取类别名称
        let names = engine.names().unwrap_or(vec!["Unknown".to_string()]);

        // 生成颜色调色板
        let mut rng = thread_rng();
        let color_palette: Vec<_> = names
            .iter()
            .map(|_| {
                (
                    rng.gen_range(0..=255), // 随机生成红色通道的值
                    rng.gen_range(0..=255), // 随机生成绿色通道的值
                    rng.gen_range(0..=255), // 随机生成蓝色通道的值
                )
            })
            .collect();

        // 返回新建的YOLOv8模型
        Ok(Self {
            engine, // ONNXRuntime引擎
            names, // 类别名称
            conf: config.conf, // 置信度阈值
            kconf: config.kconf, // 关键点的置信度阈值
            iou: config.iou, // 非极大值抑制中的交并比阈值
            color_palette, // 颜色调色板
            profile: config.profile, // 是否检查每个阶段消耗的时间
            plot: true, // 是否绘制推理结果并保存
            nc, // 类别数量
            nk, // 关键点数量
            nm, // 掩码数量
            height, // 输入图像的高度
            width, // 输入图像的宽度
            batch, // 批处理大小
            task, // YOLO任务类型
        })
    }
    // 定义一个函数，用于计算缩放后的宽度和高度
    pub fn scale_wh(&self, w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
        // 计算缩放比例，取宽度和高度的缩放比例中的最小值
        let r = (w1 / w0).min(h1 / h0);
        // 返回缩放比例，以及缩放后的宽度和高度
        (r, (w0 * r).round(), (h0 * r).round())
    }

    // 定义一个函数，用于预处理图像
    pub fn preprocess(&mut self, xs: &Vec<DynamicImage>) -> Result<Array<f32, IxDyn>> {
        // 初始化一个全为1的多维数组，用于存储预处理后的图像
        let mut ys =
            Array::ones((xs.len(), 3, self.height() as usize, self.width() as usize)).into_dyn();
        // 将数组中的所有元素填充为144.0 / 255.0
        ys.fill(144.0 / 255.0);
        // 遍历输入的图像
        for (idx, x) in xs.iter().enumerate() {
            // 根据任务类型，对图像进行不同的预处理
            let img = match self.task() {
                // 如果是分类任务，直接将图像缩放到指定大小
                YOLOTask::Classify => x.resize_exact(
                    self.width(),
                    self.height(),
                    image::imageops::FilterType::Triangle,
                ),
                // 如果是其他任务，先计算缩放后的宽度和高度，然后将图像缩放到新的大小
                _ => {
                    let (w0, h0) = x.dimensions();
                    let w0 = w0 as f32;
                    let h0 = h0 as f32;
                    let (_, w_new, h_new) =
                        self.scale_wh(w0, h0, self.width() as f32, self.height() as f32); // f32 round
                    x.resize_exact(
                        w_new as u32,
                        h_new as u32,
                        if let YOLOTask::Segment = self.task() {
                            image::imageops::FilterType::CatmullRom
                        } else {
                            image::imageops::FilterType::Triangle
                        },
                    )
                }
            };

            // 遍历图像的每个像素，将其RGB值归一化后存入数组
            for (x, y, rgb) in img.pixels() {
                let x = x as usize;
                let y = y as usize;
                let [r, g, b, _] = rgb.0;
                ys[[idx, 0, y, x]] = (r as f32) / 255.0;
                ys[[idx, 1, y, x]] = (g as f32) / 255.0;
                ys[[idx, 2, y, x]] = (b as f32) / 255.0;
            }
        }

        // 返回预处理后的图像
        Ok(ys)
    }


    pub fn run_video(&mut self, video_path: &str) -> Result<()> {
        // 创建一个视频捕获对象
        let mut capture = videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY)?;  // 0 是摄像头的设备编号

        // 检查视频是否打开
        if !videoio::VideoCapture::is_opened(&capture)? {
            return Err(anyhow::anyhow!("Cannot open the video"));
        }

        // 获取视频的FPS和帧大小
        let fps = capture.get(videoio::CAP_PROP_FPS)?;
        let frame_size = Size::new(
            capture.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32,
            capture.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32,
        );

        // 创建一个视频写入对象
        let fourcc = videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?;
        let mut writer = videoio::VideoWriter::new("output.avi", fourcc, fps, frame_size, true)?;

        // 创建一个窗口
        highgui::named_window("Video Window", highgui::WINDOW_NORMAL)?;

        let mut frame = Mat::default();
        loop {
            capture.read(&mut frame)?;

            // 如果帧是空的，跳出循环
            if frame.size()?.width <= 0 {
                break;
            }


            let mut channel: opencv::core::Vector<opencv::core::Mat> = opencv::core::Vector::new();
            split(&frame, &mut channel)?;
            let c = channel.clone();

            let data_b = c.get(0)?;
            let data_g = c.get(1)?;
            let data_r = c.get(2)?;

            let mut data = Vec::with_capacity((frame.cols() * frame.rows() * 3) as usize);
            for y in 0..frame.rows() {
                for x in 0..frame.cols() {
                    let b = *data_b.at_2d::<u8>(y, x)?;
                    let g = *data_g.at_2d::<u8>(y, x)?;
                    let r = *data_r.at_2d::<u8>(y, x)?;
                    data.push(r);
                    data.push(g);
                    data.push(b);
                }
            }


            let img_buffer = match image::ImageBuffer::from_raw(frame.cols() as u32, frame.rows() as u32, data) {
                Some(img) => img,
                None => return Err(anyhow::anyhow!("can not create image from ndarray")),
            };

            let img = DynamicImage::ImageRgb8(img_buffer);
            let ys = self.run(&vec![img.clone()])?;


            let plot_img = self.plot(&check_font("Arial.ttf"), &ys, &img, Some(&SKELETON))?;
            self.save(&plot_img, String::from("runs"))?;

            let rgb_img = plot_img.to_rgb8();
            let (w, h) = rgb_img.dimensions();

            let mut r_c = vec![0; (w * h) as usize];
            let mut g_c = vec![0; (w * h) as usize];
            let mut b_c = vec![0; (w * h) as usize];

            for (x, y, pixel) in rgb_img.enumerate_pixels() {
                let index = (y * w + x) as usize;
                let r = pixel[0];
                let g = pixel[1];
                let b = pixel[2];
                r_c[index] = r;
                g_c[index] = g;
                b_c[index] = b;
            }

            let r_2d = r_c.chunks(w as usize).map(|x| x.to_vec()).collect::<Vec<Vec<u8>>>();
            let g_2d = g_c.chunks(w as usize).map(|x| x.to_vec()).collect::<Vec<Vec<u8>>>();
            let b_2d = b_c.chunks(w as usize).map(|x| x.to_vec()).collect::<Vec<Vec<u8>>>();


            let r_mat = Mat::from_slice_2d(&r_2d)?;
            let g_mat = Mat::from_slice_2d(&g_2d)?;
            let b_mat = Mat::from_slice_2d(&b_2d)?;

            let mut v_frame = VectorOfMat::new();
            v_frame.push(b_mat);
            v_frame.push(g_mat);
            v_frame.push(r_mat);

            let mut frame = Mat::default();
            opencv::core::merge(&v_frame, &mut frame)?;

            // 将处理后的图像写入新的视频文件
            writer.write(&frame)?;

            // 在窗口中显示图像
            highgui::imshow("Video Window", &frame)?;

            // 等待1毫秒，如果在这期间用户按下任何键，就退出循环
            if highgui::wait_key(1)? > 0 {
                break;
            }
        }

        Ok(())
    }

    pub fn run_image(&mut self, img: &DynamicImage) -> Result<YOLOResult> {
        let ys = self.run(&vec![img.clone()])?;
        Ok(ys[0].clone())
    }

    // 定义一个函数，用于运行模型
    pub fn run(&mut self, xs: &Vec<DynamicImage>) -> Result<Vec<YOLOResult>> {
        // 记录预处理的开始时间
        let t_pre = std::time::Instant::now();
        // 对输入的图像进行预处理
        let xs_ = self.preprocess(xs)?;
        // 如果开启了性能分析，打印预处理的耗时
        if self.profile {
            println!("[Model Preprocess]: {:?}", t_pre.elapsed());
        }

        // 记录模型推理的开始时间
        let t_run = std::time::Instant::now();
        // 运行模型，获取推理结果
        let ys = self.engine.run(xs_, self.profile)?;
        // 如果开启了性能分析，打印模型推理的耗时
        if self.profile {
            println!("[Model Inference]: {:?}", t_run.elapsed());
        }

        // 记录后处理的开始时间
        let t_post = std::time::Instant::now();
        // 对推理结果进行后处理
        let ys = self.postprocess(ys, xs)?;
        // 如果开启了性能分析，打印后处理的耗时
        if self.profile {
            println!("[Model Postprocess]: {:?}", t_post.elapsed());
        }

        // 如果开启了结果绘制，绘制并保存推理结果
        // if self.plot {
        //     self.plot_and_save(&ys, xs, Some(&SKELETON));
        // }
        // 返回推理结果
        Ok(ys)
    }
    // 定义一个函数，用于对模型的推理结果进行后处理
    pub fn postprocess(
        &self,
        xs: Vec<Array<f32, IxDyn>>, // 输入的推理结果，是一个多维数组的向量
        xs0: &[DynamicImage], // 输入的原始图像，是一个动态图像的引用
    ) -> Result<Vec<YOLOResult>> { // 返回处理后的结果，是一个YOLO结果的向量
        // 判断任务类型是否为分类任务
        if let YOLOTask::Classify = self.task() {
            let mut ys = Vec::new(); // 初始化一个空的向量，用于存储处理后的结果
            let preds = &xs[0]; // 获取推理结果的第一个元素
            // 遍历推理结果的每个批次
            for batch in preds.axis_iter(Axis(0)) {
                // 将批次的数据转换为嵌入类型，并添加到结果向量中
                ys.push(YOLOResult::new(
                    Some(Embedding::new(batch.into_owned())),
                    None,
                    None,
                    None,
                ));
            }
            // 返回处理后的结果
            Ok(ys)
        } else {
            // 定义常量，表示边界框和关键点的偏移量
            const CXYWH_OFFSET: usize = 4; // cxcywh
            const KPT_STEP: usize = 3; // xyconf
            let preds = &xs[0]; // 获取推理结果的第一个元素
            // 判断是否存在第二个元素
            let protos = {
                if xs.len() > 1 {
                    Some(&xs[1]) // 如果存在，则获取第二个元素
                } else {
                    None // 如果不存在，则返回None
                }
            };
            let mut ys = Vec::new(); // 初始化一个空的向量，用于存储处理后的结果
            // 遍历推理结果的每个锚点
            for (idx, anchor) in preds.axis_iter(Axis(0)).enumerate() {
                // 获取输入图像的原始宽度和高度
                let width_original = xs0[idx].width() as f32;
                let height_original = xs0[idx].height() as f32;
                // 计算缩放比例
                let ratio = (self.width() as f32 / width_original)
                    .min(self.height() as f32 / height_original);

                // 初始化一个空的向量，用于存储每个结果的数据
                let mut data: Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)> = Vec::new();
                // 遍历锚点的每个预测结果
                for pred in anchor.axis_iter(Axis(1)) {
                    // 分割预测结果，获取边界框、类别、关键点和系数
                    let bbox = pred.slice(s![0..CXYWH_OFFSET]);
                    let clss = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + self.nc() as usize]);
                    let kpts = {
                        if let YOLOTask::Pose = self.task() {
                            Some(pred.slice(s![pred.len() - KPT_STEP * self.nk() as usize..]))
                        } else {
                            None
                        }
                    };
                    let coefs = {
                        if let YOLOTask::Segment = self.task() {
                            Some(pred.slice(s![pred.len() - self.nm() as usize..]).to_vec())
                        } else {
                            None
                        }
                    };

                    // 计算置信度和类别ID
                    let (id, &confidence) = clss
                        .into_iter()
                        .enumerate()
                        .reduce(|max, x| if x.1 > max.1 { x } else { max })
                        .unwrap(); // definitely will not panic!

                    // 如果置信度低于阈值，则跳过当前预测结果
                    if confidence < self.conf {
                        continue;
                    }

                    // 对边界框进行缩放
                    let cx = bbox[0] / ratio;
                    let cy = bbox[1] / ratio;
                    let w = bbox[2] / ratio;
                    let h = bbox[3] / ratio;
                    let x = cx - w / 2.;
                    let y = cy - h / 2.;
                    let y_bbox = Bbox::new(
                        x.max(0.0f32).min(width_original),
                        y.max(0.0f32).min(height_original),
                        w,
                        h,
                        id,
                        confidence,
                    );

                    // 对关键点进行缩放
                    let y_kpts = {
                        if let Some(kpts) = kpts {
                            let mut kpts_ = Vec::new();
                            // rescale
                            for i in 0..self.nk() as usize {
                                let kx = kpts[KPT_STEP * i] / ratio;
                                let ky = kpts[KPT_STEP * i + 1] / ratio;
                                let kconf = kpts[KPT_STEP * i + 2];
                                if kconf < self.kconf {
                                    kpts_.push(Point2::default());
                                } else {
                                    kpts_.push(Point2::new_with_conf(
                                        kx.max(0.0f32).min(width_original),
                                        ky.max(0.0f32).min(height_original),
                                        kconf,
                                    ));
                                }
                            }
                            Some(kpts_)
                        } else {
                            None
                        }
                    };

                    // 将数据合并到结果向量中
                    data.push((y_bbox, y_kpts, coefs));
                }

                // 对结果进行非极大值抑制
                non_max_suppression(&mut data, self.iou);

                // 解码结果
                let mut y_bboxes: Vec<Bbox> = Vec::new();
                let mut y_kpts: Vec<Vec<Point2>> = Vec::new();
                let mut y_masks: Vec<Vec<u8>> = Vec::new();
                for elem in data.into_iter() {
                    if let Some(kpts) = elem.1 {
                        y_kpts.push(kpts)
                    }

                    // 解码掩码
                    if let Some(coefs) = elem.2 {
                        let proto = protos.unwrap().slice(s![idx, .., .., ..]);
                        let (nm, nh, nw) = proto.dim();

                        // coefs * proto -> mask
                        let coefs = Array::from_shape_vec((1, nm), coefs)?; // (n, nm)
                        let proto = proto.to_owned().into_shape((nm, nh * nw))?; // (nm, nh*nw)
                        let mask = coefs.dot(&proto).into_shape((nh, nw, 1))?; // (nh, nw, n)

                        // build image from ndarray
                        let mask_im: ImageBuffer<image::Luma<_>, Vec<f32>> =
                            match ImageBuffer::from_raw(nw as u32, nh as u32, mask.into_raw_vec()) {
                                Some(image) => image,
                                None => panic!("can not create image from ndarray"),
                            };
                        let mut mask_im = image::DynamicImage::from(mask_im); // -> dyn

                        // rescale masks
                        let (_, w_mask, h_mask) =
                            self.scale_wh(width_original, height_original, nw as f32, nh as f32);
                        let mask_cropped = mask_im.crop(0, 0, w_mask as u32, h_mask as u32);
                        let mask_original = mask_cropped.resize_exact(
                            // resize_to_fill
                            width_original as u32,
                            height_original as u32,
                            match self.task() {
                                YOLOTask::Segment => image::imageops::FilterType::CatmullRom,
                                _ => image::imageops::FilterType::Triangle,
                            },
                        );

                        // crop-mask with bbox
                        let mut mask_original_cropped = mask_original.into_luma8();
                        for y in 0..height_original as usize {
                            for x in 0..width_original as usize {
                                if x < elem.0.xmin() as usize
                                    || x > elem.0.xmax() as usize
                                    || y < elem.0.ymin() as usize
                                    || y > elem.0.ymax() as usize
                                {
                                    mask_original_cropped.put_pixel(
                                        x as u32,
                                        y as u32,
                                        image::Luma([0u8]),
                                    );
                                }
                            }
                        }
                        y_masks.push(mask_original_cropped.into_raw());
                    }
                    y_bboxes.push(elem.0);
                }

                // 保存每个结果
                let y = YOLOResult {
                    probs: None,
                    bboxes: if !y_bboxes.is_empty() {
                        Some(y_bboxes)
                    } else {
                        None
                    },
                    keypoints: if !y_kpts.is_empty() {
                        Some(y_kpts)
                    } else {
                        None
                    },
                    masks: if !y_masks.is_empty() {
                        Some(y_masks)
                    } else {
                        None
                    },
                };
                ys.push(y);
            }

            // 返回处理后的结果
            Ok(ys)
        }
    }

    pub fn plot(&self, font: &Font, ys: &[YOLOResult], xs: &DynamicImage, skeletons: Option<&[(usize, usize)]>) -> Result<DynamicImage> {
        // 检查并加载字体
        // let font = check_font("Arial.ttf");
        // 将原始图像转换为RGB8格式
        let mut img = xs.to_rgb8();

        let res = match ys.get(0) {
            Some(res) => res,
            None => return Err(anyhow::anyhow!("No YOLO-Result to plot")),
        };


        // 如果YOLO结果中存在概率，则绘制分类结果
        if let Some(probs) = res.probs() {
            // 遍历概率的前5个元素
            for (i, k) in probs.topk(5).iter().enumerate() {
                // 格式化类别名称和概率
                let legend = format!("{} {:.2}%", self.names[k.0], k.1);
                // 设置图例的大小
                let scale = 32;
                let legend_size = img.width().max(img.height()) / scale;
                // 设置图例的位置
                let x = img.width() / 20;
                let y = img.height() / 20 + i as u32 * legend_size;
                // 在图像上绘制图例
                imageproc::drawing::draw_text_mut(
                    &mut img,
                    image::Rgb([0, 255, 0]),
                    x as i32,
                    y as i32,
                    rusttype::Scale::uniform(legend_size as f32 - 1.),
                    &font,
                    &legend,
                );
            }
        }

        // 如果YOLO结果中存在边界框，则绘制边界框和关键点
        if let Some(bboxes) = res.bboxes() {
            // 遍历所有的边界框
            for (_idx, bbox) in bboxes.iter().enumerate() {
                // 在图像上绘制边界框
                imageproc::drawing::draw_hollow_rect_mut(
                    &mut img,
                    imageproc::rect::Rect::at(bbox.xmin() as i32, bbox.ymin() as i32)
                        .of_size(bbox.width() as u32, bbox.height() as u32),
                    image::Rgb(self.color_palette[bbox.id()].into()),
                );

                // 格式化类别名称和置信度
                let legend = format!("{} {:.2}%", self.names[bbox.id()], bbox.confidence());
                // 设置图例的大小
                let scale = 40;
                let legend_size = img.width().max(img.height()) / scale;
                // 在图像上绘制图例
                imageproc::drawing::draw_text_mut(
                    &mut img,
                    image::Rgb(self.color_palette[bbox.id()].into()),
                    bbox.xmin() as i32,
                    (bbox.ymin() - legend_size as f32) as i32,
                    rusttype::Scale::uniform(legend_size as f32 - 1.),
                    &font,
                    &legend,
                );
            }
        }

        // 如果YOLO结果中存在关键点，则绘制关键点
        if let Some(keypoints) = res.keypoints() {
            // 遍历所有的关键点
            for kpts in keypoints.iter() {
                for kpt in kpts.iter() {
                    // 如果关键点的置信度小于设定的阈值，则跳过此关键点
                    if kpt.confidence() < self.kconf {
                        continue;
                    }

                    // 在图像上绘制关键点
                    imageproc::drawing::draw_filled_circle_mut(
                        &mut img,
                        (kpt.x() as i32, kpt.y() as i32),
                        2,
                        image::Rgb([0, 255, 0]),
                    );
                }

                // 如果存在骨架连接关系，则绘制骨架
                if let Some(skeletons) = skeletons {
                    for &(idx1, idx2) in skeletons.iter() {
                        let kpt1 = &kpts[idx1];
                        let kpt2 = &kpts[idx2];
                        // 如果关键点的置信度小于设定的阈值，则跳过此关键点
                        if kpt1.confidence() < self.kconf || kpt2.confidence() < self.kconf {
                            continue;
                        }
                        // 在图像上绘制骨架
                        imageproc::drawing::draw_line_segment_mut(
                            &mut img,
                            (kpt1.x(), kpt1.y()),
                            (kpt2.x(), kpt2.y()),
                            image::Rgb([233, 14, 57]),
                        );
                    }
                }
            }
        }

        // 如果YOLO结果中存在掩码，则绘制掩码
        if let Some(masks) = res.masks() {
            for (mask, _bbox) in masks.iter().zip(res.bboxes().unwrap().iter()) {
                // 将掩码转换为图像
                let mask_nd: ImageBuffer<image::Luma<_>, Vec<u8>> =
                    match ImageBuffer::from_vec(img.width(), img.height(), mask.to_vec()) {
                        Some(image) => image,
                        None => panic!("can not crate image from ndarray"),
                    };

                // 遍历图像的每个像素，如果掩码对应的像素值大于0，则修改图像的像素值
                for _x in 0..img.width() {
                    for _y in 0..img.height() {
                        let mask_p = imageproc::drawing::Canvas::get_pixel(&mask_nd, _x, _y);
                        if mask_p.0[0] > 0 {
                            let mut img_p = imageproc::drawing::Canvas::get_pixel(&img, _x, _y);
                            img_p.0[2] /= 2;
                            img_p.0[1] = 255 - (255 - img_p.0[2]) / 2;
                            img_p.0[0] /= 2;
                            imageproc::drawing::Canvas::draw_pixel(&mut img, _x, _y, img_p)
                        }
                    }
                }
            }
        }


        let img = DynamicImage::ImageRgb8(img);
        Ok(img)
    }

    // 定义一个函数，用于绘制并保存推理结果
    pub fn plot_and_save(
        &self,
        ys: &[YOLOResult], // 输入的YOLO结果，是一个YOLO结果的引用
        xs0: &[DynamicImage], // 输入的原始图像，是一个动态图像的引用
        skeletons: Option<&[(usize, usize)]>, // 输入的骨架连接关系，是一个可选的元组的引用
    ) {
        // 检查并加载字体
        let font = check_font("Arial.ttf");
        // 遍历输入的原始图像和YOLO结果
        for (_idb, (img0, y)) in xs0.iter().zip(ys.iter()).enumerate() {
            // 对每个原始图像和YOLO结果进行绘制
            let img = match self.plot(&font, ys, img0, skeletons){
                Ok(img) => img,
                Err(e) => {
                    eprintln!("Error: {:?}", e);
                    continue;
                }
            };

            // 如果不存在runs目录，则创建runs目录
            let mut runs = PathBuf::from("runs");
            if !runs.exists() {
                std::fs::create_dir_all(&runs).unwrap();
            }
            // 生成保存的文件名
            runs.push(gen_time_string("-"));
            let saveout = format!("{}.jpg", runs.to_str().unwrap());
            // 保存图像
            let _ = img.save(saveout);
        }
    }

    pub fn save(&self, img: &DynamicImage, path: String)-> Result<()> {
        // 如果不存在runs目录，则创建runs目录
        let mut runs = PathBuf::from(path);
        if !runs.exists() {
            std::fs::create_dir_all(&runs).unwrap();
        }
        // 生成保存的文件名
        runs.push(gen_time_string("-"));
        let saveout = format!("{}.jpg", runs.to_str().unwrap());

        // 保存图像
        match img.save(saveout) {
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow::anyhow!("Error: {:?}", e)),
        }
    }

    // 定义一个函数，用于打印模型的摘要信息
    pub fn summary(&self) {
        // 使用println!宏打印模型的摘要信息
        println!(
            "\nSummary:\n\
        > Task: {:?}{}\n\
        > EP: {:?} {}\n\
        > Dtype: {:?}\n\
        > Batch: {} ({}), Height: {} ({}), Width: {} ({})\n\
        > nc: {} nk: {}, nm: {}, conf: {}, kconf: {}, iou: {}\n\
        ",
            // 打印任务类型
            self.task(),
            // 打印模型的作者和版本信息
            match self.engine.author().zip(self.engine.version()) {
                Some((author, ver)) => format!(" ({} {})", author, ver),
                None => String::from(""),
            },
            // 打印执行提供者类型
            self.engine.ep(),
            // 如果执行提供者是CPU，打印空字符串；否则，打印"May still fall back to CPU"
            if let OrtEP::Cpu = self.engine.ep() {
                ""
            } else {
                "(May still fall back to CPU)"
            },
            // 打印数据类型
            self.engine.dtype(),
            // 打印批处理大小
            self.batch(),
            // 如果批处理大小是动态的，打印"Dynamic"；否则，打印"Const"
            if self.engine.is_batch_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            // 打印输入图像的高度
            self.height(),
            // 如果输入图像的高度是动态的，打印"Dynamic"；否则，打印"Const"
            if self.engine.is_height_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            // 打印输入图像的宽度
            self.width(),
            // 如果输入图像的宽度是动态的，打印"Dynamic"；否则，打印"Const"
            if self.engine.is_width_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            // 打印类别数量
            self.nc(),
            // 打印关键点数量
            self.nk(),
            // 打印掩码数量
            self.nm(),
            // 打印置信度阈值
            self.conf,
            // 打印关键点的置信度阈值
            self.kconf,
            // 打印非极大值抑制中的交并比阈值
            self.iou,
        );
    }

    // 定义一个函数，用于获取ONNXRuntime后端
    pub fn engine(&self) -> &OrtBackend {
        // 返回ONNXRuntime后端的引用
        &self.engine
    }

    // 定义一个函数，用于获取置信度阈值
    pub fn conf(&self) -> f32 {
        // 返回置信度阈值
        self.conf
    }

    // 定义一个函数，用于设置置信度阈值
    pub fn set_conf(&mut self, val: f32) {
        // 设置置信度阈值
        self.conf = val;
    }

    // 定义一个函数，用于获取置信度阈值的可变引用
    pub fn conf_mut(&mut self) -> &mut f32 {
        // 返回置信度阈值的可变引用
        &mut self.conf
    }

    // 定义一个函数，用于获取关键点的置信度阈值
    pub fn kconf(&self) -> f32 {
        // 返回关键点的置信度阈值
        self.kconf
    }

    // 定义一个函数，用于获取非极大值抑制中的交并比阈值
    pub fn iou(&self) -> f32 {
        // 返回非极大值抑制中的交并比阈值
        self.iou
    }

    // 定义一个函数，用于获取YOLO任务类型
    pub fn task(&self) -> &YOLOTask {
        // 返回YOLO任务类型的引用
        &self.task
    }

    // 定义一个函数，用于获取批处理大小
    pub fn batch(&self) -> u32 {
        // 返回批处理大小
        self.batch
    }

    // 定义一个函数，用于获取输入图像的宽度
    pub fn width(&self) -> u32 {
        // 返回输入图像的宽度
        self.width
    }

    // 定义一个函数，用于获取输入图像的高度
    pub fn height(&self) -> u32 {
        // 返回输入图像的高度
        self.height
    }

    // 定义一个函数，用于获取类别数量
    pub fn nc(&self) -> u32 {
        // 返回类别数量
        self.nc
    }

    // 定义一个函数，用于获取关键点数量
    pub fn nk(&self) -> u32 {
        // 返回关键点数量
        self.nk
    }

    // 定义一个函数，用于获取掩码数量
    pub fn nm(&self) -> u32 {
        // 返回掩码数量
        self.nm
    }

    // 定义一个函数，用于获取类别名称
    pub fn names(&self) -> &Vec<String> {
        // 返回类别名称的引用
        &self.names
    }
}
