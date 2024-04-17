// 引入ndarray库，用于处理多维数组
use ndarray::{Array, Axis, IxDyn};

// 定义YOLO的结果结构体
#[derive(Clone, PartialEq, Default)]
pub struct YOLOResult {
    // YOLO任务的结果
    pub probs: Option<Embedding>,       // 概率
    pub bboxes: Option<Vec<Bbox>>,      // 边界框
    pub keypoints: Option<Vec<Vec<Point2>>>, // 关键点
    pub masks: Option<Vec<Vec<u8>>>,    // 掩码
}

// 为YOLOResult实现Debug trait，方便打印和调试
impl std::fmt::Debug for YOLOResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("YOLOResult")
            .field(
                "Probs(top5)",
                &format_args!("{:?}", self.probs().map(|probs| probs.topk(5))),
            )
            .field("Bboxes", &self.bboxes)
            .field("Keypoints", &self.keypoints)
            .field(
                "Masks",
                &format_args!("{:?}", self.masks().map(|masks| masks.len())),
            )
            .finish()
    }
}

// 为YOLOResult实现一些方法
impl YOLOResult {
    // 新建一个YOLOResult实例
    pub fn new(
        probs: Option<Embedding>,
        bboxes: Option<Vec<Bbox>>,
        keypoints: Option<Vec<Vec<Point2>>>,
        masks: Option<Vec<Vec<u8>>>,
    ) -> Self {
        Self {
            probs,
            bboxes,
            keypoints,
            masks,
        }
    }

    // 获取概率
    pub fn probs(&self) -> Option<&Embedding> {
        self.probs.as_ref()
    }

    // 获取关键点
    pub fn keypoints(&self) -> Option<&Vec<Vec<Point2>>> {
        self.keypoints.as_ref()
    }

    // 获取掩码
    pub fn masks(&self) -> Option<&Vec<Vec<u8>>> {
        self.masks.as_ref()
    }

    // 获取边界框
    pub fn bboxes(&self) -> Option<&Vec<Bbox>> {
        self.bboxes.as_ref()
    }

    // 获取边界框的可变引用
    pub fn bboxes_mut(&mut self) -> Option<&mut Vec<Bbox>> {
        self.bboxes.as_mut()
    }
}

// 定义二维点的结构体
#[derive(Debug, PartialEq, Clone, Default)]
pub struct Point2 {
    // 一个二维点，包含x、y坐标和置信度
    x: f32,
    y: f32,
    confidence: f32,
}

// 为Point2实现一些方法
impl Point2 {
    // 新建一个带有置信度的Point2实例
    pub fn new_with_conf(x: f32, y: f32, confidence: f32) -> Self {
        Self { x, y, confidence }
    }

    // 新建一个Point2实例
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            ..Default::default()
        }
    }

    // 获取x坐标
    pub fn x(&self) -> f32 {
        self.x
    }

    // 获取y坐标
    pub fn y(&self) -> f32 {
        self.y
    }

    // 获取置信度
    pub fn confidence(&self) -> f32 {
        self.confidence
    }
}

// 定义嵌入的结构体
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Embedding {
    // 一个浮点数的n维张量
    data: Array<f32, IxDyn>,
}

// 为Embedding实现一些方法
impl Embedding {
    // 新建一个Embedding实例
    pub fn new(data: Array<f32, IxDyn>) -> Self {
        Self { data }
    }

    // 获取数据
    pub fn data(&self) -> &Array<f32, IxDyn> {
        &self.data
    }

    // 获取前k个最大的元素
    pub fn topk(&self, k: usize) -> Vec<(usize, f32)> {
        let mut probs = self
            .data
            .iter()
            .enumerate()
            .map(|(a, b)| (a, *b))
            .collect::<Vec<_>>();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut topk = Vec::new();
        for &(id, confidence) in probs.iter().take(k) {
            topk.push((id, confidence));
        }
        topk
    }

    // 归一化
    pub fn norm(&self) -> Array<f32, IxDyn> {
        let std_ = self.data.mapv(|x| x * x).sum_axis(Axis(0)).mapv(f32::sqrt);
        self.data.clone() / std_
    }

    // 获取最大的元素
    pub fn top1(&self) -> (usize, f32) {
        self.topk(1)[0]
    }
}

// 定义边界框的结构体
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Bbox {
    // 一个围绕物体的边界框
    xmin: f32,
    ymin: f32,
    width: f32,
    height: f32,
    id: usize,
    confidence: f32,
}

// 为Bbox实现一些方法
impl Bbox {
    // 从xywh新建一个Bbox实例
    pub fn new_from_xywh(xmin: f32, ymin: f32, width: f32, height: f32) -> Self {
        Self {
            xmin,
            ymin,
            width,
            height,
            ..Default::default()
        }
    }

    // 新建一个Bbox实例
    pub fn new(xmin: f32, ymin: f32, width: f32, height: f32, id: usize, confidence: f32) -> Self {
        Self {
            xmin,
            ymin,
            width,
            height,
            id,
            confidence,
        }
    }

    // 获取宽度
    pub fn width(&self) -> f32 {
        self.width
    }

    // 获取高度
    pub fn height(&self) -> f32 {
        self.height
    }

    // 获取xmin
    pub fn xmin(&self) -> f32 {
        self.xmin
    }

    // 获取ymin
    pub fn ymin(&self) -> f32 {
        self.ymin
    }

    // 获取xmax
    pub fn xmax(&self) -> f32 {
        self.xmin + self.width
    }

    // 获取ymax
    pub fn ymax(&self) -> f32 {
        self.ymin + self.height
    }

    // 获取左上角的点
    pub fn tl(&self) -> Point2 {
        Point2::new(self.xmin, self.ymin)
    }

    // 获取右下角的点
    pub fn br(&self) -> Point2 {
        Point2::new(self.xmax(), self.ymax())
    }

    // 获取中心点
    pub fn cxcy(&self) -> Point2 {
        Point2::new(self.xmin + self.width / 2., self.ymin + self.height / 2.)
    }

    // 获取id
    pub fn id(&self) -> usize {
        self.id
    }

    // 获取置信度
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    // 获取面积
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    // 获取与另一个边界框的交集面积
    pub fn intersection_area(&self, another: &Bbox) -> f32 {
        let l = self.xmin.max(another.xmin);
        let r = (self.xmin + self.width).min(another.xmin + another.width);
        let t = self.ymin.max(another.ymin);
        let b = (self.ymin + self.height).min(another.ymin + another.height);
        (r - l + 1.).max(0.) * (b - t + 1.).max(0.)
    }

    // 获取与另一个边界框的并集
    pub fn union(&self, another: &Bbox) -> f32 {
        self.area() + another.area() - self.intersection_area(another)
    }

    // 获取与另一个边界框的交并比
    pub fn iou(&self, another: &Bbox) -> f32 {
        self.intersection_area(another) / self.union(another)
    }
}