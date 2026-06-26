//! Full franken_ocr vision tower (SAM -> CLIP -> projector bridge) on baidu's
//! exact sam_in, dumping CLIP [257,1024] and projector [256,1280] for parity.
use franken_ocr::native_engine::tensor::Mat;
use franken_ocr::native_engine::{vision_bridge, vision_clip, vision_sam};
use franken_ocr::native_engine::weights::Weights;
use std::io::{Read, Write};
use std::path::Path;

fn dump(path: &str, m: &Mat) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for v in &m.data { f.write_all(&v.to_le_bytes()).unwrap(); }
    eprintln!("  wrote [{}, {}] -> {path}", m.rows, m.cols);
}

fn main() {
    let mut a = std::env::args().skip(1);
    let model = a.next().expect("model");
    let sam_in = a.next().expect("sam_in.f32");
    let clip_out = a.next().expect("clip_out.f32");
    let bridge_out = a.next().expect("bridge_out.f32");

    eprintln!("loading weights ...");
    let w = Weights::load(Path::new(&model)).expect("weights");
    let mut buf = Vec::new();
    std::fs::File::open(&sam_in).unwrap().read_to_end(&mut buf).unwrap();
    let data: Vec<f32> = buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    let side = ((data.len()/3) as f64).sqrt() as usize;
    let img = Mat::from_vec(3, side*side, data);

    eprintln!("SAM ...");
    let sam = vision_sam::forward(&w, &img).expect("sam");
    eprintln!("  sam [{},{}]", sam.rows, sam.cols);
    eprintln!("CLIP ...");
    let clip = vision_clip::forward(&w, &img, &sam).expect("clip");
    eprintln!("  clip [{},{}]", clip.rows, clip.cols);
    eprintln!("bridge ...");
    let bridge = vision_bridge::forward(&w, &clip, &sam).expect("bridge");
    eprintln!("  bridge [{},{}]", bridge.rows, bridge.cols);

    dump(&clip_out, &clip);
    dump(&bridge_out, &bridge);
    eprintln!("FULL_VISION_DONE");
}
