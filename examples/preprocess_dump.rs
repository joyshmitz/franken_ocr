//! Dump franken_ocr's Base(1024) preprocessed global-view tensor as raw LE f32
//! ([3,1024,1024], channel-major) for stage-0 parity vs baidu's image_ori.
use franken_ocr::preprocess::{preprocess_image, PreprocessMode};
use std::io::Write;
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let page = args.next().expect("usage: preprocess_dump <page.png> <out.f32>");
    let out = args.next().expect("usage: preprocess_dump <page.png> <out.f32>");
    let pre = preprocess_image(Path::new(&page), PreprocessMode::base()).expect("preprocess failed");
    let m = &pre.global.pixels;
    eprintln!("global.pixels rows={} cols={} (expect 3 x 1048576)", m.rows, m.cols);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).expect("create out"));
    for v in &m.data {
        f.write_all(&v.to_le_bytes()).expect("write");
    }
    eprintln!("wrote {} f32 -> {}", m.data.len(), out);
}
