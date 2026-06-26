//! Run franken_ocr's wired vision_sam::forward on a raw [3,1024,1024] f32 input
//! (baidu's exact sam_in) and dump the [1024,256] SAM feature as LE f32, for
//! parity vs baidu's sam_out.npy.
use franken_ocr::native_engine::tensor::Mat;
use franken_ocr::native_engine::vision_sam;
use franken_ocr::native_engine::weights::Weights;
use std::io::{Read, Write};
use std::path::Path;

fn main() {
    let mut a = std::env::args().skip(1);
    let model = a.next().expect("model shard path");
    let sam_in = a.next().expect("sam_in.f32 path");
    let out = a.next().expect("out.f32 path");

    eprintln!("loading weights from {model} ...");
    let w = Weights::load(Path::new(&model)).expect("weights load");

    let mut buf = Vec::new();
    std::fs::File::open(&sam_in).expect("open sam_in").read_to_end(&mut buf).unwrap();
    let n = buf.len() / 4;
    let mut data = Vec::with_capacity(n);
    for c in buf.chunks_exact(4) {
        data.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    let side = ((n / 3) as f64).sqrt() as usize;
    eprintln!("sam_in n={n} -> [3, {}*{}]", side, side);
    let img = Mat::from_vec(3, side * side, data);

    let feat = vision_sam::forward(&w, &img).expect("vision_sam::forward");
    eprintln!("sam_out rows={} cols={} (expect 1024 x 256)", feat.rows, feat.cols);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    for v in &feat.data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    eprintln!("wrote {} f32 -> {out}", feat.data.len());
}
