//! `focr` / `franken_ocr` CLI entrypoint.
//!
//! `fn main()` is **synchronous by design** (plan §3.3, §7.1): the asupersync
//! runtime is owned BELOW main, inside `OcrEngine`, never spanning the whole
//! process. main parses, dispatches, and maps errors to stable exit codes.
#![forbid(unsafe_code)]

mod cli;

use clap::Parser;
use std::process::ExitCode;

fn main() -> ExitCode {
    let cli = cli::Cli::parse();
    match cli::run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("focr: {err}");
            ExitCode::from(err.exit_code() as u8)
        }
    }
}
