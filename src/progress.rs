//! Human progress reporting for long OCR runs (GitHub issue #2).
//!
//! One self-contained, dependency-free progress line on STDERR for the
//! long-running drivers: the per-page PDF loops, the `--multi-page` raster +
//! cross-page decode, and the sequential `ocr-batch` loop. A big scanned book
//! is minutes of silent compute otherwise; the bar answers "is it alive, how
//! far along, and roughly how long to go" (done/total, percent, elapsed, and a
//! simple linear ETA).
//!
//! Design constraints, in priority order:
//!
//! 1. **Machine output is sacred.** focr's robot NDJSON stream and `--json`
//!    outputs are consumed by agents; a progress bar must never be able to
//!    corrupt them. Three independent guards enforce that:
//!    * the bar writes ONLY to stderr — stdout is never touched;
//!    * callers construct it explicitly disabled in `--robot` / `--json`
//!      modes (the `human_mode` argument to [`Progress::new`]);
//!    * it auto-disables when stderr is not an interactive terminal, so a
//!      piped or redirected run sees the exact byte stream it always saw
//!      (`\r` redraw sequences never land in a log file).
//! 2. **No new dependencies.** The repo's dependency block is deliberately
//!    minimal and every entry is justified; a `\r`-redraw line over
//!    `std::io::stderr` needs none of `indicatif`'s machinery. ASCII-only
//!    output, so it renders on any terminal.
//! 3. **Explicit kill switch.** `FOCR_NO_PROGRESS=1` (any value other than
//!    empty/`0`) disables the bar even on a TTY, and `TERM=dumb` terminals
//!    are respected — same conventions the wider CLI ecosystem uses.
//!
//! When the bar is disabled every method is a no-op and [`Progress::note`]
//! degrades to a plain `eprintln!`, so non-TTY stderr output is byte-identical
//! to what the CLI printed before this module existed.

use std::io::{IsTerminal, Write};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

/// Environment kill switch: set (to anything but empty/`0`) to suppress the
/// bar even when stderr is an interactive terminal.
pub const NO_PROGRESS_ENV: &str = "FOCR_NO_PROGRESS";

/// Width of the `[####----]` fill region, in cells.
const BAR_CELLS: usize = 24;

/// Mutable bar state behind the [`Progress`] handle. `Arc` so a boxed
/// `'static` page sink (see [`Progress::page_sink`]) can advance the same bar
/// the owning driver renders.
struct Inner {
    /// Short driver tag rendered after `[focr]` (e.g. `ocr`, `raster`, `batch`).
    label: &'static str,
    /// Total number of work items (pages / images).
    total: usize,
    /// Completed work items.
    done: usize,
    /// Human description of the item in flight (e.g. `page 3/12`).
    current: String,
    /// Run start, for elapsed/ETA.
    started: Instant,
    /// Character width of the last-drawn line, so a shorter redraw blanks the
    /// leftover tail instead of leaving stale characters on screen.
    last_len: usize,
    /// Set by [`Progress::finish`] (and `Drop`): every later call is a no-op,
    /// so a stray sink tick can never redraw over subsequent output.
    finished: bool,
}

/// A single-line stderr progress bar; disabled instances are free no-ops.
///
/// Dropping an unfinished bar clears its line, so an early `?` return inside
/// a driver never leaves a half-drawn `\r` line for the error message to
/// collide with.
pub struct Progress {
    inner: Option<Arc<Mutex<Inner>>>,
}

impl Progress {
    /// A bar over `total` items, enabled only when ALL of these hold:
    /// `human_mode` (the caller is NOT in `--robot`/`--json` mode), `total`
    /// is non-zero, and stderr is an interactive terminal that has not opted
    /// out (see [`progress_allowed`]).
    pub fn new(label: &'static str, total: usize, human_mode: bool) -> Self {
        if !human_mode || total == 0 || !stderr_supports_progress() {
            return Self { inner: None };
        }
        Self {
            inner: Some(Arc::new(Mutex::new(Inner {
                label,
                total,
                done: 0,
                current: String::new(),
                started: Instant::now(),
                last_len: 0,
                finished: false,
            }))),
        }
    }

    /// True when the bar will actually draw. Drivers with a cheaper
    /// non-streaming path may branch on this to keep the disabled path
    /// byte-for-byte what it always was.
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Mark `current` as the item now in flight and redraw. The fill/percent
    /// reflect COMPLETED items, so a 12-page run draws `0%` while page 1
    /// decodes and reaches `100%` only as the line is retired.
    pub fn start_item(&self, current: impl Into<String>) {
        let Some(inner) = &self.inner else { return };
        let mut g = lock(inner);
        if g.finished {
            return;
        }
        g.current = current.into();
        draw(&mut g);
    }

    /// Record one completed item. No redraw: the next `start_item` (or sink
    /// tick) renders the advanced state, and the terminal `finish` clears the
    /// line anyway — drawing here would just flicker.
    pub fn complete_item(&self) {
        let Some(inner) = &self.inner else { return };
        let mut g = lock(inner);
        if g.finished {
            return;
        }
        g.done = (g.done + 1).min(g.total);
    }

    /// Temporarily clear the line (state kept) so the caller can print other
    /// output — e.g. `ocr-batch` writing a result block — without the bar
    /// visually fused to it. The next `start_item` restores the bar.
    pub fn suspend(&self) {
        let Some(inner) = &self.inner else { return };
        let mut g = lock(inner);
        if g.finished {
            return;
        }
        clear_line(&mut g);
    }

    /// Print a one-off stderr message (a skipped-page warning) WITHOUT
    /// corrupting the bar: clear the line, print, redraw. Disabled bars
    /// degrade to a plain `eprintln!`, keeping non-TTY stderr byte-identical
    /// to the pre-bar CLI.
    pub fn note(&self, msg: &str) {
        let Some(inner) = &self.inner else {
            eprintln!("{msg}");
            return;
        };
        let mut g = lock(inner);
        if g.finished {
            eprintln!("{msg}");
            return;
        }
        clear_line(&mut g);
        eprintln!("{msg}");
        draw(&mut g);
    }

    /// Retire the bar: clear the line and make every later call (including
    /// stray sink ticks) a no-op. The drivers' existing human summary lines
    /// (`[focr] … (12.3s)`) then print on a clean stderr.
    pub fn finish(&self) {
        let Some(inner) = &self.inner else { return };
        let mut g = lock(inner);
        if g.finished {
            return;
        }
        clear_line(&mut g);
        g.finished = true;
    }

    /// A `'static` boxed per-page sink ([`crate::PageSink`]) that advances
    /// this bar as the multi-page decode crosses each `<PAGE>` boundary — the
    /// human-mode twin of the robot NDJSON `page` events. The clone of the
    /// shared state keeps the sink `'static` for the blocking pool; once the
    /// owner finishes/drops the bar, ticks are no-ops.
    pub fn page_sink(&self) -> crate::PageSink {
        let Some(inner) = &self.inner else {
            return Box::new(|_page, _body| {});
        };
        let inner = Arc::clone(inner);
        Box::new(move |page, _body| {
            let mut g = lock(&inner);
            if g.finished {
                return;
            }
            g.done = page.min(g.total);
            g.current = format!("page {page}/{}", g.total);
            draw(&mut g);
        })
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Lock helper: a panicked holder can't poison anything worse than a garbled
/// progress line, so recover instead of unwrapping.
fn lock(inner: &Arc<Mutex<Inner>>) -> std::sync::MutexGuard<'_, Inner> {
    inner.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Redraw the line in place (`\r` + content + blank the stale tail).
/// Write errors are ignored — a broken stderr must never fail an OCR run.
fn draw(g: &mut Inner) {
    let line = render_line(
        g.label,
        g.done,
        g.total,
        &g.current,
        g.started.elapsed().as_secs_f64(),
    );
    let width = line.chars().count();
    let pad = g.last_len.saturating_sub(width);
    let mut err = std::io::stderr().lock();
    let _ = write!(err, "\r{line}{:pad$}", "");
    let _ = err.flush();
    g.last_len = width;
}

/// Blank the currently-drawn line and park the cursor at column 0.
fn clear_line(g: &mut Inner) {
    if g.last_len == 0 {
        return;
    }
    let mut err = std::io::stderr().lock();
    let _ = write!(err, "\r{:width$}\r", "", width = g.last_len);
    let _ = err.flush();
    g.last_len = 0;
}

/// Render one bar line (pure, for tests). ASCII only; never contains `\r` or
/// `\n` — the caller owns cursor movement.
fn render_line(label: &str, done: usize, total: usize, current: &str, elapsed_secs: f64) -> String {
    let total = total.max(1);
    let done = done.min(total);
    let filled = done * BAR_CELLS / total;
    let pct = done * 100 / total;
    let mut line = format!(
        "[focr] {label} [{}{}] {pct:>3}%",
        "#".repeat(filled),
        "-".repeat(BAR_CELLS - filled)
    );
    if !current.is_empty() {
        line.push(' ');
        line.push_str(current);
    }
    line.push_str(&format!(", {} elapsed", fmt_duration(elapsed_secs)));
    // Linear ETA once at least one item has completed (before that there is
    // no rate to extrapolate; after the last item the bar is about to clear).
    if done > 0 && done < total {
        let eta = elapsed_secs / done as f64 * (total - done) as f64;
        line.push_str(&format!(", ~{} left", fmt_duration(eta)));
    }
    line
}

/// `62.4s -> "1m02s"`, `3.2s -> "3s"`, `3725s -> "1h02m"`. Coarse on purpose:
/// an OCR ETA is an estimate, not a stopwatch.
fn fmt_duration(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    }
}

/// The full runtime gate for the current process's stderr.
fn stderr_supports_progress() -> bool {
    progress_allowed(
        std::io::stderr().is_terminal(),
        std::env::var("TERM").ok().as_deref(),
        std::env::var(NO_PROGRESS_ENV).ok().as_deref(),
    )
}

/// Pure decision function (unit-testable without touching the real process
/// environment): a bar is allowed only on an interactive, non-`dumb` terminal
/// that has not set the [`NO_PROGRESS_ENV`] kill switch.
pub fn progress_allowed(
    stderr_is_tty: bool,
    term: Option<&str>,
    no_progress: Option<&str>,
) -> bool {
    if !stderr_is_tty {
        return false;
    }
    if term == Some("dumb") {
        return false;
    }
    match no_progress {
        None => true,
        Some(v) => {
            let v = v.trim();
            v.is_empty() || v == "0"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_a_tty_never_draws() {
        // The non-negotiable robot/pipe guarantee: no TTY, no bar — whatever
        // TERM or the kill switch say.
        assert!(!progress_allowed(false, None, None));
        assert!(!progress_allowed(false, Some("xterm-256color"), None));
        assert!(!progress_allowed(false, Some("xterm"), Some("0")));
    }

    #[test]
    fn tty_gate_respects_term_and_kill_switch() {
        assert!(progress_allowed(true, None, None));
        assert!(progress_allowed(true, Some("xterm-256color"), None));
        assert!(!progress_allowed(true, Some("dumb"), None));
        assert!(!progress_allowed(true, Some("xterm"), Some("1")));
        assert!(!progress_allowed(true, Some("xterm"), Some("true")));
        // Explicit opt-back-in values keep the bar.
        assert!(progress_allowed(true, Some("xterm"), Some("0")));
        assert!(progress_allowed(true, Some("xterm"), Some("")));
    }

    #[test]
    fn robot_and_json_modes_construct_disabled() {
        // `human_mode = false` is how the CLI spells --robot/--json; the bar
        // must be inert regardless of the terminal.
        let bar = Progress::new("ocr", 10, false);
        assert!(!bar.is_enabled());
        // And every method on a disabled bar is a safe no-op.
        bar.start_item("page 1/10");
        bar.complete_item();
        bar.suspend();
        bar.finish();
        let mut sink = bar.page_sink();
        sink(3, "body");
    }

    #[test]
    fn zero_total_constructs_disabled() {
        assert!(!Progress::new("ocr", 0, true).is_enabled());
    }

    #[test]
    fn render_line_shape() {
        let line = render_line("ocr", 4, 12, "page 5/12", 42.0);
        assert!(line.starts_with("[focr] ocr ["));
        assert!(line.contains(" 33%"), "{line}");
        assert!(line.contains("page 5/12"), "{line}");
        assert!(line.contains("42s elapsed"), "{line}");
        // 4 done in 42s -> 10.5 s/page -> 8 remaining ~ 84s = 1m24s.
        assert!(line.contains("~1m24s left"), "{line}");
        // Single line, ASCII only — the caller owns all cursor movement.
        assert!(!line.contains('\r') && !line.contains('\n'));
        assert!(line.is_ascii());
    }

    #[test]
    fn render_line_eta_bounds() {
        // No ETA before the first completion (no rate to extrapolate)...
        assert!(!render_line("ocr", 0, 5, "page 1/5", 3.0).contains("left"));
        // ...and none once everything is done.
        let done = render_line("ocr", 5, 5, "", 30.0);
        assert!(!done.contains("left"));
        assert!(done.contains("100%"));
    }

    #[test]
    fn duration_formatting() {
        assert_eq!(fmt_duration(3.2), "3s");
        assert_eq!(fmt_duration(62.4), "1m02s");
        assert_eq!(fmt_duration(3725.0), "1h02m");
        assert_eq!(fmt_duration(-1.0), "0s");
    }
}
