use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

pub struct ProgressReporter {
    label: String,
    total: usize,
    completed: AtomicUsize,
    last_reported_pct: AtomicUsize,
    report_lock: Mutex<()>,
    started_at: Instant,
}

impl ProgressReporter {
    pub fn new(label: impl Into<String>, total: usize) -> Self {
        Self {
            label: label.into(),
            total,
            completed: AtomicUsize::new(0),
            last_reported_pct: AtomicUsize::new(0),
            report_lock: Mutex::new(()),
            started_at: Instant::now(),
        }
    }

    pub fn tick(&self) {
        let done = self.completed.fetch_add(1, Ordering::Relaxed) + 1;
        self.maybe_report(done);
    }

    pub fn finish(&self) {
        self.maybe_report(self.total.max(self.completed.load(Ordering::Relaxed)));
    }

    fn maybe_report(&self, done: usize) {
        if self.total == 0 {
            return;
        }
        let _guard = self.report_lock.lock().expect("progress lock poisoned");
        let pct = ((done.min(self.total)) * 100) / self.total.max(1);
        let last = self.last_reported_pct.load(Ordering::Relaxed);
        if pct <= last {
            return;
        }
        self.last_reported_pct.store(pct, Ordering::Relaxed);
        self.report(done.min(self.total), pct as f64);
    }

    fn report(&self, done: usize, pct: f64) {
        eprintln!(
            "[progress] {}: {}/{} ({:.1}%) elapsed={}s",
            self.label,
            done,
            self.total,
            pct,
            self.started_at.elapsed().as_secs()
        );
    }
}
