pub mod events;
pub mod log_layer;
pub mod render;

use std::cell::Cell;
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use ratatui::layout::Rect;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use crate::analysis::Analyzer;
use crate::cache::Cache;
use crate::domain::{ChartMode, ScanSpec, TxFilter, parse_filter};
use crate::pipeline::{Pipeline, PipelineEvent};
use crate::rpc::RpcClient;
use crate::tui::events::{poll_event, AppEvent};
use crate::tui::log_layer::LogBuffer;
use crate::tui::render::render;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    RangeInput,
    FilterInput,
    Fetching,
    Results,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottomPanel {
    Hidden,
    Logs,
    Rpc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeField {
    Start,
    End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    Auto,
    Fixed(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMode {
    Linear,
    Log10,
    Sqrt,
}

impl ScaleMode {
    fn raw_apply(self, v: f64) -> f64 {
        match self {
            ScaleMode::Linear => v,
            ScaleMode::Log10 => {
                if v <= 0.0 {
                    0.0
                } else {
                    v.log10()
                }
            }
            ScaleMode::Sqrt => {
                if v <= 0.0 {
                    0.0
                } else {
                    v.sqrt()
                }
            }
        }
    }

    fn raw_invert(self, v: f64) -> f64 {
        match self {
            ScaleMode::Linear => v,
            ScaleMode::Log10 => 10.0_f64.powf(v),
            ScaleMode::Sqrt => v * v,
        }
    }

    pub fn build_transform(self, data: &[(f64, f64)]) -> ScaleTransform {
        let offset = match self {
            ScaleMode::Linear | ScaleMode::Sqrt => 0.0,
            ScaleMode::Log10 => {
                let min_y = data
                    .iter()
                    .map(|(_, y)| *y)
                    .filter(|y| *y > 0.0)
                    .fold(f64::INFINITY, f64::min);
                if min_y.is_finite() {
                    self.raw_apply(min_y)
                } else {
                    0.0
                }
            }
        };
        ScaleTransform {
            mode: self,
            offset,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ScaleMode::Linear => "",
            ScaleMode::Log10 => " [log]",
            ScaleMode::Sqrt => " [sqrt]",
        }
    }

    pub fn next(self) -> Self {
        match self {
            ScaleMode::Linear => ScaleMode::Log10,
            ScaleMode::Log10 => ScaleMode::Sqrt,
            ScaleMode::Sqrt => ScaleMode::Linear,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScaleTransform {
    pub mode: ScaleMode,
    offset: f64,
}

impl ScaleTransform {
    pub fn apply(&self, v: f64) -> f64 {
        self.mode.raw_apply(v) - self.offset
    }

    pub fn invert(&self, v: f64) -> f64 {
        self.mode.raw_invert(v + self.offset)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistogramMode {
    FilterMatches,
    AllBlocks,
    Stacked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkState {
    Pending,
    Cached,
    Fetching,
    Done,
    Failed,
}

pub struct InputState {
    pub range_field: RangeField,
    pub start_block_input: String,
    pub end_block_input: String,
    pub current_filter_input: String,
    pub selected_filter: usize,
    pub granularity_input: Option<String>,
}

pub struct ViewState {
    pub mouse_col: u16,
    pub mouse_row: u16,
    pub tx_chart_rect: Cell<Rect>,
    pub bf_chart_rect: Cell<Rect>,
    pub view_start: Option<f64>,
    pub view_end: Option<f64>,
    pub full_x_range: Cell<(f64, f64)>,
    pub last_y_label_w: Cell<u16>,
    pub auto_granularity: Cell<usize>,
    pub bottom_panel: BottomPanel,
    pub granularity: Granularity,
    pub hist_mode: HistogramMode,
    pub scale_mode: ScaleMode,
    pub chart_mode: ChartMode,
    pub show_help: bool,
    pub hovered_block: Cell<Option<u64>>,
    pub first_x_label_w: Cell<u16>,
}

pub struct App {
    pub mode: AppMode,
    pub input: InputState,
    pub view: ViewState,
    pub filters: Vec<TxFilter>,
    pub analysis: Option<Analyzer>,
    pub snapshot: Option<Arc<crate::domain::AnalysisSnapshot>>,
    pub pipeline_rx: Option<mpsc::UnboundedReceiver<PipelineEvent>>,
    cancel_token: CancellationToken,
    pub status_message: String,
    pub rpc_client: RpcClient,
    pub cache: Cache,
    pub concurrency: usize,
    pub should_quit: bool,
    pub toast: Option<(String, Instant)>,
    pub log_buffer: LogBuffer,
    pub chunk_states: Vec<ChunkState>,
    pub chunk_progress: Vec<f32>,
    pub chunk_ranges: Vec<(u64, u64)>,
}

impl App {
    pub fn new(
        rpc_client: RpcClient,
        cache: Cache,
        concurrency: usize,
        cli_spec: Option<ScanSpec>,
        log_buffer: LogBuffer,
    ) -> Self {
        let mut app = Self {
            mode: AppMode::RangeInput,
            input: InputState {
                range_field: RangeField::Start,
                start_block_input: String::new(),
                end_block_input: String::new(),
                current_filter_input: String::new(),
                selected_filter: 0,
                granularity_input: None,
            },
            view: ViewState {
                mouse_col: 0,
                mouse_row: 0,
                tx_chart_rect: Cell::new(Rect::default()),
                bf_chart_rect: Cell::new(Rect::default()),
                view_start: None,
                view_end: None,
                full_x_range: Cell::new((0.0, 0.0)),
                last_y_label_w: Cell::new(8),
                auto_granularity: Cell::new(1),
                bottom_panel: BottomPanel::Logs,
                granularity: Granularity::Auto,
                hist_mode: HistogramMode::FilterMatches,
                scale_mode: ScaleMode::Linear,
                chart_mode: ChartMode::TxCount,
                show_help: false,
                hovered_block: Cell::new(None),
                first_x_label_w: Cell::new(0),
            },
            filters: Vec::new(),
            analysis: None,
            snapshot: None,
            pipeline_rx: None,
            cancel_token: CancellationToken::new(),
            status_message: String::new(),
            rpc_client,
            cache,
            concurrency,
            should_quit: false,
            toast: None,
            log_buffer,
            chunk_states: Vec::new(),
            chunk_progress: Vec::new(),
            chunk_ranges: Vec::new(),
        };

        if let Some(spec) = cli_spec {
            app.input.start_block_input = spec.start_block.to_string();
            app.input.end_block_input = spec.end_block.to_string();
            app.filters = spec.filters.clone();
            app.start_pipeline(spec);
            app.mode = AppMode::Fetching;
        }

        app
    }

    pub fn run(&mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<()> {
        while !self.should_quit {
            let event = poll_event(Duration::from_millis(50))?;
            match event {
                AppEvent::Key(key) => self.handle_key(key)?,
                AppEvent::Mouse(mouse) => self.handle_mouse(mouse),
                AppEvent::Tick => self.handle_tick()?,
            }
            terminal.draw(|frame| render(self, frame))?;
        }
        Ok(())
    }

    fn handle_tick(&mut self) -> Result<()> {
        if let Some((_, created)) = &self.toast
            && created.elapsed() >= Duration::from_secs(2)
        {
            self.toast = None;
        }
        if self.mode != AppMode::Fetching {
            return Ok(());
        }
        let Some(rx) = &mut self.pipeline_rx else {
            return Ok(());
        };

        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }

        let mut done = false;
        for event in events {
            match event {
                PipelineEvent::ChunkCached(chunk) => {
                    self.set_chunk_state(chunk.start_block, ChunkState::Cached);
                    if let Some(analyzer) = &mut self.analysis {
                        analyzer.process_chunk(&chunk);
                        self.snapshot = Some(analyzer.snapshot());
                    }
                }
                PipelineEvent::ChunkStarted { start, .. } => {
                    self.set_chunk_state(start, ChunkState::Fetching);
                }
                PipelineEvent::ChunkProgress { start, fetched, total } => {
                    if let Some(idx) = self.chunk_ranges.iter().position(|(s, _)| *s == start) {
                        self.chunk_progress[idx] = fetched as f32 / total as f32;
                    }
                }
                PipelineEvent::ChunkComplete(chunk) => {
                    self.set_chunk_state(chunk.start_block, ChunkState::Done);
                    if let Some(analyzer) = &mut self.analysis {
                        analyzer.process_chunk(&chunk);
                        self.snapshot = Some(analyzer.snapshot());
                    }
                }
                PipelineEvent::ChunkFailed { start, end, error } => {
                    self.set_chunk_state(start, ChunkState::Failed);
                    self.status_message = format!(
                        "chunk {start}-{end} failed: {error}",
                    );
                }
                PipelineEvent::Done => {
                    done = true;
                }
            }
        }
        if done {
            self.mode = AppMode::Results;
            self.pipeline_rx = None;
        }
        Ok(())
    }

    fn handle_key(&mut self, key: KeyEvent) -> Result<()> {
        if self.input.granularity_input.is_some() {
            return self.handle_granularity_input(key);
        }

        let in_text_input = matches!(self.mode, AppMode::RangeInput | AppMode::FilterInput);

        if key.code == KeyCode::Char('q') && !in_text_input
            || key.code == KeyCode::Esc && in_text_input
        {
            self.cancel_token.cancel();
            self.should_quit = true;
            return Ok(());
        }

        if key.code == KeyCode::Char('?') && !in_text_input {
            self.view.show_help = !self.view.show_help;
            return Ok(());
        }

        if matches!(self.mode, AppMode::Fetching | AppMode::Results) {
            match key.code {
                KeyCode::Char('l') => {
                    self.view.bottom_panel = match self.view.bottom_panel {
                        BottomPanel::Logs => BottomPanel::Hidden,
                        _ => BottomPanel::Logs,
                    };
                    return Ok(());
                }
                KeyCode::Char('r') => {
                    self.view.bottom_panel = match self.view.bottom_panel {
                        BottomPanel::Rpc => BottomPanel::Hidden,
                        _ => BottomPanel::Rpc,
                    };
                    return Ok(());
                }
                KeyCode::Char('g') => {
                    self.view.granularity = match self.view.granularity {
                        Granularity::Fixed(1) => Granularity::Fixed(10),
                        Granularity::Fixed(10) => Granularity::Fixed(100),
                        Granularity::Fixed(100) => Granularity::Fixed(1000),
                        Granularity::Fixed(1000) => Granularity::Auto,
                        Granularity::Auto => Granularity::Fixed(1),
                        Granularity::Fixed(_) => Granularity::Fixed(1),
                    };
                    return Ok(());
                }
                KeyCode::Char('G') => {
                    self.input.granularity_input = Some(String::new());
                    return Ok(());
                }
                KeyCode::Char('h') => {
                    self.view.hist_mode = match self.view.hist_mode {
                        HistogramMode::FilterMatches => HistogramMode::AllBlocks,
                        HistogramMode::AllBlocks => HistogramMode::Stacked,
                        HistogramMode::Stacked => HistogramMode::FilterMatches,
                    };
                    return Ok(());
                }
                KeyCode::Char('s') => {
                    self.view.scale_mode = self.view.scale_mode.next();
                    return Ok(());
                }
                KeyCode::Char('t') => {
                    self.view.chart_mode = self.view.chart_mode.next();
                    return Ok(());
                }
                KeyCode::Char('c') => {
                    self.copy_hovered_block();
                    return Ok(());
                }
                KeyCode::Char('a') => {
                    if let Some(analyzer) = &mut self.analysis {
                        analyzer.toggle_aggregate();
                        self.snapshot = Some(analyzer.snapshot());
                    }
                    return Ok(());
                }
                KeyCode::Char('z') => {
                    self.zoom(true);
                    return Ok(());
                }
                KeyCode::Char('Z') => {
                    self.zoom(false);
                    return Ok(());
                }
                KeyCode::Left => {
                    self.pan(-PAN_FRACTION);
                    return Ok(());
                }
                KeyCode::Right => {
                    self.pan(PAN_FRACTION);
                    return Ok(());
                }
                KeyCode::Home => {
                    self.view.view_start = None;
                    self.view.view_end = None;
                    return Ok(());
                }
                KeyCode::Char(c) if c.is_ascii_digit() => {
                    let index = c.to_digit(10).unwrap_or(0) as usize;
                    if index > 0 {
                        let filter_index = index - 1;
                        if let Some(filter) = self.filters.get(filter_index)
                            && let Some(analyzer) = &mut self.analysis
                        {
                            analyzer.toggle_filter(filter.id);
                            self.snapshot = Some(analyzer.snapshot());
                        }
                    }
                    return Ok(());
                }
                _ => {}
            }
        }

        match self.mode {
            AppMode::RangeInput => self.handle_range_input(key),
            AppMode::FilterInput => self.handle_filter_input(key),
            AppMode::Fetching | AppMode::Results => Ok(()),
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::Moved => {
                self.view.mouse_col = mouse.column;
                self.view.mouse_row = mouse.row;
            }
            MouseEventKind::Down(crossterm::event::MouseButton::Left) => {
                self.view.mouse_col = mouse.column;
                self.view.mouse_row = mouse.row;
                if matches!(self.mode, AppMode::Fetching | AppMode::Results) {
                    self.copy_hovered_block();
                }
            }
            MouseEventKind::ScrollUp => {
                self.view.mouse_col = mouse.column;
                self.view.mouse_row = mouse.row;
                if matches!(self.mode, AppMode::Fetching | AppMode::Results) {
                    self.zoom(true);
                }
            }
            MouseEventKind::ScrollDown => {
                self.view.mouse_col = mouse.column;
                self.view.mouse_row = mouse.row;
                if matches!(self.mode, AppMode::Fetching | AppMode::Results) {
                    self.zoom(false);
                }
            }
            MouseEventKind::ScrollLeft => {
                if matches!(self.mode, AppMode::Fetching | AppMode::Results) {
                    self.pan(-PAN_FRACTION);
                }
            }
            MouseEventKind::ScrollRight => {
                if matches!(self.mode, AppMode::Fetching | AppMode::Results) {
                    self.pan(PAN_FRACTION);
                }
            }
            _ => {}
        }
    }

    fn copy_hovered_block(&mut self) {
        if let Some(block) = self.view.hovered_block.get() {
            let text = block.to_string();
            let msg = if copy_to_clipboard(&text) {
                format!("copied block {text}")
            } else {
                format!("block {text} (clipboard unavailable)")
            };
            self.toast = Some((msg, Instant::now()));
        }
    }

    fn handle_range_input(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Tab => {
                self.input.range_field = match self.input.range_field {
                    RangeField::Start => RangeField::End,
                    RangeField::End => RangeField::Start,
                };
            }
            KeyCode::Enter => {
                let (start, end) = self.parse_range_inputs()?;
                if start > end {
                    self.status_message = format!("start block {start} > end block {end}");
                    return Ok(());
                }
                self.status_message.clear();
                self.mode = AppMode::FilterInput;
            }
            KeyCode::Backspace => match self.input.range_field {
                RangeField::Start => {
                    self.input.start_block_input.pop();
                }
                RangeField::End => {
                    self.input.end_block_input.pop();
                }
            },
            KeyCode::Char(c) if c.is_ascii_digit() => match self.input.range_field {
                RangeField::Start => self.input.start_block_input.push(c),
                RangeField::End => self.input.end_block_input.push(c),
            },
            _ => {}
        }
        Ok(())
    }

    fn handle_filter_input(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Enter => {
                if self.input.current_filter_input.trim().is_empty() {
                    if self.filters.is_empty() {
                        self.status_message = "add at least one filter".to_string();
                        return Ok(());
                    }
                    let spec = self.build_spec()?;
                    self.start_pipeline(spec);
                    self.mode = AppMode::Fetching;
                    self.status_message.clear();
                } else {
                    let (kind, label) = parse_filter(self.input.current_filter_input.trim())
                        .map_err(|e| anyhow!(e))?;
                    let filter = TxFilter::new(self.filters.len(), kind, label);
                    self.filters.push(filter);
                    self.input.current_filter_input.clear();
                    self.status_message.clear();
                }
            }
            KeyCode::Backspace => {
                self.input.current_filter_input.pop();
            }
            KeyCode::Char('d') if self.input.current_filter_input.is_empty() => {
                if self.input.selected_filter < self.filters.len() {
                    self.filters.remove(self.input.selected_filter);
                    self.reindex_filters();
                    if self.input.selected_filter >= self.filters.len() && !self.filters.is_empty() {
                        self.input.selected_filter = self.filters.len() - 1;
                    }
                }
            }
            KeyCode::Up => {
                if self.input.selected_filter > 0 {
                    self.input.selected_filter -= 1;
                }
            }
            KeyCode::Down => {
                if self.input.selected_filter + 1 < self.filters.len() {
                    self.input.selected_filter += 1;
                }
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.input.current_filter_input.push(c);
            }
            _ => {}
        }
        Ok(())
    }

    fn build_spec(&self) -> Result<ScanSpec> {
        let (start, end) = self.parse_range_inputs()?;
        if start > end {
            return Err(anyhow!("start block {start} > end block {end}"));
        }
        if self.filters.is_empty() {
            return Err(anyhow!("no filters provided"));
        }
        Ok(ScanSpec {
            start_block: start,
            end_block: end,
            filters: self.filters.clone(),
        })
    }

    fn parse_range_inputs(&self) -> Result<(u64, u64)> {
        let start = self
            .input
            .start_block_input
            .trim()
            .parse::<u64>()
            .map_err(|_| anyhow!("invalid start block"))?;
        let end = self
            .input
            .end_block_input
            .trim()
            .parse::<u64>()
            .map_err(|_| anyhow!("invalid end block"))?;
        Ok((start, end))
    }

    fn start_pipeline(&mut self, spec: ScanSpec) {
        // Cancel any previous pipeline.
        self.cancel_token.cancel();
        self.cancel_token = CancellationToken::new();

        let range = spec.end_block.saturating_sub(spec.start_block).max(1) as usize;
        self.view.auto_granularity.set(auto_granularity(range));

        self.analysis = Some(Analyzer::new(&spec.filters));
        self.snapshot = self.analysis.as_mut().map(|a| a.snapshot());

        let ranges = spec.chunk_ranges();
        self.chunk_states = vec![ChunkState::Pending; ranges.len()];
        self.chunk_progress = vec![0.0; ranges.len()];
        self.chunk_ranges = ranges;

        let (event_tx, event_rx) = mpsc::unbounded_channel();
        self.pipeline_rx = Some(event_rx);
        let pipeline = Pipeline::new(
            self.rpc_client.clone(),
            self.cache.clone(),
            self.concurrency,
            self.cancel_token.clone(),
        );

        tokio::spawn(async move {
            if let Err(err) = pipeline.run(&spec, event_tx.clone()).await {
                let _ = event_tx.send(PipelineEvent::ChunkFailed {
                    start: spec.start_block,
                    end: spec.end_block,
                    error: err.to_string(),
                });
            }
        });
    }

    fn set_chunk_state(&mut self, start_block: u64, state: ChunkState) {
        if let Some(idx) = self.chunk_ranges.iter().position(|(s, _)| *s == start_block) {
            self.chunk_states[idx] = state;
        }
    }

    fn handle_granularity_input(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Enter => {
                if let Some(ref input) = self.input.granularity_input {
                    let trimmed = input.trim();
                    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
                        self.view.granularity = Granularity::Auto;
                    } else if let Ok(v) = trimmed.parse::<usize>() {
                        self.view.granularity = Granularity::Fixed(v.max(1));
                    }
                }
                self.input.granularity_input = None;
            }
            KeyCode::Esc => {
                self.input.granularity_input = None;
            }
            KeyCode::Backspace => {
                if let Some(ref mut input) = self.input.granularity_input {
                    input.pop();
                }
            }
            KeyCode::Char(c) if c.is_ascii_alphanumeric() => {
                if let Some(ref mut input) = self.input.granularity_input {
                    input.push(c);
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn effective_granularity(&self) -> usize {
        match self.view.granularity {
            Granularity::Auto => self.view.auto_granularity.get(),
            Granularity::Fixed(v) => v,
        }
    }

    pub fn granularity_label(&self) -> String {
        let g = self.effective_granularity();
        match self.view.granularity {
            Granularity::Auto if g > 1 => format!(" (auto {}blk)", g),
            Granularity::Auto => String::new(),
            Granularity::Fixed(v) => format!(" ({}blk)", v),
        }
    }

    fn reindex_filters(&mut self) {
        for (idx, filter) in self.filters.iter_mut().enumerate() {
            filter.id = crate::domain::FilterId(idx);
            filter.color_index = idx;
        }
    }

    fn mouse_to_data_x(&self, x_min: f64, x_max: f64) -> Option<f64> {
        let col = self.view.mouse_col;
        let row = self.view.mouse_row;
        let tx_rect = self.view.tx_chart_rect.get();
        let bf_rect = self.view.bf_chart_rect.get();

        let in_tx = col >= tx_rect.x
            && col < tx_rect.x + tx_rect.width
            && row >= tx_rect.y
            && row < tx_rect.y + tx_rect.height;
        let in_bf = col >= bf_rect.x
            && col < bf_rect.x + bf_rect.width
            && row >= bf_rect.y
            && row < bf_rect.y + bf_rect.height;

        if !in_tx && !in_bf {
            return None;
        }

        let chart_rect = if in_tx { tx_rect } else { bf_rect };
        let y_label_w = self.view.last_y_label_w.get();
        let first_x_label_w = self.view.first_x_label_w.get();
        let inner = render::chart_inner(chart_rect, y_label_w, first_x_label_w);

        if col < inner.x || col >= inner.x + inner.width || inner.width == 0 {
            return None;
        }

        let graph_w = inner.width.saturating_sub(1).max(1) as f64;
        let frac = ((col - inner.x) as f64) / graph_w;
        let frac = frac.clamp(0.0, 1.0);
        Some(x_min + frac * (x_max - x_min))
    }

    fn zoom(&mut self, zoom_in: bool) {
        let (full_min, full_max) = self.view.full_x_range.get();
        if full_max <= full_min {
            return;
        }

        let cur_min = self.view.view_start.unwrap_or(full_min);
        let cur_max = self.view.view_end.unwrap_or(full_max);
        let cur_range = cur_max - cur_min;

        let center = self
            .mouse_to_data_x(cur_min, cur_max)
            .unwrap_or((cur_min + cur_max) / 2.0);

        let factor = if zoom_in { ZOOM_IN_FACTOR } else { ZOOM_OUT_FACTOR };
        let new_range = (cur_range * factor).max(MIN_ZOOM_RANGE);

        if new_range >= (full_max - full_min) {
            self.view.view_start = None;
            self.view.view_end = None;
            return;
        }

        let left_frac = if cur_range > 0.0 {
            (center - cur_min) / cur_range
        } else {
            0.5
        };
        let new_min = center - left_frac * new_range;
        let new_max = new_min + new_range;

        let (clamped_min, clamped_max) = if new_min < full_min {
            (full_min, (full_min + new_range).min(full_max))
        } else if new_max > full_max {
            ((full_max - new_range).max(full_min), full_max)
        } else {
            (new_min, new_max)
        };

        self.view.view_start = Some(clamped_min);
        self.view.view_end = Some(clamped_max);
    }

    fn pan(&mut self, fraction: f64) {
        let (full_min, full_max) = self.view.full_x_range.get();
        if full_max <= full_min || (self.view.view_start.is_none() && self.view.view_end.is_none()) {
            return;
        }

        let cur_min = self.view.view_start.unwrap_or(full_min);
        let cur_max = self.view.view_end.unwrap_or(full_max);
        let cur_range = cur_max - cur_min;
        let delta = cur_range * fraction;
        let mut new_min = cur_min + delta;
        let mut new_max = cur_max + delta;

        if new_min < full_min {
            new_max += full_min - new_min;
            new_min = full_min;
        }
        if new_max > full_max {
            new_min -= new_max - full_max;
            new_max = full_max;
            new_min = new_min.max(full_min);
        }

        self.view.view_start = Some(new_min);
        self.view.view_end = Some(new_max);
    }
}

fn copy_to_clipboard(text: &str) -> bool {
    let mut child = if cfg!(target_os = "macos") {
        std::process::Command::new("pbcopy")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
    } else {
        std::process::Command::new("xclip")
            .args(["-selection", "clipboard"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
    };
    match child {
        Ok(ref mut c) => {
            if let Some(ref mut stdin) = c.stdin {
                let _ = stdin.write_all(text.as_bytes());
            }
            c.wait().is_ok_and(|s| s.success())
        }
        Err(_) => false,
    }
}

const ZOOM_IN_FACTOR: f64 = 0.8;
const ZOOM_OUT_FACTOR: f64 = 1.25;
const MIN_ZOOM_RANGE: f64 = 20.0;
const PAN_FRACTION: f64 = 0.1;

pub(crate) fn auto_granularity(block_range: usize) -> usize {
    const TARGET_POINTS: usize = 2500;
    let raw = block_range / TARGET_POINTS;
    match raw {
        0 => 1,
        1..=7 => 1,
        8..=30 => 10,
        31..=300 => 100,
        301..=3000 => 1000,
        _ => 10000,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_granularity_small() {
        assert_eq!(auto_granularity(100), 1);
    }

    #[test]
    fn auto_granularity_medium() {
        assert_eq!(auto_granularity(25_000), 10);
    }

    #[test]
    fn auto_granularity_large() {
        assert_eq!(auto_granularity(250_000), 100);
    }

    #[test]
    fn auto_granularity_very_large() {
        assert_eq!(auto_granularity(2_500_000), 1000);
    }

    #[test]
    fn auto_granularity_huge() {
        assert_eq!(auto_granularity(25_000_000), 10000);
    }

    #[test]
    fn scale_mode_linear_roundtrip() {
        let transform = ScaleMode::Linear.build_transform(&[(1.0, 5.0), (2.0, 10.0)]);
        let val = 7.5;
        let scaled = transform.apply(val);
        let inverted = transform.invert(scaled);
        assert!((inverted - val).abs() < 1e-10);
    }

    #[test]
    fn scale_mode_log10_roundtrip() {
        let data = vec![(1.0, 1.0), (2.0, 100.0)];
        let transform = ScaleMode::Log10.build_transform(&data);
        let val = 50.0;
        let scaled = transform.apply(val);
        let inverted = transform.invert(scaled);
        assert!((inverted - val).abs() < 1e-6);
    }

    #[test]
    fn scale_mode_sqrt_roundtrip() {
        let data = vec![(1.0, 4.0), (2.0, 16.0)];
        let transform = ScaleMode::Sqrt.build_transform(&data);
        let val = 9.0;
        let scaled = transform.apply(val);
        let inverted = transform.invert(scaled);
        assert!((inverted - val).abs() < 1e-10);
    }

    #[test]
    fn scale_mode_next_cycles() {
        assert_eq!(ScaleMode::Linear.next(), ScaleMode::Log10);
        assert_eq!(ScaleMode::Log10.next(), ScaleMode::Sqrt);
        assert_eq!(ScaleMode::Sqrt.next(), ScaleMode::Linear);
    }
}
