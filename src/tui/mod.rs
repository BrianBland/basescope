pub mod events;
pub mod log_layer;
pub mod render;

use std::cell::Cell;
use std::time::Duration;

use anyhow::{anyhow, Result};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use ratatui::layout::Rect;
use tokio::sync::mpsc;
use crate::analysis::Analyzer;
use crate::cache::Cache;
use crate::domain::{ScanSpec, TxFilter, parse_filter};
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
pub enum HistogramMode {
    FilterMatches,
    AllBlocks,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkState {
    Pending,
    Cached,
    Fetching,
    Done,
    Failed,
}

pub struct App {
    pub mode: AppMode,
    pub range_field: RangeField,
    pub start_block_input: String,
    pub end_block_input: String,
    pub filters: Vec<TxFilter>,
    pub current_filter_input: String,
    pub analysis: Option<Analyzer>,
    pub snapshot: Option<crate::domain::AnalysisSnapshot>,
    pub pipeline_rx: Option<mpsc::UnboundedReceiver<PipelineEvent>>,
    pub status_message: String,
    pub selected_filter: usize,
    pub rpc_client: RpcClient,
    pub cache: Cache,
    pub concurrency: usize,
    pub should_quit: bool,
    pub log_buffer: LogBuffer,
    pub chunk_states: Vec<ChunkState>,
    pub chunk_progress: Vec<f32>,
    pub chunk_ranges: Vec<(u64, u64)>,
    pub bottom_panel: BottomPanel,
    pub granularity: Granularity,
    pub auto_granularity: usize,
    pub granularity_input: Option<String>,
    pub mouse_col: u16,
    pub mouse_row: u16,
    pub hist_mode: HistogramMode,
    pub show_help: bool,
    pub tx_chart_rect: Cell<Rect>,
    pub bf_chart_rect: Cell<Rect>,
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
            range_field: RangeField::Start,
            start_block_input: String::new(),
            end_block_input: String::new(),
            filters: Vec::new(),
            current_filter_input: String::new(),
            analysis: None,
            snapshot: None,
            pipeline_rx: None,
            status_message: String::new(),
            selected_filter: 0,
            rpc_client,
            cache,
            concurrency,
            should_quit: false,
            log_buffer,
            chunk_states: Vec::new(),
            chunk_progress: Vec::new(),
            chunk_ranges: Vec::new(),
            bottom_panel: BottomPanel::Logs,
            granularity: Granularity::Auto,
            auto_granularity: 1,
            granularity_input: None,
            mouse_col: 0,
            mouse_row: 0,
            hist_mode: HistogramMode::FilterMatches,
            show_help: false,
            tx_chart_rect: Cell::new(Rect::default()),
            bf_chart_rect: Cell::new(Rect::default()),
        };

        if let Some(spec) = cli_spec {
            app.start_block_input = spec.start_block.to_string();
            app.end_block_input = spec.end_block.to_string();
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
        if self.granularity_input.is_some() {
            return self.handle_granularity_input(key);
        }

        if key.code == KeyCode::Char('q') {
            self.should_quit = true;
            return Ok(());
        }

        if key.code == KeyCode::Char('?') {
            self.show_help = !self.show_help;
            return Ok(());
        }

        if matches!(self.mode, AppMode::Fetching | AppMode::Results) {
            match key.code {
                KeyCode::Char('l') => {
                    self.bottom_panel = match self.bottom_panel {
                        BottomPanel::Logs => BottomPanel::Hidden,
                        _ => BottomPanel::Logs,
                    };
                    return Ok(());
                }
                KeyCode::Char('r') => {
                    self.bottom_panel = match self.bottom_panel {
                        BottomPanel::Rpc => BottomPanel::Hidden,
                        _ => BottomPanel::Rpc,
                    };
                    return Ok(());
                }
                KeyCode::Char('g') => {
                    self.granularity = match self.granularity {
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
                    self.granularity_input = Some(String::new());
                    return Ok(());
                }
                KeyCode::Char('h') => {
                    self.hist_mode = match self.hist_mode {
                        HistogramMode::FilterMatches => HistogramMode::AllBlocks,
                        HistogramMode::AllBlocks => HistogramMode::FilterMatches,
                    };
                    return Ok(());
                }
                KeyCode::Char('a') => {
                    if let Some(analyzer) = &mut self.analysis {
                        analyzer.toggle_aggregate();
                        self.snapshot = Some(analyzer.snapshot());
                    }
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
        if let MouseEventKind::Moved = mouse.kind {
            self.mouse_col = mouse.column;
            self.mouse_row = mouse.row;
        }
    }

    fn handle_range_input(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Tab => {
                self.range_field = match self.range_field {
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
            KeyCode::Backspace => match self.range_field {
                RangeField::Start => {
                    self.start_block_input.pop();
                }
                RangeField::End => {
                    self.end_block_input.pop();
                }
            },
            KeyCode::Char(c) if c.is_ascii_digit() => match self.range_field {
                RangeField::Start => self.start_block_input.push(c),
                RangeField::End => self.end_block_input.push(c),
            },
            _ => {}
        }
        Ok(())
    }

    fn handle_filter_input(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Enter => {
                if self.current_filter_input.trim().is_empty() {
                    if self.filters.is_empty() {
                        self.status_message = "add at least one filter".to_string();
                        return Ok(());
                    }
                    let spec = self.build_spec()?;
                    self.start_pipeline(spec);
                    self.mode = AppMode::Fetching;
                    self.status_message.clear();
                } else {
                    let (kind, label) = parse_filter(self.current_filter_input.trim())
                        .map_err(|e| anyhow!(e))?;
                    let filter = TxFilter::new(self.filters.len(), kind, label);
                    self.filters.push(filter);
                    self.current_filter_input.clear();
                    self.status_message.clear();
                }
            }
            KeyCode::Backspace => {
                self.current_filter_input.pop();
            }
            KeyCode::Char('d') => {
                if self.selected_filter < self.filters.len() {
                    self.filters.remove(self.selected_filter);
                    self.reindex_filters();
                    if self.selected_filter >= self.filters.len() && !self.filters.is_empty() {
                        self.selected_filter = self.filters.len() - 1;
                    }
                }
            }
            KeyCode::Up => {
                if self.selected_filter > 0 {
                    self.selected_filter -= 1;
                }
            }
            KeyCode::Down => {
                if self.selected_filter + 1 < self.filters.len() {
                    self.selected_filter += 1;
                }
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.current_filter_input.push(c);
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
            .start_block_input
            .trim()
            .parse::<u64>()
            .map_err(|_| anyhow!("invalid start block"))?;
        let end = self
            .end_block_input
            .trim()
            .parse::<u64>()
            .map_err(|_| anyhow!("invalid end block"))?;
        Ok((start, end))
    }

    fn start_pipeline(&mut self, spec: ScanSpec) {
        let range = spec.end_block.saturating_sub(spec.start_block).max(1) as usize;
        self.auto_granularity = auto_granularity(range);

        self.analysis = Some(Analyzer::new(&spec.filters));
        self.snapshot = self.analysis.as_ref().map(|a| a.snapshot());

        let ranges = spec.chunk_ranges();
        self.chunk_states = vec![ChunkState::Pending; ranges.len()];
        self.chunk_progress = vec![0.0; ranges.len()];
        self.chunk_ranges = ranges;

        let (event_tx, event_rx) = mpsc::unbounded_channel();
        self.pipeline_rx = Some(event_rx);
        let pipeline = Pipeline::new(self.rpc_client.clone(), self.cache.clone(), self.concurrency);

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
                if let Some(ref input) = self.granularity_input {
                    let trimmed = input.trim();
                    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
                        self.granularity = Granularity::Auto;
                    } else if let Ok(v) = trimmed.parse::<usize>() {
                        self.granularity = Granularity::Fixed(v.max(1));
                    }
                }
                self.granularity_input = None;
            }
            KeyCode::Esc => {
                self.granularity_input = None;
            }
            KeyCode::Backspace => {
                if let Some(ref mut input) = self.granularity_input {
                    input.pop();
                }
            }
            KeyCode::Char(c) if c.is_ascii_alphanumeric() => {
                if let Some(ref mut input) = self.granularity_input {
                    input.push(c);
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn effective_granularity(&self) -> usize {
        match self.granularity {
            Granularity::Auto => self.auto_granularity,
            Granularity::Fixed(v) => v,
        }
    }

    pub fn granularity_label(&self) -> String {
        let g = self.effective_granularity();
        match self.granularity {
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
}

fn auto_granularity(block_range: usize) -> usize {
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
