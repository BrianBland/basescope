use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Frame;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::domain::AnalysisSnapshot;
use crate::tui::{App, ChunkState};

use super::colors::filter_color;

pub(super) fn is_complete(state: &ChunkState) -> bool {
    matches!(state, ChunkState::Done | ChunkState::Cached)
}

pub(super) fn render_sidebar(
    app: &App,
    snapshot: &AnalysisSnapshot,
    frame: &mut Frame,
    area: Rect,
) {
    let all_done = !app.chunk_states.is_empty() && app.chunk_states.iter().all(is_complete);
    let chunk_h = if all_done {
        1
    } else {
        let inner_w = area.width.saturating_sub(2).max(1) as usize;
        let total_chunks = app.chunk_states.len().max(1);
        let chunk_rows = total_chunks.div_ceil(inner_w) as u16;
        (chunk_rows + 2).max(3)
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(chunk_h),
            Constraint::Min(3),
            Constraint::Length(3),
        ])
        .split(area);

    if all_done {
        let total = app.chunk_states.len();
        let label = Paragraph::new(format!("chunks {total}/{total} ✓"))
            .style(Style::default().fg(Color::Green));
        frame.render_widget(label, chunks[0]);
    } else {
        render_chunk_map(app, frame, chunks[0]);
    }

    let legend_lines: Vec<Line> = snapshot
        .filters
        .iter()
        .map(|filter| {
            let status = if filter.enabled { "on" } else { "off" };
            let content = format!("{}: {}", filter.label, status);
            Line::from(Span::styled(
                content,
                Style::default().fg(filter_color(filter.color_index)),
            ))
        })
        .collect();
    let legend =
        Paragraph::new(legend_lines).block(Block::default().title("filters").borders(Borders::ALL));
    frame.render_widget(legend, chunks[1]);

    let hint = Paragraph::new("?: help").block(Block::default().borders(Borders::TOP));
    frame.render_widget(hint, chunks[2]);
}

// Dot layout:  1 4    Bit values: 1   8
//              2 5                2  16
//              3 6                4  32
//              7 8               64 128
const BRAILLE_FILL_LEFT: [char; 9] = ['⠀', '⠁', '⠉', '⠋', '⠛', '⠟', '⠿', '⡿', '⣿'];
const BRAILLE_FILL_RIGHT: [char; 9] = ['⠀', '⢀', '⣀', '⣠', '⣤', '⣴', '⣶', '⣾', '⣿'];

fn chunk_braille(state: ChunkState, progress: f32, forward: bool) -> (char, Color) {
    let fill = if forward {
        &BRAILLE_FILL_LEFT
    } else {
        &BRAILLE_FILL_RIGHT
    };
    match state {
        ChunkState::Pending => (fill[0], Color::DarkGray),
        ChunkState::Cached => (fill[8], Color::Blue),
        ChunkState::Fetching => {
            let level = (progress * 8.0).round().max(1.0) as usize;
            (fill[level.min(8)], Color::Yellow)
        }
        ChunkState::Done => (fill[8], Color::Green),
        ChunkState::Failed => (fill[8], Color::Red),
    }
}

fn chunk_forward(states: &[ChunkState], idx: usize) -> bool {
    let before = idx > 0 && is_complete(&states[idx - 1]);
    let after = idx + 1 < states.len() && is_complete(&states[idx + 1]);
    match (before, after) {
        (true, false) => true,
        (false, true) => false,
        _ => true,
    }
}

fn render_chunk_map(app: &App, frame: &mut Frame, area: Rect) {
    let total = app.chunk_states.len();
    let done = app.chunk_states.iter().filter(|s| is_complete(s)).count();
    let title = format!("chunks {done}/{total}");

    let inner_w = area.width.saturating_sub(2) as usize;
    let inner_h = area.height.saturating_sub(2) as usize;
    let capacity = inner_w * inner_h;

    let spans = if total <= capacity || capacity == 0 {
        chunk_spans_all(app)
    } else {
        chunk_spans_windowed(app, capacity)
    };

    let paragraph = Paragraph::new(Line::from(spans))
        .block(Block::default().title(title).borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn chunk_spans_all(app: &App) -> Vec<Span<'static>> {
    app.chunk_states
        .iter()
        .enumerate()
        .map(|(i, state)| {
            let progress = app.chunk_progress.get(i).copied().unwrap_or(0.0);
            let forward = chunk_forward(&app.chunk_states, i);
            let (ch, color) = chunk_braille(*state, progress, forward);
            Span::styled(ch.to_string(), Style::default().fg(color))
        })
        .collect()
}

fn active_frontier(states: &[ChunkState]) -> usize {
    if let Some(pos) = states
        .iter()
        .position(|s| matches!(s, ChunkState::Fetching))
    {
        return pos;
    }
    if let Some(last_done) = states.iter().rposition(is_complete) {
        return last_done;
    }
    0
}

fn chunk_spans_windowed(app: &App, capacity: usize) -> Vec<Span<'static>> {
    let total = app.chunk_states.len();
    let ellipsis = Span::styled("…", Style::default().fg(Color::DarkGray));
    let context = 3;

    let first_incomplete = app
        .chunk_states
        .iter()
        .position(|s| !is_complete(s))
        .unwrap_or(total);
    let last_incomplete = app
        .chunk_states
        .iter()
        .rposition(|s| !is_complete(s))
        .unwrap_or(0);

    let window_start = first_incomplete.saturating_sub(context);
    let window_end = (last_incomplete + context + 1).min(total);
    let window_len = window_end - window_start;

    if window_len + 2 <= capacity {
        let mut spans = Vec::new();
        if window_start > 0 {
            spans.push(ellipsis.clone());
        }
        for i in window_start..window_end {
            let progress = app.chunk_progress.get(i).copied().unwrap_or(0.0);
            let forward = chunk_forward(&app.chunk_states, i);
            let (ch, color) = chunk_braille(app.chunk_states[i], progress, forward);
            spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
        }
        if window_end < total {
            spans.push(ellipsis);
        }
        spans
    } else {
        let usable = capacity.saturating_sub(2);
        let anchor = active_frontier(&app.chunk_states);
        let half = usable / 2;
        let start = if anchor <= half {
            0
        } else if anchor + (usable - half) >= total {
            total.saturating_sub(usable)
        } else {
            anchor - half
        };
        let end = (start + usable).min(total);

        let mut spans = Vec::new();
        if start > 0 {
            spans.push(ellipsis.clone());
        }
        for i in start..end {
            let progress = app.chunk_progress.get(i).copied().unwrap_or(0.0);
            let forward = chunk_forward(&app.chunk_states, i);
            let (ch, color) = chunk_braille(app.chunk_states[i], progress, forward);
            spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
        }
        if end < total {
            spans.push(ellipsis);
        }
        spans
    }
}
