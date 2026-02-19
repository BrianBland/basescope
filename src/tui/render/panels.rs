use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Frame;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use tracing::Level;

use crate::tui::{App, AppMode, Granularity};

use super::truncate_to;

pub(super) fn render_log_panel(app: &App, frame: &mut Frame, area: Rect) {
    let inner_height = area.height.saturating_sub(2) as usize;
    let lines = app.log_buffer.recent(inner_height);
    let log_lines: Vec<Line> = lines
        .into_iter()
        .map(|(level, msg)| {
            let color = match level {
                Level::ERROR => Color::Red,
                Level::WARN => Color::Yellow,
                Level::INFO => Color::Gray,
                Level::DEBUG => Color::DarkGray,
                Level::TRACE => Color::DarkGray,
            };
            let prefix = match level {
                Level::ERROR => "ERR",
                Level::WARN => "WRN",
                Level::INFO => "INF",
                Level::DEBUG => "DBG",
                Level::TRACE => "TRC",
            };
            Line::from(vec![
                Span::styled(format!("{prefix} "), Style::default().fg(color)),
                Span::styled(msg, Style::default().fg(color)),
            ])
        })
        .collect();

    let block = Block::default().title("logs").borders(Borders::ALL);
    let paragraph = Paragraph::new(log_lines).block(block);
    frame.render_widget(paragraph, area);
}

fn rpc_endpoint_line(ep: &crate::rpc::EndpointInfo, max_url_len: usize) -> Line<'static> {
    let bar_width = 10;
    let filled = (ep.score * bar_width as f64).round() as usize;
    let empty = bar_width - filled;
    let bar: String = "█".repeat(filled) + &"░".repeat(empty);

    let bar_color = if ep.score >= 0.7 {
        Color::Green
    } else if ep.score >= 0.3 {
        Color::Yellow
    } else {
        Color::Red
    };

    let url_display = truncate_to(&ep.url, max_url_len);

    Line::from(vec![
        Span::styled(format!("{bar} "), Style::default().fg(bar_color)),
        Span::styled(
            format!("{:>3.0}% ", ep.score * 100.0),
            Style::default().fg(bar_color),
        ),
        Span::styled(url_display, Style::default().fg(Color::Gray)),
    ])
}

pub(super) fn render_rpc_panel(app: &App, frame: &mut Frame, area: Rect) {
    let endpoints = app.rpc_client.endpoint_info();
    let inner_height = area.height.saturating_sub(2).max(1) as usize;

    if endpoints.len() <= inner_height {
        let inner_width = area.width.saturating_sub(2) as usize;
        let url_max = inner_width.saturating_sub(16);
        let lines: Vec<Line> = endpoints
            .iter()
            .map(|ep| rpc_endpoint_line(ep, url_max))
            .collect();
        let block = Block::default()
            .title("rpc endpoints")
            .borders(Borders::ALL);
        let paragraph = Paragraph::new(lines).block(block);
        frame.render_widget(paragraph, area);
    } else {
        let cols = endpoints.len().div_ceil(inner_height);
        let constraints: Vec<Constraint> = (0..cols)
            .map(|_| Constraint::Ratio(1, cols as u32))
            .collect();

        let block = Block::default()
            .title("rpc endpoints")
            .borders(Borders::ALL);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let col_areas = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(constraints)
            .split(inner);

        for (col_idx, col_area) in col_areas.iter().enumerate() {
            let start = col_idx * inner_height;
            let end = (start + inner_height).min(endpoints.len());
            let url_max = (col_area.width as usize).saturating_sub(16);
            let lines: Vec<Line> = endpoints[start..end]
                .iter()
                .map(|ep| rpc_endpoint_line(ep, url_max))
                .collect();
            let paragraph = Paragraph::new(lines);
            frame.render_widget(paragraph, *col_area);
        }
    }
}

fn help_lines(mode: AppMode) -> Vec<Line<'static>> {
    let key_style = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);
    let desc_style = Style::default().fg(Color::Gray);

    let mut entries: Vec<(&str, &str)> = vec![("?", "toggle this help")];

    match mode {
        AppMode::RangeInput => {
            entries.push(("Esc", "quit"));
            entries.push(("Tab", "switch field"));
            entries.push(("Enter", "continue"));
        }
        AppMode::FilterInput => {
            entries.push(("Esc", "quit"));
            entries.push(("Enter", "add filter / start scan"));
            entries.push(("d", "delete selected filter"));
            entries.push(("↑/↓", "select filter"));
        }
        AppMode::Fetching | AppMode::Results => {
            entries.push(("q", "quit"));
            entries.push(("1-9", "toggle filter"));
            entries.push(("a", "aggregate mode"));
            entries.push(("g", "cycle granularity"));
            entries.push(("G", "set granularity"));
            entries.push(("h", "switch histogram"));
            entries.push(("s", "cycle scale mode"));
            entries.push(("t", "cycle chart type"));
            entries.push(("l", "toggle logs"));
            entries.push(("r", "toggle rpc info"));
            entries.push(("mouse", "crosshair"));
            entries.push(("z/Z/scroll", "zoom in/out"));
            entries.push(("←/→/hscroll", "pan"));
            entries.push(("Home", "reset zoom"));
        }
    }

    entries
        .into_iter()
        .map(|(key, desc)| {
            Line::from(vec![
                Span::styled(format!("{key:>5}"), key_style),
                Span::styled(format!("  {desc}"), desc_style),
            ])
        })
        .collect()
}

pub(super) fn render_help_panel(app: &App, frame: &mut Frame, outer: Rect) {
    let lines = help_lines(app.mode);
    let panel_h = (lines.len() as u16 + 2).min(outer.height);
    let panel_w = 30u16.min(outer.width);
    let area = Rect {
        x: outer.x + outer.width.saturating_sub(panel_w),
        y: outer.y,
        width: panel_w,
        height: panel_h,
    };

    frame.render_widget(ratatui::widgets::Clear, area);
    let block = Block::default()
        .title("help")
        .borders(Borders::ALL)
        .style(Style::default().bg(Color::Black));
    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}

pub(super) fn render_granularity_input(app: &App, frame: &mut Frame, outer: Rect) {
    let panel_w = 32u16.min(outer.width);
    let panel_h = 3u16;
    let area = Rect {
        x: outer.x + (outer.width.saturating_sub(panel_w)) / 2,
        y: outer.y + (outer.height.saturating_sub(panel_h)) / 2,
        width: panel_w,
        height: panel_h,
    };

    frame.render_widget(ratatui::widgets::Clear, area);

    let input_text = app.input.granularity_input.as_deref().unwrap_or("");
    let hint = if input_text.is_empty() {
        match app.view.granularity {
            Granularity::Auto => "auto".to_string(),
            Granularity::Fixed(v) => v.to_string(),
        }
    } else {
        input_text.to_string()
    };

    let style = if input_text.is_empty() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
    };
    let paragraph = Paragraph::new(hint).style(style).block(
        Block::default()
            .title("granularity (number or 'auto')")
            .borders(Borders::ALL)
            .style(Style::default().bg(Color::Black)),
    );
    frame.render_widget(paragraph, area);

    let cursor_x = area.x + 1 + input_text.len() as u16;
    let cursor_y = area.y + 1;
    frame.set_cursor_position((cursor_x, cursor_y));
}
