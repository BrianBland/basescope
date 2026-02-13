use std::collections::HashMap;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Frame;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Bar, BarChart, BarGroup, Block, Borders, Chart, Dataset, GraphType, List, ListItem,
    Paragraph, Wrap,
};
use tracing::Level;

use crate::domain::approx_head_block;
use crate::tui::{App, AppMode, ChunkState, HistogramMode, RangeField};

const LOG_PANEL_HEIGHT: u16 = 8;

pub fn render(app: &App, frame: &mut Frame) {
    let outer = frame.area();

    if app.show_logs {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(LOG_PANEL_HEIGHT)])
            .split(outer);

        let main_area = layout[0];
        let log_area = layout[1];

        render_main(app, frame, main_area);
        render_log_panel(app, frame, log_area);
    } else {
        render_main(app, frame, outer);
    }
}

fn render_main(app: &App, frame: &mut Frame, area: Rect) {
    match app.mode {
        AppMode::RangeInput => render_range_input(app, frame, area),
        AppMode::FilterInput => render_filter_input(app, frame, area),
        AppMode::Fetching | AppMode::Results => render_results(app, frame, area),
    }
}

fn render_range_input(app: &App, frame: &mut Frame, area: Rect) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .margin(2)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(1),
        ])
        .split(area);

    let title = Paragraph::new(Line::from(Span::styled(
        "basescope",
        Style::default().add_modifier(Modifier::BOLD),
    )));
    frame.render_widget(title, layout[0]);

    let start_block = Paragraph::new(app.start_block_input.as_str())
        .block(Block::default().title("start block").borders(Borders::ALL));
    let end_block = Paragraph::new(app.end_block_input.as_str())
        .block(Block::default().title("end block").borders(Borders::ALL));
    frame.render_widget(start_block, layout[1]);
    frame.render_widget(end_block, layout[2]);

    let hint = Paragraph::new(format!("approx head block: {}", approx_head_block()));
    frame.render_widget(hint, layout[3]);

    let instructions = Paragraph::new("Tab: switch field  Enter: continue  q: quit")
        .block(Block::default().borders(Borders::TOP));
    frame.render_widget(instructions, layout[4]);

    let cursor_x = match app.range_field {
        RangeField::Start => layout[1].x + 1 + app.start_block_input.len() as u16,
        RangeField::End => layout[2].x + 1 + app.end_block_input.len() as u16,
    };
    let cursor_y = match app.range_field {
        RangeField::Start => layout[1].y + 1,
        RangeField::End => layout[2].y + 1,
    };
    frame.set_cursor_position((cursor_x, cursor_y));

    if !app.status_message.is_empty() {
        let status =
            Paragraph::new(app.status_message.as_str()).style(Style::default().fg(Color::LightRed));
        let status_area = Rect {
            x: layout[4].x,
            y: layout[4].y - 1,
            width: layout[4].width,
            height: 1,
        };
        frame.render_widget(status, status_area);
    }
}

fn render_filter_input(app: &App, frame: &mut Frame, area: Rect) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .margin(2)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(6),
            Constraint::Length(3),
            Constraint::Length(3),
        ])
        .split(area);

    let title = Paragraph::new(Line::from(Span::styled(
        "filters",
        Style::default().add_modifier(Modifier::BOLD),
    )));
    frame.render_widget(title, layout[0]);

    let items: Vec<ListItem> = app
        .filters
        .iter()
        .enumerate()
        .map(|(idx, filter)| {
            let line = Line::from(format!("{}. {}", idx + 1, filter.label));
            let style = if idx == app.selected_filter {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            ListItem::new(line).style(style)
        })
        .collect();
    let list = List::new(items).block(
        Block::default()
            .title("added filters")
            .borders(Borders::ALL),
    );
    frame.render_widget(list, layout[1]);

    let input = Paragraph::new(app.current_filter_input.as_str())
        .block(Block::default().title("filter input").borders(Borders::ALL));
    frame.render_widget(input, layout[2]);
    frame.set_cursor_position((
        layout[2].x + 1 + app.current_filter_input.len() as u16,
        layout[2].y + 1,
    ));

    let instructions =
        Paragraph::new("Enter: add filter (label=to:0x… | to:0x…)  Enter∅: scan  d: del  q: quit")
            .block(Block::default().borders(Borders::TOP));
    frame.render_widget(instructions, layout[3]);

    if !app.status_message.is_empty() {
        let status =
            Paragraph::new(app.status_message.as_str()).style(Style::default().fg(Color::LightRed));
        let status_area = Rect {
            x: layout[3].x,
            y: layout[3].y - 1,
            width: layout[3].width,
            height: 1,
        };
        frame.render_widget(status, status_area);
    }
}

fn render_results(app: &App, frame: &mut Frame, area: Rect) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(30),
            Constraint::Percentage(30),
        ])
        .split(area);

    let snapshot = match &app.snapshot {
        Some(snapshot) => snapshot,
        None => {
            let placeholder =
                Paragraph::new("waiting for data...").block(Block::default().borders(Borders::ALL));
            frame.render_widget(placeholder, area);
            return;
        }
    };

    app.tx_chart_rect.set(layout[0]);
    app.bf_chart_rect.set(layout[1]);

    let g = app.granularity;
    let (x_min, x_max) = series_x_bounds(&snapshot.base_fee_series);
    let x_labels = x_axis_labels(x_min, x_max, layout[0].width.saturating_sub(8) as usize);

    let grouped_filter_series: Vec<(String, usize, Vec<(f64, f64)>)> = snapshot
        .filters
        .iter()
        .filter(|f| f.enabled)
        .filter_map(|f| {
            snapshot
                .filter_series
                .get(&f.id)
                .map(|series| (f.label.clone(), f.color_index, group_series_sum(series, g)))
        })
        .collect();

    let grouped_base_fee = group_series_avg(&snapshot.base_fee_series, g);
    let (by_min, by_max) = series_y_bounds(&[&grouped_base_fee]);

    let tx_series_refs: Vec<&[(f64, f64)]> = grouped_filter_series
        .iter()
        .map(|(_, _, v)| v.as_slice())
        .collect();
    let (ty_min, ty_max) = series_y_bounds(&tx_series_refs);

    let y_label_w = by_labels_width(by_min, by_max);
    let graph_w_chars = chart_inner(layout[0], y_label_w).width as f64;
    let x_range = x_max - x_min;
    let cell_w = if graph_w_chars > 0.0 && x_range > 0.0 {
        x_range / graph_w_chars
    } else {
        1.0
    };

    let crosshair = compute_crosshair(
        app,
        layout[0],
        layout[1],
        x_min,
        x_max,
        ty_min,
        ty_max,
        by_min,
        by_max,
        y_label_w,
        &grouped_base_fee,
    );

    let gran_suffix = if g > 1 {
        format!(" ({}blk)", g)
    } else {
        String::new()
    };

    let tx_title = match &crosshair {
        Some(ch) => format!(
            "tx count per block{gran_suffix}  │  blk {:.0}  fee {:.3}",
            ch.data_x, ch.base_fee_y
        ),
        None => format!("tx count per block{gran_suffix}"),
    };

    let tx_graph_h = chart_inner(layout[0], y_label_w).height as f64;
    let ty_range = ty_max - ty_min;
    let cell_h = if tx_graph_h > 0.0 && ty_range > 0.0 {
        ty_range / tx_graph_h
    } else {
        1.0
    };
    let tx_overlay = build_tx_overlays(&grouped_filter_series, x_min, cell_w, ty_min, cell_h);
    let tx_datasets: Vec<Dataset<'_>> = tx_overlay
        .iter()
        .map(|(color, data)| {
            Dataset::default()
                .marker(ratatui::symbols::Marker::Braille)
                .graph_type(GraphType::Scatter)
                .style(Style::default().fg(*color))
                .data(data)
        })
        .collect();
    let ty_labels = y_labels_int(ty_min, ty_max);

    let tx_chart = Chart::new(tx_datasets)
        .block(Block::default().title(tx_title).borders(Borders::ALL))
        .x_axis(
            Axis::default()
                .bounds([x_min, x_max])
                .title("block")
                .labels(x_labels.clone()),
        )
        .y_axis(
            Axis::default()
                .bounds([ty_min, ty_max])
                .title("txs")
                .labels(ty_labels),
        );
    frame.render_widget(tx_chart, layout[0]);

    let by_labels = y_labels_gwei(by_min, by_max);

    let base_fee_by_x: HashMap<u64, f64> = grouped_base_fee
        .iter()
        .map(|(x, y)| (*x as u64, *y))
        .collect();

    let overlay_series =
        build_base_fee_overlays(&grouped_filter_series, &base_fee_by_x, x_min, cell_w);

    let base_fee_dataset = Dataset::default()
        .name("base fee")
        .marker(ratatui::symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::DarkGray))
        .data(&grouped_base_fee);

    let mut bf_datasets = vec![base_fee_dataset];
    for (color, series) in &overlay_series {
        bf_datasets.push(
            Dataset::default()
                .marker(ratatui::symbols::Marker::Braille)
                .graph_type(GraphType::Scatter)
                .style(Style::default().fg(*color))
                .data(series),
        );
    }

    let base_fee_chart = Chart::new(bf_datasets)
        .block(
            Block::default()
                .title(format!("base fee (gwei){gran_suffix}"))
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .bounds([x_min, x_max])
                .title("block")
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .bounds([by_min, by_max])
                .title("gwei")
                .labels(by_labels),
        );
    frame.render_widget(base_fee_chart, layout[1]);

    if let Some(ch) = &crosshair {
        draw_crosshair_highlight(
            frame, ch, layout[0], layout[1], x_min, x_max, by_min, by_max, y_label_w,
        );
    }

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(layout[2]);
    render_histogram(app, snapshot, frame, bottom[0]);
    render_sidebar(app, snapshot, frame, bottom[1]);
}

fn rebucket(raw: &[(f64, f64)], max_buckets: usize) -> Vec<(f64, f64, f64)> {
    if raw.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<(f64, f64)> = raw.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let target = max_buckets.max(1).min(sorted.len());
    let fee_min = sorted.first().unwrap().0;
    let fee_max = sorted.last().unwrap().0;

    if target >= sorted.len() || (fee_max - fee_min).abs() < 1e-9 {
        return sorted.iter().map(|(b, c)| (*b, *b, *c)).collect();
    }

    let step = (fee_max - fee_min) / target as f64;
    let mut merged: Vec<(f64, f64, f64)> = Vec::with_capacity(target);
    let mut lo = fee_min;
    let mut idx = 0;
    for _ in 0..target {
        let hi = lo + step;
        let mut count = 0.0;
        while idx < sorted.len() && sorted[idx].0 < hi + 1e-12 {
            count += sorted[idx].1;
            idx += 1;
        }
        if count > 0.0 {
            merged.push((lo, hi, count));
        }
        lo = hi;
    }
    while idx < sorted.len() {
        if let Some(last) = merged.last_mut() {
            last.2 += sorted[idx].1;
            last.1 = sorted[idx].0;
        }
        idx += 1;
    }
    merged
}

fn format_fee_label(lo: f64, hi: f64) -> String {
    if (hi - lo).abs() < 0.0005 {
        format!("{lo:.3}")
    } else {
        format!("{lo:.2}-{hi:.2}")
    }
}

fn render_histogram(
    app: &App,
    snapshot: &crate::domain::AnalysisSnapshot,
    frame: &mut Frame,
    area: Rect,
) {
    let inner_width = area.width.saturating_sub(2) as usize;
    let gap = 1usize;

    match app.hist_mode {
        HistogramMode::AllBlocks => {
            render_histogram_all_blocks(app, snapshot, frame, area, inner_width, gap);
        }
        HistogramMode::FilterMatches => {
            render_histogram_filter_matches(snapshot, frame, area, inner_width, gap);
        }
    }

    if !app.status_message.is_empty() {
        let status =
            Paragraph::new(app.status_message.as_str()).style(Style::default().fg(Color::LightRed));
        let status_area = Rect {
            x: area.x,
            y: area.y,
            width: area.width,
            height: 1,
        };
        frame.render_widget(status, status_area);
    }
}

fn render_histogram_filter_matches(
    snapshot: &crate::domain::AnalysisSnapshot,
    frame: &mut Frame,
    area: Rect,
    inner_width: usize,
    gap: usize,
) {
    let raw_hists: Vec<(&str, &[(f64, f64)])> = if snapshot.show_aggregate {
        vec![("", snapshot.aggregate_histogram.as_slice())]
    } else {
        snapshot
            .filters
            .iter()
            .filter(|f| f.enabled)
            .filter_map(|f| {
                snapshot
                    .filter_histograms
                    .get(&f.id)
                    .map(|h| (f.label.as_str(), h.as_slice()))
            })
            .collect()
    };

    let sample_label_len = 6;
    let bar_w = sample_label_len.max(3) as u16;
    let max_buckets = if (bar_w as usize + gap) > 0 {
        inner_width / (bar_w as usize + gap)
    } else {
        20
    }
    .max(2);

    let mut entries: Vec<(f64, &str, String, u64)> = Vec::new();
    for (prefix, hist) in &raw_hists {
        let merged = rebucket(hist, max_buckets);
        for (lo, hi, count) in &merged {
            let label = if prefix.is_empty() {
                format_fee_label(*lo, *hi)
            } else {
                let fee = format_fee_label(*lo, *hi);
                format!("{prefix}:{fee}")
            };
            entries.push((*lo, prefix, label, *count as u64));
        }
    }
    entries.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    let bars: Vec<(String, u64)> = entries.into_iter().map(|(_, _, l, c)| (l, c)).collect();

    let max_label_len = bars.iter().map(|(l, _)| l.len()).max().unwrap_or(3);
    let dynamic_bar_w = max_label_len.max(3) as u16;
    let bar_data: Vec<(&str, u64)> = bars.iter().map(|(l, v)| (l.as_str(), *v)).collect();

    let chart = BarChart::default()
        .block(
            Block::default()
                .title("base fee histogram — filter matches (gwei) [h: switch]")
                .borders(Borders::ALL),
        )
        .bar_width(dynamic_bar_w)
        .bar_gap(gap as u16)
        .data(&bar_data);
    frame.render_widget(chart, area);
}

fn bucket_overlaps(bucket_lo: f64, bucket_hi: f64, raw: &[(f64, f64)]) -> bool {
    raw.iter()
        .any(|(b, count)| *count > 0.0 && *b >= bucket_lo - 1e-9 && *b <= bucket_hi + 1e-9)
}

fn render_histogram_all_blocks(
    _app: &App,
    snapshot: &crate::domain::AnalysisSnapshot,
    frame: &mut Frame,
    area: Rect,
    inner_width: usize,
    gap: usize,
) {
    let sample_label_len = 6;
    let bar_w = sample_label_len.max(3) as u16;
    let max_buckets = if (bar_w as usize + gap) > 0 {
        inner_width / (bar_w as usize + gap)
    } else {
        20
    }
    .max(2);

    let merged = rebucket(&snapshot.all_blocks_histogram, max_buckets);

    let enabled_filters: Vec<_> = snapshot.filters.iter().filter(|f| f.enabled).collect();

    let styled_bars: Vec<(String, u64, Style)> = merged
        .iter()
        .map(|(lo, hi, count)| {
            let label = format_fee_label(*lo, *hi);
            let matching_colors: Vec<(u8, u8, u8)> = enabled_filters
                .iter()
                .filter(|f| {
                    snapshot
                        .filter_histograms
                        .get(&f.id)
                        .map(|h| bucket_overlaps(*lo, *hi, h))
                        .unwrap_or(false)
                })
                .map(|f| filter_rgb(f.color_index))
                .collect();

            let style = if matching_colors.is_empty() {
                Style::default().fg(Color::DarkGray)
            } else {
                let color = blend_colors(&matching_colors);
                Style::default().fg(color)
            };

            (label, *count as u64, style)
        })
        .collect();

    let max_label_len = styled_bars
        .iter()
        .map(|(l, _, _)| l.len())
        .max()
        .unwrap_or(3);
    let dynamic_bar_w = max_label_len.max(3) as u16;

    let bar_group: Vec<Bar> = styled_bars
        .iter()
        .map(|(label, value, style)| {
            Bar::default()
                .value(*value)
                .label(Line::from(label.as_str()))
                .style(*style)
        })
        .collect();

    let chart = BarChart::default()
        .block(
            Block::default()
                .title("base fee histogram — all blocks (gwei) [h: switch]")
                .borders(Borders::ALL),
        )
        .bar_width(dynamic_bar_w)
        .bar_gap(gap as u16)
        .data(BarGroup::default().bars(&bar_group));
    frame.render_widget(chart, area);
}

fn render_sidebar(
    app: &App,
    snapshot: &crate::domain::AnalysisSnapshot,
    frame: &mut Frame,
    area: Rect,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(4),
            Constraint::Min(3),
            Constraint::Length(3),
        ])
        .split(area);

    render_chunk_map(app, frame, chunks[0]);

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

    let hints = match app.mode {
        AppMode::Fetching | AppMode::Results => {
            "1-9: toggle  a: agg  g: gran  h: hist  l: logs  mouse: crosshair  q: quit"
        }
        _ => "l: logs  q: quit",
    };
    let hint = Paragraph::new(hints).block(Block::default().borders(Borders::TOP));
    frame.render_widget(hint, chunks[2]);
}

// Braille fill levels from empty to full, filling bottom-to-top.
// Both columns fill simultaneously so the bar rises evenly.
// Dot layout:  1 4    Bit values: 1   8
//              2 5                2  16
//              3 6                4  32
//              7 8               64 128
const BRAILLE_FILL: [char; 9] = [
    '\u{2800}', // 0/8: empty
    '\u{2880}', // 1/8: dots 8       (128)
    '\u{28C0}', // 2/8: dots 7,8     (64+128)
    '\u{28E0}', // 3/8: dots 6,7,8   (32+64+128)
    '\u{28E4}', // 4/8: dots 3,6,7,8 (4+32+64+128)
    '\u{28F4}', // 5/8: dots 3,5,6,7,8 (4+16+32+64+128)
    '\u{28F6}', // 6/8: dots 2,3,5,6,7,8 (2+4+16+32+64+128)
    '\u{28FE}', // 7/8: dots 2,3,4,5,6,7,8 (2+4+8+16+32+64+128)
    '\u{28FF}', // 8/8: all dots
];

fn chunk_braille(state: ChunkState, progress: f32) -> (char, Color) {
    match state {
        ChunkState::Pending => (BRAILLE_FILL[0], Color::DarkGray),
        ChunkState::Cached => (BRAILLE_FILL[8], Color::Blue),
        ChunkState::Fetching => {
            let level = (progress * 8.0).round() as usize;
            (BRAILLE_FILL[level.min(8)], Color::Yellow)
        }
        ChunkState::Done => (BRAILLE_FILL[8], Color::Green),
        ChunkState::Failed => (BRAILLE_FILL[8], Color::Red),
    }
}

fn is_complete(state: &ChunkState) -> bool {
    matches!(state, ChunkState::Done | ChunkState::Cached)
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
            let (ch, color) = chunk_braille(*state, progress);
            Span::styled(ch.to_string(), Style::default().fg(color))
        })
        .collect()
}

fn chunk_spans_windowed(app: &App, capacity: usize) -> Vec<Span<'static>> {
    let total = app.chunk_states.len();
    let ellipsis = Span::styled("…", Style::default().fg(Color::DarkGray));

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

    let context = 3;
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
            let (ch, color) = chunk_braille(app.chunk_states[i], progress);
            spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
        }
        if window_end < total {
            spans.push(ellipsis);
        }
        spans
    } else {
        let usable = capacity.saturating_sub(1);
        let start = first_incomplete
            .saturating_sub(context)
            .min(total.saturating_sub(usable));
        let end = (start + usable).min(total);
        let mut spans = Vec::new();
        if start > 0 {
            spans.push(ellipsis.clone());
        }
        for i in start..end {
            let progress = app.chunk_progress.get(i).copied().unwrap_or(0.0);
            let (ch, color) = chunk_braille(app.chunk_states[i], progress);
            spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
        }
        if end < total {
            spans.push(ellipsis);
        }
        spans
    }
}

fn series_x_bounds(series: &[(f64, f64)]) -> (f64, f64) {
    if series.is_empty() {
        return (0.0, 1.0);
    }
    let min = series.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
    let max = series
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < f64::EPSILON {
        (min, min + 1.0)
    } else {
        (min, max)
    }
}

fn series_y_bounds(datasets: &[&[(f64, f64)]]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for series in datasets {
        for (_, y) in *series {
            min = min.min(*y);
            max = max.max(*y);
        }
    }
    if min == f64::INFINITY {
        return (0.0, 1.0);
    }
    if (max - min).abs() < f64::EPSILON {
        (0.0, max + 1.0)
    } else {
        (0.0, max * 1.05)
    }
}

fn render_log_panel(app: &App, frame: &mut Frame, area: Rect) {
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

fn group_series_sum(series: &[(f64, f64)], granularity: usize) -> Vec<(f64, f64)> {
    if granularity <= 1 || series.is_empty() {
        return series.to_vec();
    }
    series
        .chunks(granularity)
        .map(|chunk| {
            let x = chunk.iter().map(|(x, _)| *x).sum::<f64>() / chunk.len() as f64;
            let y = chunk.iter().map(|(_, y)| *y).sum::<f64>();
            (x, y)
        })
        .collect()
}

fn group_series_avg(series: &[(f64, f64)], granularity: usize) -> Vec<(f64, f64)> {
    if granularity <= 1 || series.is_empty() {
        return series.to_vec();
    }
    series
        .chunks(granularity)
        .map(|chunk| {
            let x = chunk.iter().map(|(x, _)| *x).sum::<f64>() / chunk.len() as f64;
            let y = chunk.iter().map(|(_, y)| *y).sum::<f64>() / chunk.len() as f64;
            (x, y)
        })
        .collect()
}

struct Crosshair {
    data_x: f64,
    base_fee_y: f64,
}

fn by_labels_width(by_min: f64, by_max: f64) -> u16 {
    let sample = format!("{:.3}", by_max.max(by_min.abs()));
    (sample.len() + 2) as u16
}

fn compute_crosshair(
    app: &App,
    tx_rect: Rect,
    bf_rect: Rect,
    x_min: f64,
    x_max: f64,
    _ty_min: f64,
    _ty_max: f64,
    _by_min: f64,
    _by_max: f64,
    y_label_w: u16,
    grouped_base_fee: &[(f64, f64)],
) -> Option<Crosshair> {
    let col = app.mouse_col;
    let row = app.mouse_row;

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
    let inner = chart_inner(chart_rect, y_label_w);

    if col < inner.x || col >= inner.x + inner.width || inner.width == 0 {
        return None;
    }

    let graph_w = inner.width.saturating_sub(1).max(1) as f64;
    let frac = ((col - inner.x) as f64) / graph_w;
    let frac = frac.clamp(0.0, 1.0);
    let data_x = x_min + frac * (x_max - x_min);

    let base_fee_y = nearest_y(grouped_base_fee, data_x);

    Some(Crosshair { data_x, base_fee_y })
}

fn chart_inner(chart_rect: Rect, y_label_w: u16) -> Rect {
    // Must match ratatui's Chart::layout() graph_area calculation:
    //   block border:     1 row top, 1 row bottom, 1 col left, 1 col right
    //   y-axis labels:    y_label_w columns
    //   y-axis line:      1 column
    //   x-axis labels:    1 row
    //   x-axis line:      1 row
    let left = chart_rect.x + 1 + y_label_w + 1;
    let right = chart_rect.x + chart_rect.width.saturating_sub(1);
    let top = chart_rect.y + 1;
    let bottom = chart_rect.y + chart_rect.height.saturating_sub(4);
    let w = right.saturating_sub(left);
    let h = bottom.saturating_sub(top).saturating_add(1);
    Rect::new(left, top, w, h)
}

fn data_to_col(data_x: f64, x_min: f64, x_max: f64, inner: Rect) -> Option<u16> {
    let x_range = x_max - x_min;
    if x_range <= 0.0 || inner.width == 0 {
        return None;
    }
    let frac = (data_x - x_min) / x_range;
    if !(0.0..=1.0).contains(&frac) {
        return None;
    }
    Some(inner.x + (frac * (inner.width.saturating_sub(1)) as f64).round() as u16)
}

fn data_to_row(data_y: f64, y_min: f64, y_max: f64, inner: Rect) -> Option<u16> {
    let y_range = y_max - y_min;
    if y_range <= 0.0 || inner.height == 0 {
        return None;
    }
    let frac = (data_y - y_min) / y_range;
    if !(0.0..=1.0).contains(&frac) {
        return None;
    }
    Some(
        inner.y + inner.height.saturating_sub(1)
            - (frac * (inner.height.saturating_sub(1)) as f64).round() as u16,
    )
}

const CROSSHAIR_BG: Color = Color::Rgb(40, 40, 50);
const CROSSHAIR_INTERSECT_BG: Color = Color::Rgb(60, 60, 75);

fn highlight_cell(buf: &mut ratatui::buffer::Buffer, col: u16, row: u16, bg: Color) {
    if col < buf.area.width && row < buf.area.height {
        let cell = &mut buf[(col, row)];
        cell.set_bg(bg);
    }
}

fn draw_crosshair_highlight(
    frame: &mut Frame,
    ch: &Crosshair,
    tx_rect: Rect,
    bf_rect: Rect,
    x_min: f64,
    x_max: f64,
    by_min: f64,
    by_max: f64,
    y_label_w: u16,
) {
    let tx_inner = chart_inner(tx_rect, y_label_w);
    let bf_inner = chart_inner(bf_rect, y_label_w);

    let tx_col = data_to_col(ch.data_x, x_min, x_max, tx_inner);
    let bf_col = data_to_col(ch.data_x, x_min, x_max, bf_inner);
    let bf_row = data_to_row(ch.base_fee_y, by_min, by_max, bf_inner);

    let buf = frame.buffer_mut();

    if let Some(col) = tx_col {
        for row in tx_inner.y..tx_inner.y + tx_inner.height {
            highlight_cell(buf, col, row, CROSSHAIR_BG);
        }
    }

    if let Some(col) = bf_col {
        for row in bf_inner.y..bf_inner.y + bf_inner.height {
            let bg = if bf_row == Some(row) {
                CROSSHAIR_INTERSECT_BG
            } else {
                CROSSHAIR_BG
            };
            highlight_cell(buf, col, row, bg);
        }
    }

    if let Some(row) = bf_row {
        for col in bf_inner.x..bf_inner.x + bf_inner.width {
            let bg = if bf_col == Some(col) {
                CROSSHAIR_INTERSECT_BG
            } else {
                CROSSHAIR_BG
            };
            highlight_cell(buf, col, row, bg);
        }
    }
}

fn nearest_y(series: &[(f64, f64)], target_x: f64) -> f64 {
    if series.is_empty() {
        return 0.0;
    }
    let mut best = &series[0];
    let mut best_dist = (best.0 - target_x).abs();
    for pt in series.iter().skip(1) {
        let dist = (pt.0 - target_x).abs();
        if dist < best_dist {
            best = pt;
            best_dist = dist;
        }
    }
    best.1
}

fn quantize(val: f64, min: f64, cell_size: f64) -> i64 {
    if cell_size <= 0.0 {
        return val as i64;
    }
    ((val - min) / cell_size).floor() as i64
}

fn cell_color(key: &[usize], grouped_filter_series: &[(String, usize, Vec<(f64, f64)>)]) -> Color {
    if key.len() == 1 {
        filter_color(grouped_filter_series[key[0]].1)
    } else {
        let rgbs: Vec<(u8, u8, u8)> = key
            .iter()
            .map(|&i| filter_rgb(grouped_filter_series[i].1))
            .collect();
        blend_colors(&rgbs)
    }
}

fn build_tx_overlays(
    grouped_filter_series: &[(String, usize, Vec<(f64, f64)>)],
    x_min: f64,
    cell_w: f64,
    y_min: f64,
    cell_h: f64,
) -> Vec<(Color, Vec<(f64, f64)>)> {
    let mut cell_filters: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, (_label, _color_idx, series)) in grouped_filter_series.iter().enumerate() {
        for &(x, y) in series {
            if y > 0.0 {
                let cx = quantize(x, x_min, cell_w);
                let cy = quantize(y, y_min, cell_h);
                let entry = cell_filters.entry((cx, cy)).or_default();
                if !entry.contains(&i) {
                    entry.push(i);
                }
            }
        }
    }

    let mut cell_colors: HashMap<(i64, i64), Color> = HashMap::new();
    for (cell, mut indices) in cell_filters {
        indices.sort();
        indices.dedup();
        cell_colors.insert(cell, cell_color(&indices, grouped_filter_series));
    }

    let mut color_points: HashMap<Color, Vec<(f64, f64)>> = HashMap::new();
    for (_i, (_label, _color_idx, series)) in grouped_filter_series.iter().enumerate() {
        for &(x, y) in series {
            if y > 0.0 {
                let cx = quantize(x, x_min, cell_w);
                let cy = quantize(y, y_min, cell_h);
                if let Some(&color) = cell_colors.get(&(cx, cy)) {
                    color_points.entry(color).or_default().push((x, y));
                }
            }
        }
    }

    let mut result: Vec<(Color, Vec<(f64, f64)>)> = color_points.into_iter().collect();
    for (_, series) in &mut result {
        series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }
    result
}

fn build_base_fee_overlays(
    grouped_filter_series: &[(String, usize, Vec<(f64, f64)>)],
    base_fee_by_x: &HashMap<u64, f64>,
    x_min: f64,
    cell_w: f64,
) -> Vec<(Color, Vec<(f64, f64)>)> {
    let mut cell_filters: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, (_label, _color_idx, series)) in grouped_filter_series.iter().enumerate() {
        for &(x, y) in series {
            if y > 0.0 {
                let cx = quantize(x, x_min, cell_w);
                let entry = cell_filters.entry(cx).or_default();
                if !entry.contains(&i) {
                    entry.push(i);
                }
            }
        }
    }

    let mut color_buckets: HashMap<Color, Vec<(f64, f64)>> = HashMap::new();
    for (i, (_label, _color_idx, series)) in grouped_filter_series.iter().enumerate() {
        for &(x, y) in series {
            if y > 0.0 {
                let cx = quantize(x, x_min, cell_w);
                let mut indices: Vec<usize> = cell_filters.get(&cx).cloned().unwrap_or_default();
                indices.sort();
                indices.dedup();
                if !indices.contains(&i) {
                    continue;
                }
                let color = cell_color(&indices, grouped_filter_series);
                if let Some(&base_fee) = base_fee_by_x.get(&(x as u64)) {
                    color_buckets.entry(color).or_default().push((x, base_fee));
                }
            }
        }
    }

    let mut result: Vec<(Color, Vec<(f64, f64)>)> = color_buckets.into_iter().collect();
    for (_, series) in &mut result {
        series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }
    result
}

fn x_axis_labels(x_min: f64, x_max: f64, available_chars: usize) -> Vec<String> {
    let range = x_max - x_min;
    if range <= 0.0 {
        return vec![format!("{:.0}", x_min)];
    }
    let max_labels = (available_chars / 12).max(2).min(7);
    let step = range / (max_labels - 1) as f64;
    (0..max_labels)
        .map(|i| format!("{:.0}", x_min + step * i as f64))
        .collect()
}

fn y_labels_int(y_min: f64, y_max: f64) -> Vec<String> {
    let mid = (y_min + y_max) / 2.0;
    let q1 = (y_min + mid) / 2.0;
    let q3 = (mid + y_max) / 2.0;
    vec![
        format!("{:.0}", y_min),
        format!("{:.0}", q1),
        format!("{:.0}", mid),
        format!("{:.0}", q3),
        format!("{:.0}", y_max),
    ]
}

fn y_labels_gwei(y_min: f64, y_max: f64) -> Vec<String> {
    let mid = (y_min + y_max) / 2.0;
    let q1 = (y_min + mid) / 2.0;
    let q3 = (mid + y_max) / 2.0;
    vec![
        format!("{:.3}", y_min),
        format!("{:.3}", q1),
        format!("{:.3}", mid),
        format!("{:.3}", q3),
        format!("{:.3}", y_max),
    ]
}

fn filter_color(index: usize) -> Color {
    let palette = FILTER_PALETTE;
    palette[index % palette.len()]
}

const FILTER_PALETTE: [Color; 6] = [
    Color::Rgb(255, 0, 0),
    Color::Rgb(0, 0, 255),
    Color::Rgb(0, 200, 0),
    Color::Rgb(255, 255, 0),
    Color::Rgb(0, 220, 220),
    Color::Rgb(200, 0, 255),
];

fn filter_rgb(index: usize) -> (u8, u8, u8) {
    match FILTER_PALETTE[index % FILTER_PALETTE.len()] {
        Color::Rgb(r, g, b) => (r, g, b),
        _ => (255, 255, 255),
    }
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let rf = r as f64 / 255.0;
    let gf = g as f64 / 255.0;
    let bf = b as f64 / 255.0;
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let delta = max - min;

    let h = if delta < 1e-9 {
        0.0
    } else if (max - rf).abs() < 1e-9 {
        60.0 * (((gf - bf) / delta) % 6.0)
    } else if (max - gf).abs() < 1e-9 {
        60.0 * (((bf - rf) / delta) + 2.0)
    } else {
        60.0 * (((rf - gf) / delta) + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    let s = if max < 1e-9 { 0.0 } else { delta / max };
    (h, s, max)
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - ((hp % 2.0) - 1.0).abs());
    let (r1, g1, b1) = if hp < 1.0 {
        (c, x, 0.0)
    } else if hp < 2.0 {
        (x, c, 0.0)
    } else if hp < 3.0 {
        (0.0, c, x)
    } else if hp < 4.0 {
        (0.0, x, c)
    } else if hp < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = v - c;
    (
        ((r1 + m) * 255.0).round() as u8,
        ((g1 + m) * 255.0).round() as u8,
        ((b1 + m) * 255.0).round() as u8,
    )
}

fn blend_colors(colors: &[(u8, u8, u8)]) -> Color {
    if colors.is_empty() {
        return Color::White;
    }
    if colors.len() == 1 {
        return Color::Rgb(colors[0].0, colors[0].1, colors[0].2);
    }
    // HSV blend: average hues (circular mean), take max saturation and value.
    // This gives perceptually correct blends (red+blue=purple, not pink).
    let n = colors.len() as f64;
    let mut sin_sum = 0.0_f64;
    let mut cos_sum = 0.0_f64;
    let mut s_max = 0.0_f64;
    let mut v_max = 0.0_f64;
    for &(r, g, b) in colors {
        let (h, s, v) = rgb_to_hsv(r, g, b);
        let h_rad = h.to_radians();
        sin_sum += h_rad.sin();
        cos_sum += h_rad.cos();
        s_max = s_max.max(s);
        v_max = v_max.max(v);
    }
    let avg_h = (sin_sum / n).atan2(cos_sum / n).to_degrees();
    let avg_h = if avg_h < 0.0 { avg_h + 360.0 } else { avg_h };
    let (r, g, b) = hsv_to_rgb(avg_h, s_max, v_max);
    Color::Rgb(r, g, b)
}
