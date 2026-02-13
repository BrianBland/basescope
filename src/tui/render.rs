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

use crate::domain::{approx_head_block, BASE_BLOCK_TIME_SECS, BASE_GENESIS_TIMESTAMP};
use crate::tui::{App, AppMode, BottomPanel, ChunkState, Granularity, HistogramMode, RangeField};

type FilterEntry = (String, usize, Vec<(f64, f64)>);
type FilterSeries = Vec<FilterEntry>;
type HistSlices<'a> = Vec<(&'a str, usize, &'a [(f64, f64)])>;

const BOTTOM_PANEL_HEIGHT: u16 = 8;

pub fn render(app: &App, frame: &mut Frame) {
    let outer = frame.area();

    let all_done = !app.chunk_states.is_empty() && app.chunk_states.iter().all(is_complete);

    let effective_panel = match app.bottom_panel {
        BottomPanel::Logs if all_done && app.log_buffer.is_empty() => BottomPanel::Hidden,
        other => other,
    };

    match effective_panel {
        BottomPanel::Logs => {
            let layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(BOTTOM_PANEL_HEIGHT)])
                .split(outer);
            render_main(app, frame, layout[0]);
            render_log_panel(app, frame, layout[1]);
        }
        BottomPanel::Rpc => {
            let layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(BOTTOM_PANEL_HEIGHT)])
                .split(outer);
            render_main(app, frame, layout[0]);
            render_rpc_panel(app, frame, layout[1]);
        }
        BottomPanel::Hidden => {
            render_main(app, frame, outer);
        }
    }

    if app.show_help {
        render_help_panel(app, frame, outer);
    }

    if app.granularity_input.is_some() {
        render_granularity_input(app, frame, outer);
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

    let instructions = Paragraph::new("?: help").block(Block::default().borders(Borders::TOP));
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

    let instructions = Paragraph::new("?: help").block(Block::default().borders(Borders::TOP));
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
        .horizontal_margin(1)
        .vertical_margin(0)
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

    let g = app.effective_granularity();
    let (x_min, x_max) = series_x_bounds(&snapshot.base_fee_series);
    let x_labels = x_axis_labels(x_min, x_max, layout[0].width.saturating_sub(8) as usize);

    let grouped_filter_series: FilterSeries = snapshot
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

    let gran_suffix = app.granularity_label();

    let tx_title = match &crosshair {
        Some(ch) => {
            let fee_str = format_fee_value(ch.base_fee_y);
            format!(
                "tx count per block{gran_suffix}  │  blk {:.0}  fee {fee_str}",
                ch.data_x
            )
        }
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
                .title(format!("base fee{gran_suffix}"))
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
                .title("fee")
                .labels(by_labels),
        );
    frame.render_widget(base_fee_chart, layout[1]);

    let tx_inner = chart_inner(layout[0], y_label_w);
    let bf_inner = chart_inner(layout[1], y_label_w);
    paint_time_of_day_bg(frame, tx_inner, x_min, x_max);
    paint_time_of_day_bg(frame, bf_inner, x_min, x_max);

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

    if target >= sorted.len() {
        return sorted.iter().map(|(b, c)| (*b, *b, *c)).collect();
    }

    let total_count: f64 = sorted.iter().map(|(_, c)| *c).sum();
    let count_per_bucket = total_count / target as f64;

    let mut merged: Vec<(f64, f64, f64)> = Vec::with_capacity(target);
    let mut lo = sorted[0].0;
    let mut hi = lo;
    let mut accum = 0.0;

    for &(fee, count) in &sorted {
        if accum >= count_per_bucket && merged.len() < target - 1 {
            merged.push((lo, hi, accum));
            lo = fee;
            accum = 0.0;
        }
        hi = fee;
        accum += count;
    }
    if accum > 0.0 {
        merged.push((lo, hi, accum));
    }
    merged
}

#[derive(Clone, Copy)]
enum FeeUnit {
    Gwei,
    Mwei,
}

fn pick_fee_unit(max_gwei: f64) -> FeeUnit {
    if max_gwei.abs() < 1.0 {
        FeeUnit::Mwei
    } else {
        FeeUnit::Gwei
    }
}

fn format_fee_with_unit(gwei: f64, unit: FeeUnit) -> String {
    match unit {
        FeeUnit::Gwei => {
            let s = strip_trailing_zeros(&format!("{gwei:.3}"));
            format!("{s}G")
        }
        FeeUnit::Mwei => {
            let mwei = gwei * 1000.0;
            let s = strip_trailing_zeros(&format!("{mwei:.1}"));
            format!("{s}M")
        }
    }
}

fn format_fee_value(gwei: f64) -> String {
    format_fee_with_unit(gwei, pick_fee_unit(gwei))
}

fn strip_trailing_zeros(s: &str) -> String {
    if let Some(dot) = s.find('.') {
        let trimmed = s.trim_end_matches('0');
        if trimmed.ends_with('.') {
            trimmed[..dot].to_string()
        } else {
            trimmed.to_string()
        }
    } else {
        s.to_string()
    }
}

fn truncate_to(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else if max >= 2 {
        let keep = max.saturating_sub(1);
        let mut t: String = s.chars().take(keep).collect();
        t.push('…');
        t
    } else if max == 1 {
        s.chars().next().map(|c| c.to_string()).unwrap_or_default()
    } else {
        String::new()
    }
}

fn format_fee_label(lo: f64, hi: f64, unit: FeeUnit) -> String {
    if (hi - lo).abs() < 0.0005 {
        format_fee_with_unit(lo, unit)
    } else {
        let lo_s = format_fee_with_unit(lo, unit);
        let hi_s = format_fee_with_unit(hi, unit);
        format!("{lo_s}-{hi_s}")
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
    let raw_hists: HistSlices<'_> = if snapshot.show_aggregate {
        vec![("", 0, snapshot.aggregate_histogram.as_slice())]
    } else {
        snapshot
            .filters
            .iter()
            .filter(|f| f.enabled)
            .filter_map(|f| {
                snapshot
                    .filter_histograms
                    .get(&f.id)
                    .map(|h| (f.label.as_str(), f.color_index, h.as_slice()))
            })
            .collect()
    };

    let bar_w: u16 = 3;
    let max_buckets = if (bar_w as usize + gap) > 0 {
        inner_width / (bar_w as usize + gap)
    } else {
        20
    }
    .max(2);

    let mut all_merged: Vec<(f64, f64, f64, &str, usize)> = Vec::new();
    for (prefix, color_idx, hist) in &raw_hists {
        let merged = rebucket(hist, max_buckets);
        for (lo, hi, count) in merged {
            all_merged.push((lo, hi, count, prefix, *color_idx));
        }
    }
    let hist_max = all_merged
        .iter()
        .map(|(_, hi, _, _, _)| *hi)
        .fold(0.0_f64, f64::max);
    let unit = pick_fee_unit(hist_max);

    let mut entries: Vec<(f64, &str, String, String, u64, usize)> = Vec::new();
    for (lo, hi, count, prefix, color_idx) in &all_merged {
        let fee = format_fee_label(*lo, *hi, unit);
        entries.push((
            *lo,
            prefix,
            fee,
            prefix.to_string(),
            *count as u64,
            *color_idx,
        ));
    }
    entries.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(b.1))
    });

    let bar_group: Vec<Bar> = entries
        .iter()
        .map(|(_, _, fee, prefix, value, color_idx)| {
            let color = if snapshot.show_aggregate {
                Color::White
            } else {
                filter_color(*color_idx)
            };
            let w = bar_w as usize;
            let label = if prefix.is_empty() {
                truncate_to(fee, w)
            } else {
                let fee_len = fee.len();
                if fee_len >= w {
                    truncate_to(fee, w)
                } else {
                    let remaining = w - fee_len;
                    if remaining >= 2 {
                        let pfx = truncate_to(prefix, remaining);
                        format!("{pfx}{fee}")
                    } else {
                        truncate_to(fee, w)
                    }
                }
            };
            Bar::default()
                .value(*value)
                .label(Line::from(label))
                .style(Style::default().fg(color))
        })
        .collect();

    let chart = BarChart::default()
        .block(
            Block::default()
                .title("base fee histogram — filter matches [h: switch]")
                .borders(Borders::ALL),
        )
        .bar_width(bar_w)
        .bar_gap(gap as u16)
        .data(BarGroup::default().bars(&bar_group));
    frame.render_widget(chart, area);
}

fn bucket_overlaps(bucket_lo: f64, bucket_hi: f64, raw: &[(f64, f64)]) -> bool {
    raw.iter()
        .any(|(b, count)| *count > 0.0 && *b >= bucket_lo - 1e-9 && *b <= bucket_hi + 1e-9)
}

fn render_histogram_all_blocks(
    app: &App,
    snapshot: &crate::domain::AnalysisSnapshot,
    frame: &mut Frame,
    area: Rect,
    inner_width: usize,
    gap: usize,
) {
    let g = app.effective_granularity();
    let sample_label_len = 6;
    let bar_w = sample_label_len.max(3) as u16;
    let max_buckets = if (bar_w as usize + gap) > 0 {
        inner_width / (bar_w as usize + gap)
    } else {
        20
    }
    .max(2);

    let hist_data: Vec<(f64, f64)> = if g > 1 {
        let grouped = group_series_avg(&snapshot.base_fee_series, g);
        let mut hist: Vec<(f64, f64)> = Vec::new();
        for &(_block, avg_fee) in &grouped {
            let bucket = (avg_fee * 1000.0).floor() / 1000.0;
            if let Some((_, count)) = hist.iter_mut().find(|(b, _)| (*b - bucket).abs() < 1e-9) {
                *count += 1.0;
            } else {
                hist.push((bucket, 1.0));
            }
        }
        hist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        hist
    } else {
        snapshot.all_blocks_histogram.clone()
    };

    let merged = rebucket(&hist_data, max_buckets);
    let hist_max = merged.iter().map(|(_, hi, _)| *hi).fold(0.0_f64, f64::max);
    let unit = pick_fee_unit(hist_max);

    let enabled_filters: Vec<_> = snapshot.filters.iter().filter(|f| f.enabled).collect();

    let styled_bars: Vec<(String, u64, Style)> = merged
        .iter()
        .map(|(lo, hi, count)| {
            let label = format_fee_label(*lo, *hi, unit);
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

    let gran_suffix = app.granularity_label();
    let title = format!("base fee histogram — all blocks{gran_suffix} [h: switch]");

    let chart = BarChart::default()
        .block(Block::default().title(title).borders(Borders::ALL))
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

fn render_rpc_panel(app: &App, frame: &mut Frame, area: Rect) {
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

    let mut entries: Vec<(&str, &str)> = vec![("?", "toggle this help"), ("q", "quit")];

    match mode {
        AppMode::RangeInput => {
            entries.push(("Tab", "switch field"));
            entries.push(("Enter", "continue"));
        }
        AppMode::FilterInput => {
            entries.push(("Enter", "add filter / start scan"));
            entries.push(("d", "delete selected filter"));
            entries.push(("↑/↓", "select filter"));
        }
        AppMode::Fetching | AppMode::Results => {
            entries.push(("1-9", "toggle filter"));
            entries.push(("a", "aggregate mode"));
            entries.push(("g", "cycle granularity"));
            entries.push(("G", "set granularity"));
            entries.push(("h", "switch histogram"));
            entries.push(("l", "toggle logs"));
            entries.push(("r", "toggle rpc info"));
            entries.push(("mouse", "crosshair"));
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

fn render_help_panel(app: &App, frame: &mut Frame, outer: Rect) {
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

fn render_granularity_input(app: &App, frame: &mut Frame, outer: Rect) {
    let panel_w = 32u16.min(outer.width);
    let panel_h = 3u16;
    let area = Rect {
        x: outer.x + (outer.width.saturating_sub(panel_w)) / 2,
        y: outer.y + (outer.height.saturating_sub(panel_h)) / 2,
        width: panel_w,
        height: panel_h,
    };

    frame.render_widget(ratatui::widgets::Clear, area);

    let input_text = app.granularity_input.as_deref().unwrap_or("");
    let hint = if input_text.is_empty() {
        match app.granularity {
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

#[allow(clippy::too_many_arguments)]
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

fn local_utc_offset_secs() -> i64 {
    use std::sync::OnceLock;
    static OFFSET: OnceLock<i64> = OnceLock::new();
    *OFFSET.get_or_init(|| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        unsafe { libc::localtime_r(&now, &mut tm) };
        tm.tm_gmtoff as i64
    })
}

fn block_to_hour_frac(block_number: f64) -> f64 {
    let utc_timestamp = BASE_GENESIS_TIMESTAMP as f64 + block_number * BASE_BLOCK_TIME_SECS as f64;
    let offset = local_utc_offset_secs() as f64;
    let local_secs = ((utc_timestamp + offset) % 86400.0 + 86400.0) % 86400.0;
    local_secs / 86400.0
}

fn time_of_day_bg(hour_frac: f64) -> Color {
    let angle = (hour_frac - 0.5) * std::f64::consts::TAU;
    let day = (angle.cos() + 1.0) / 2.0;

    let r = (15.0 + day * 25.0) as u8;
    let g = (15.0 + day * 20.0) as u8;
    let b = (35.0 - day * 15.0) as u8;
    Color::Rgb(r, g, b)
}

fn paint_time_of_day_bg(frame: &mut Frame, inner: Rect, x_min: f64, x_max: f64) {
    let buf = frame.buffer_mut();
    let x_range = x_max - x_min;
    if x_range <= 0.0 || inner.width == 0 {
        return;
    }
    for col in inner.x..inner.x + inner.width {
        let frac = (col - inner.x) as f64 / inner.width.saturating_sub(1).max(1) as f64;
        let block = x_min + frac * x_range;
        let hour_frac = block_to_hour_frac(block);
        let bg = time_of_day_bg(hour_frac);
        for row in inner.y..inner.y + inner.height {
            if col < buf.area.x + buf.area.width && row < buf.area.y + buf.area.height {
                let cell = &mut buf[(col, row)];
                cell.set_bg(bg);
            }
        }
    }
}

const CROSSHAIR_BG: Color = Color::Rgb(40, 40, 50);
const CROSSHAIR_INTERSECT_BG: Color = Color::Rgb(60, 60, 75);

fn highlight_cell(buf: &mut ratatui::buffer::Buffer, col: u16, row: u16, bg: Color) {
    if col < buf.area.width && row < buf.area.height {
        let cell = &mut buf[(col, row)];
        cell.set_bg(bg);
    }
}

#[allow(clippy::too_many_arguments)]
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

fn cell_color(key: &[usize], grouped_filter_series: &[FilterEntry]) -> Color {
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
    grouped_filter_series: &[FilterEntry],
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
    for (_label, _color_idx, series) in grouped_filter_series {
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
    grouped_filter_series: &[FilterEntry],
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
    let max_labels = (available_chars / 12).clamp(2, 7);
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
    let unit = pick_fee_unit(y_max);
    let mid = (y_min + y_max) / 2.0;
    let q1 = (y_min + mid) / 2.0;
    let q3 = (mid + y_max) / 2.0;
    vec![
        format_fee_with_unit(y_min, unit),
        format_fee_with_unit(q1, unit),
        format_fee_with_unit(mid, unit),
        format_fee_with_unit(q3, unit),
        format_fee_with_unit(y_max, unit),
    ]
}

fn filter_color(index: usize) -> Color {
    let palette = FILTER_PALETTE;
    palette[index % palette.len()]
}

const FILTER_PALETTE: [Color; 6] = [
    Color::Rgb(255, 0, 0),
    Color::Rgb(0, 0, 255),
    Color::Rgb(0, 255, 0),
    Color::Rgb(255, 255, 0),
    Color::Rgb(0, 255, 255),
    Color::Rgb(255, 0, 255),
];

fn filter_rgb(index: usize) -> (u8, u8, u8) {
    match FILTER_PALETTE[index % FILTER_PALETTE.len()] {
        Color::Rgb(r, g, b) => (r, g, b),
        _ => (255, 255, 255),
    }
}

fn blend_colors(colors: &[(u8, u8, u8)]) -> Color {
    if colors.is_empty() {
        return Color::White;
    }
    if colors.len() == 1 {
        return Color::Rgb(colors[0].0, colors[0].1, colors[0].2);
    }
    let n = colors.len() as f64;
    let mut r_sum = 0.0_f64;
    let mut g_sum = 0.0_f64;
    let mut b_sum = 0.0_f64;
    for &(r, g, b) in colors {
        r_sum += (r as f64 / 255.0).powi(2);
        g_sum += (g as f64 / 255.0).powi(2);
        b_sum += (b as f64 / 255.0).powi(2);
    }
    let r = ((r_sum / n).sqrt() * 255.0).round() as u8;
    let g = ((g_sum / n).sqrt() * 255.0).round() as u8;
    let b = ((b_sum / n).sqrt() * 255.0).round() as u8;
    Color::Rgb(r, g, b)
}
