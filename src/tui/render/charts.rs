use std::collections::HashMap;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Frame;
use ratatui::style::{Color, Style};
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph};

use crate::domain::{BASE_BLOCK_TIME_SECS, BASE_GENESIS_TIMESTAMP};
use crate::tui::App;

use super::colors::{blend_colors, filter_color, filter_rgb};
use super::histogram::render_histogram;
use super::sidebar::render_sidebar;
use super::{
    by_labels_width, chart_inner, filter_visible, format_fee_value, group_series_avg,
    group_series_sum, nearest_y, scaled_y_labels_gwei, series_x_bounds, series_y_bounds,
    x_axis_labels, y_labels_int, FilterEntry, FilterSeries,
};

pub(super) fn render_results(app: &App, frame: &mut Frame, area: Rect) {
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

    let (full_x_min, full_x_max) = series_x_bounds(&snapshot.base_fee_series);
    app.full_x_range.set((full_x_min, full_x_max));
    let x_min = app.view_start.unwrap_or(full_x_min);
    let x_max = app.view_end.unwrap_or(full_x_max);
    let visible_range = (x_max - x_min).max(1.0) as usize;
    app.auto_granularity
        .set(crate::tui::auto_granularity(visible_range));
    let g = app.effective_granularity();
    let x_labels = x_axis_labels(x_min, x_max, layout[0].width.saturating_sub(8) as usize);

    let grouped_filter_series: FilterSeries = snapshot
        .filters
        .iter()
        .filter(|f| f.enabled)
        .filter_map(|f| {
            snapshot.filter_series.get(&f.id).map(|series| {
                let visible = filter_visible(series, x_min, x_max);
                (f.label.clone(), f.color_index, group_series_sum(visible, g))
            })
        })
        .collect();

    let visible_base_fee = filter_visible(&snapshot.base_fee_series, x_min, x_max);
    let grouped_base_fee = group_series_avg(visible_base_fee, g);
    let scale = app.scale_mode.build_transform(&grouped_base_fee);
    let scaled_base_fee: Vec<(f64, f64)> = grouped_base_fee
        .iter()
        .map(|(x, y)| (*x, scale.apply(*y)))
        .collect();
    let (by_min, by_max) = series_y_bounds(&[&scaled_base_fee]);

    let tx_series_refs: Vec<&[(f64, f64)]> = grouped_filter_series
        .iter()
        .map(|(_, _, v)| v.as_slice())
        .collect();
    let (ty_min, ty_max) = series_y_bounds(&tx_series_refs);

    let original_by_max = scale.invert(by_max);
    let original_by_min = scale.invert(by_min);
    let y_label_w = by_labels_width(original_by_min, original_by_max);
    app.last_y_label_w.set(y_label_w);
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
    let scaled_crosshair_y = crosshair.as_ref().map(|ch| scale.apply(ch.base_fee_y));

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

    let by_labels = scaled_y_labels_gwei(by_min, by_max, &scale);

    let scaled_base_fee_by_x: HashMap<u64, f64> = scaled_base_fee
        .iter()
        .map(|(x, y)| (*x as u64, *y))
        .collect();

    let overlay_series =
        build_base_fee_overlays(&grouped_filter_series, &scaled_base_fee_by_x, x_min, cell_w);

    let base_fee_dataset = Dataset::default()
        .name("base fee")
        .marker(ratatui::symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::DarkGray))
        .data(&scaled_base_fee);

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

    let scale_suffix = app.scale_mode.label();
    let base_fee_chart = Chart::new(bf_datasets)
        .block(
            Block::default()
                .title(format!("base fee{gran_suffix}{scale_suffix}"))
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
        let scaled_fee_y = scaled_crosshair_y.unwrap_or(0.0);
        draw_crosshair_highlight(
            frame,
            ch.data_x,
            scaled_fee_y,
            layout[0],
            layout[1],
            x_min,
            x_max,
            by_min,
            by_max,
            y_label_w,
        );
    }

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(layout[2]);
    render_histogram(app, snapshot, frame, bottom[0], x_min, x_max);
    render_sidebar(app, snapshot, frame, bottom[1]);
}

struct Crosshair {
    data_x: f64,
    base_fee_y: f64,
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
    data_x: f64,
    scaled_fee_y: f64,
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

    let tx_col = data_to_col(data_x, x_min, x_max, tx_inner);
    let bf_col = data_to_col(data_x, x_min, x_max, bf_inner);
    let bf_row = data_to_row(scaled_fee_y, by_min, by_max, bf_inner);

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
    // Build cell → sorted filter indices mapping.
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
    // Pre-sort and dedup each cell's indices once.
    for indices in cell_filters.values_mut() {
        indices.sort();
        indices.dedup();
    }

    // Pre-compute color for each cell.
    let cell_colors: HashMap<i64, Color> = cell_filters
        .iter()
        .map(|(cx, indices)| (*cx, cell_color(indices, grouped_filter_series)))
        .collect();

    // Assign each data point its cell's pre-computed color.
    let mut color_buckets: HashMap<Color, Vec<(f64, f64)>> = HashMap::new();
    for (_label, _color_idx, series) in grouped_filter_series {
        for &(x, y) in series {
            if y > 0.0 {
                let cx = quantize(x, x_min, cell_w);
                if let Some(&color) = cell_colors.get(&cx)
                    && let Some(&base_fee) = base_fee_by_x.get(&(x as u64))
                {
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
