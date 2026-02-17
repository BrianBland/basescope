mod charts;
mod colors;
mod histogram;
mod input;
mod panels;
mod sidebar;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Frame;

use crate::tui::{App, AppMode, BottomPanel, ScaleTransform};

pub(super) type FilterEntry = (String, usize, Vec<(f64, f64)>);
pub(super) type FilterSeries = Vec<FilterEntry>;

const BOTTOM_PANEL_HEIGHT: u16 = 8;

pub fn render(app: &App, frame: &mut Frame) {
    let outer = frame.area();

    let all_done =
        !app.chunk_states.is_empty() && app.chunk_states.iter().all(sidebar::is_complete);

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
            panels::render_log_panel(app, frame, layout[1]);
        }
        BottomPanel::Rpc => {
            let layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(BOTTOM_PANEL_HEIGHT)])
                .split(outer);
            render_main(app, frame, layout[0]);
            panels::render_rpc_panel(app, frame, layout[1]);
        }
        BottomPanel::Hidden => {
            render_main(app, frame, outer);
        }
    }

    if app.show_help {
        panels::render_help_panel(app, frame, outer);
    }

    if app.granularity_input.is_some() {
        panels::render_granularity_input(app, frame, outer);
    }
}

fn render_main(app: &App, frame: &mut Frame, area: Rect) {
    match app.mode {
        AppMode::RangeInput => input::render_range_input(app, frame, area),
        AppMode::FilterInput => input::render_filter_input(app, frame, area),
        AppMode::Fetching | AppMode::Results => charts::render_results(app, frame, area),
    }
}

#[derive(Clone, Copy)]
pub(super) enum FeeUnit {
    Gwei,
    Mwei,
}

pub(super) fn pick_fee_unit(max_gwei: f64) -> FeeUnit {
    if max_gwei.abs() < 1.0 {
        FeeUnit::Mwei
    } else {
        FeeUnit::Gwei
    }
}

pub(super) fn format_fee_with_unit(gwei: f64, unit: FeeUnit) -> String {
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

pub(super) fn format_fee_value(gwei: f64) -> String {
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

pub(super) fn truncate_to(s: &str, max: usize) -> String {
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

pub(super) fn format_fee_label(lo: f64, hi: f64, unit: FeeUnit) -> String {
    if (hi - lo).abs() < 0.0005 {
        format_fee_with_unit(lo, unit)
    } else {
        let lo_s = format_fee_with_unit(lo, unit);
        let hi_s = format_fee_with_unit(hi, unit);
        format!("{lo_s}-{hi_s}")
    }
}

pub(super) fn filter_visible(series: &[(f64, f64)], x_min: f64, x_max: f64) -> &[(f64, f64)] {
    let start = series.partition_point(|(x, _)| *x < x_min);
    let end = series.partition_point(|(x, _)| *x <= x_max);
    &series[start..end]
}

pub(super) fn series_x_bounds(series: &[(f64, f64)]) -> (f64, f64) {
    if series.is_empty() {
        return (0.0, 1.0);
    }
    let min = series
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::INFINITY, f64::min);
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

pub(super) fn series_y_bounds(datasets: &[&[(f64, f64)]]) -> (f64, f64) {
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

pub(super) fn group_series_sum(series: &[(f64, f64)], granularity: usize) -> Vec<(f64, f64)> {
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

pub(super) fn group_series_avg(series: &[(f64, f64)], granularity: usize) -> Vec<(f64, f64)> {
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

pub(super) fn x_axis_labels(x_min: f64, x_max: f64, available_chars: usize) -> Vec<String> {
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

pub(super) fn y_labels_int(y_min: f64, y_max: f64) -> Vec<String> {
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

pub(super) fn scaled_y_labels_gwei(
    scaled_min: f64,
    scaled_max: f64,
    scale: &ScaleTransform,
) -> Vec<String> {
    let mid = (scaled_min + scaled_max) / 2.0;
    let q1 = (scaled_min + mid) / 2.0;
    let q3 = (mid + scaled_max) / 2.0;
    let positions = [scaled_min, q1, mid, q3, scaled_max];
    let originals: Vec<f64> = positions.iter().map(|v| scale.invert(*v)).collect();
    let max_original = originals.iter().fold(0.0_f64, |a, b| a.max(*b));
    let unit = pick_fee_unit(max_original);
    originals
        .iter()
        .map(|v| format_fee_with_unit(*v, unit))
        .collect()
}

pub(super) fn nearest_y(series: &[(f64, f64)], target_x: f64) -> f64 {
    if series.is_empty() {
        return 0.0;
    }
    let idx = series.partition_point(|(x, _)| *x < target_x);
    let candidates = [idx.saturating_sub(1), idx.min(series.len() - 1)];
    candidates
        .iter()
        .map(|&i| &series[i])
        .min_by(|a, b| {
            (a.0 - target_x)
                .abs()
                .partial_cmp(&(b.0 - target_x).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|pt| pt.1)
        .unwrap_or(0.0)
}

pub(super) fn by_labels_width(by_min: f64, by_max: f64) -> u16 {
    let sample = format!("{:.3}", by_max.max(by_min.abs()));
    (sample.len() + 2) as u16
}

pub(crate) fn chart_inner(chart_rect: Rect, y_label_w: u16) -> Rect {
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
