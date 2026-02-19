use std::collections::{BTreeMap, HashMap, HashSet};

use ratatui::layout::Rect;
use ratatui::prelude::Frame;
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::{Bar, BarChart, BarGroup, Block, Borders, Paragraph};

use crate::domain::{AnalysisSnapshot, ChartMode, FilterId};
use crate::tui::{App, HistogramMode};

use super::colors::{blend_colors, filter_color, filter_rgb};
use super::{
    filter_visible, format_fee_label, group_series_avg, group_series_sum, pick_fee_unit,
    truncate_to, FilterSeries,
};

type HistSlices<'a> = Vec<(&'a str, usize, &'a [(f64, f64)])>;

fn bucket_precision(mode: ChartMode) -> f64 {
    match mode {
        ChartMode::TxCount => 1000.0,
        ChartMode::GasUsed | ChartMode::DaSize => 1.0,
    }
}

fn quantize_bucket(value: f64, mode: ChartMode) -> i64 {
    let p = bucket_precision(mode);
    (value * p).floor() as i64
}

fn key_to_bucket(key: i64, mode: ChartMode) -> f64 {
    let p = bucket_precision(mode);
    key as f64 / p
}

fn format_si(value: f64) -> String {
    if value >= 1_000_000_000.0 {
        format!("{:.1}B", value / 1_000_000_000.0)
    } else if value >= 1_000_000.0 {
        format!("{:.1}M", value / 1_000_000.0)
    } else if value >= 1_000.0 {
        format!("{:.0}K", value / 1_000.0)
    } else {
        format!("{:.0}", value)
    }
}

fn format_si_bytes(value: f64) -> String {
    if value >= 1_000_000.0 {
        format!("{:.1}MB", value / 1_000_000.0)
    } else if value >= 1_000.0 {
        format!("{:.0}KB", value / 1_000.0)
    } else {
        format!("{:.0}B", value)
    }
}

fn format_bucket_label(lo: f64, hi: f64, mode: ChartMode) -> String {
    match mode {
        ChartMode::TxCount => {
            let unit = pick_fee_unit(hi);
            format_fee_label(lo, hi, unit)
        }
        ChartMode::GasUsed => {
            if (hi - lo).abs() < 1.0 {
                format_si(lo)
            } else {
                format!("{}-{}", format_si(lo), format_si(hi))
            }
        }
        ChartMode::DaSize => {
            if (hi - lo).abs() < 1.0 {
                format_si_bytes(lo)
            } else {
                format!("{}-{}", format_si_bytes(lo), format_si_bytes(hi))
            }
        }
    }
}

fn estimate_bar_label_width(hi: f64, mode: ChartMode) -> usize {
    let sample = format_bucket_label(hi * 0.9, hi, mode);
    sample.len().max(3)
}

fn accumulate_histogram(source: &[(f64, f64)], mode: ChartMode) -> Vec<(f64, f64)> {
    let mut tree: BTreeMap<i64, f64> = BTreeMap::new();
    for &(_block, value) in source {
        let key = quantize_bucket(value, mode);
        *tree.entry(key).or_default() += 1.0;
    }
    tree.into_iter()
        .map(|(k, c)| (key_to_bucket(k, mode), c))
        .collect()
}

fn accumulate_histogram_weighted(
    block_series: &[(f64, f64)],
    value_lookup: &HashMap<u64, f64>,
    mode: ChartMode,
) -> Vec<(f64, f64)> {
    let mut tree: BTreeMap<i64, f64> = BTreeMap::new();
    for &(block, weight) in block_series {
        if weight <= 0.0 {
            continue;
        }
        if let Some(&value) = value_lookup.get(&(block as u64)) {
            let key = quantize_bucket(value, mode);
            *tree.entry(key).or_default() += weight;
        }
    }
    tree.into_iter()
        .map(|(k, c)| (key_to_bucket(k, mode), c))
        .collect()
}

fn build_value_lookup(snapshot: &AnalysisSnapshot, mode: ChartMode) -> HashMap<u64, f64> {
    let series = snapshot.block_series_for(mode);
    series
        .iter()
        .map(|(block, val)| (*block as u64, *val))
        .collect()
}

pub(super) fn render_histogram(
    app: &App,
    snapshot: &AnalysisSnapshot,
    frame: &mut Frame,
    area: Rect,
    x_min: f64,
    x_max: f64,
) {
    let inner_width = area.width.saturating_sub(2) as usize;
    let gap = 1usize;
    let chart_mode = app.view.chart_mode;

    match app.view.hist_mode {
        HistogramMode::AllBlocks => {
            render_histogram_all_blocks(
                app, snapshot, frame, area, inner_width, gap, x_min, x_max, chart_mode,
            );
        }
        HistogramMode::FilterMatches => {
            render_histogram_filter_matches(
                app, snapshot, frame, area, inner_width, gap, x_min, x_max, chart_mode,
            );
        }
        HistogramMode::Stacked => {
            render_histogram_stacked(
                app, snapshot, frame, area, inner_width, gap, x_min, x_max, chart_mode,
            );
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

#[allow(clippy::too_many_arguments)]
fn render_histogram_filter_matches(
    app: &App,
    snapshot: &AnalysisSnapshot,
    frame: &mut Frame,
    area: Rect,
    inner_width: usize,
    gap: usize,
    x_min: f64,
    x_max: f64,
    chart_mode: ChartMode,
) {
    let value_lookup = build_value_lookup(snapshot, chart_mode);

    let owned_hists: FilterSeries = if snapshot.show_aggregate {
        let agg = &snapshot.aggregate_series;
        let visible = filter_visible(agg, x_min, x_max);
        let hist = accumulate_histogram_weighted(visible, &value_lookup, chart_mode);
        vec![("".to_string(), 0, hist)]
    } else {
        snapshot
            .filters
            .iter()
            .filter(|f| f.enabled)
            .filter_map(|f| {
                snapshot.filter_series.get(&f.id).map(|series| {
                    let visible = filter_visible(series, x_min, x_max);
                    let hist = accumulate_histogram_weighted(visible, &value_lookup, chart_mode);
                    (f.label.clone(), f.color_index, hist)
                })
            })
            .collect()
    };

    let raw_hists: HistSlices<'_> = owned_hists
        .iter()
        .map(|(label, color_idx, hist)| (label.as_str(), *color_idx, hist.as_slice()))
        .collect();

    let hist_max_val = raw_hists
        .iter()
        .flat_map(|(_, _, h)| h.iter().map(|(v, _)| *v))
        .fold(0.0_f64, f64::max);
    let label_w = estimate_bar_label_width(hist_max_val, chart_mode);
    let bar_w = label_w.max(3) as u16;
    let max_buckets = if (bar_w as usize + gap) > 0 {
        inner_width / (bar_w as usize + gap)
    } else {
        20
    }
    .max(2);

    let mut all_merged: Vec<(f64, f64, f64, &str, usize)> = Vec::new();
    for (prefix, color_idx, hist) in &raw_hists {
        let merged = smart_rebucket(hist, max_buckets, chart_mode);
        for (lo, hi, count) in merged {
            all_merged.push((lo, hi, count, prefix, *color_idx));
        }
    }

    let mut entries: Vec<(f64, &str, String, String, u64, usize)> = Vec::new();
    for (lo, hi, count, prefix, color_idx) in &all_merged {
        let label = format_bucket_label(*lo, *hi, chart_mode);
        entries.push((
            *lo,
            prefix,
            label,
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

    let max_label_len = entries.iter().map(|(_, _, l, _, _, _)| l.len()).max().unwrap_or(3);
    let dynamic_bar_w = max_label_len.max(3) as u16;

    let bar_group: Vec<Bar> = entries
        .iter()
        .map(|(_, _, label_str, prefix, value, color_idx)| {
            let color = if snapshot.show_aggregate {
                Color::White
            } else {
                filter_color(*color_idx)
            };
            let w = dynamic_bar_w as usize;
            let label = if prefix.is_empty() {
                truncate_to(label_str, w)
            } else {
                let label_len = label_str.len();
                if label_len >= w {
                    truncate_to(label_str, w)
                } else {
                    let remaining = w - label_len;
                    if remaining >= 2 {
                        let pfx = truncate_to(prefix, remaining);
                        format!("{pfx}{label_str}")
                    } else {
                        truncate_to(label_str, w)
                    }
                }
            };
            Bar::default()
                .value(*value)
                .label(Line::from(label))
                .style(Style::default().fg(color))
        })
        .collect();

    let _ = app;
    let title = format!(
        "{} histogram — filter matches [h: switch]",
        hist_title_prefix(chart_mode)
    );

    let chart = BarChart::default()
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL),
        )
        .bar_width(dynamic_bar_w)
        .bar_gap(gap as u16)
        .data(BarGroup::default().bars(&bar_group));
    frame.render_widget(chart, area);
}

fn bucket_overlaps(bucket_lo: f64, bucket_hi: f64, raw: &[(f64, f64)]) -> bool {
    raw.iter()
        .any(|(b, count)| *count > 0.0 && *b >= bucket_lo - 1e-9 && *b <= bucket_hi + 1e-9)
}

#[allow(clippy::too_many_arguments)]
fn render_histogram_all_blocks(
    app: &App,
    snapshot: &AnalysisSnapshot,
    frame: &mut Frame,
    area: Rect,
    inner_width: usize,
    gap: usize,
    x_min: f64,
    x_max: f64,
    chart_mode: ChartMode,
) {
    let g = app.effective_granularity();

    let block_series = snapshot.block_series_for(chart_mode);
    let visible = filter_visible(block_series, x_min, x_max);
    let grouped;
    let source: &[(f64, f64)] = if g > 1 {
        grouped = match chart_mode {
            ChartMode::TxCount => group_series_avg(visible, g),
            ChartMode::GasUsed | ChartMode::DaSize => group_series_sum(visible, g),
        };
        &grouped
    } else {
        visible
    };

    let hist_data = accumulate_histogram(source, chart_mode);

    let hist_max_val = hist_data.iter().map(|(v, _)| *v).fold(0.0_f64, f64::max);
    let label_w = estimate_bar_label_width(hist_max_val, chart_mode);
    let bar_w_init = label_w.max(3) as u16;
    let max_buckets = if (bar_w_init as usize + gap) > 0 {
        inner_width / (bar_w_init as usize + gap)
    } else {
        20
    }
    .max(2);

    let merged = smart_rebucket(&hist_data, max_buckets, chart_mode);

    let enabled_filters: Vec<_> = snapshot.filters.iter().filter(|f| f.enabled).collect();

    let value_lookup = build_value_lookup(snapshot, chart_mode);
    let visible_filter_hists: HashMap<FilterId, Vec<(f64, f64)>> = enabled_filters
        .iter()
        .filter_map(|f| {
            snapshot.filter_series.get(&f.id).map(|series| {
                let vis = filter_visible(series, x_min, x_max);
                (
                    f.id,
                    accumulate_histogram_weighted(vis, &value_lookup, chart_mode),
                )
            })
        })
        .collect();

    let styled_bars: Vec<(String, u64, Style)> = merged
        .iter()
        .map(|(lo, hi, count)| {
            let label = format_bucket_label(*lo, *hi, chart_mode);
            let matching_colors: Vec<(u8, u8, u8)> = enabled_filters
                .iter()
                .filter(|f| {
                    visible_filter_hists
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
    let title = format!(
        "{} histogram — all blocks{gran_suffix} [h: switch]",
        hist_title_prefix(chart_mode)
    );

    let chart = BarChart::default()
        .block(Block::default().title(title).borders(Borders::ALL))
        .bar_width(dynamic_bar_w)
        .bar_gap(gap as u16)
        .data(BarGroup::default().bars(&bar_group));
    frame.render_widget(chart, area);
}

#[allow(clippy::too_many_arguments)]
fn render_histogram_stacked(
    app: &App,
    snapshot: &AnalysisSnapshot,
    frame: &mut Frame,
    area: Rect,
    inner_width: usize,
    gap: usize,
    x_min: f64,
    x_max: f64,
    chart_mode: ChartMode,
) {
    let g = app.effective_granularity();
    let block_series = snapshot.block_series_for(chart_mode);
    let visible = filter_visible(block_series, x_min, x_max);

    let enabled_filters: Vec<_> = snapshot.filters.iter().filter(|f| f.enabled).collect();
    let filter_match_sets: Vec<HashSet<u64>> = enabled_filters
        .iter()
        .map(|f| {
            snapshot
                .filter_series
                .get(&f.id)
                .map(|series| {
                    filter_visible(series, x_min, x_max)
                        .iter()
                        .filter(|(_, count)| *count > 0.0)
                        .map(|(block, _)| *block as u64)
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect();

    let entries: Vec<(f64, u16)> = visible
        .chunks(g.max(1))
        .map(|chunk| {
            let agg_value = match chart_mode {
                ChartMode::TxCount => {
                    let avg = chunk.iter().map(|(_, f)| f).sum::<f64>() / chunk.len() as f64;
                    (avg * bucket_precision(chart_mode)).floor() / bucket_precision(chart_mode)
                }
                ChartMode::GasUsed | ChartMode::DaSize => {
                    let sum = chunk.iter().map(|(_, f)| f).sum::<f64>();
                    (sum * bucket_precision(chart_mode)).floor() / bucket_precision(chart_mode)
                }
            };
            let mask: u16 = filter_match_sets
                .iter()
                .enumerate()
                .filter(|(_, set)| chunk.iter().any(|(b, _)| set.contains(&(*b as u64))))
                .fold(0u16, |acc, (i, _)| acc | (1 << i));
            (agg_value, mask)
        })
        .collect();

    let mut raw_tree: BTreeMap<i64, f64> = BTreeMap::new();
    for &(bucket, _) in &entries {
        let key = quantize_bucket(bucket, chart_mode);
        *raw_tree.entry(key).or_default() += 1.0;
    }
    let raw_hist: Vec<(f64, f64)> = raw_tree
        .into_iter()
        .map(|(k, c)| (key_to_bucket(k, chart_mode), c))
        .collect();

    let hist_max_val = raw_hist.iter().map(|(v, _)| *v).fold(0.0_f64, f64::max);
    let label_w = estimate_bar_label_width(hist_max_val, chart_mode);
    let bar_w_init = label_w.max(3) as u16;
    let max_buckets = if (bar_w_init as usize + gap) > 0 {
        inner_width / (bar_w_init as usize + gap)
    } else {
        20
    }
    .max(2);
    let merged = smart_rebucket(&raw_hist, max_buckets, chart_mode);

    let gran_suffix = app.granularity_label();
    let title = format!(
        "{} histogram — stacked{gran_suffix} [h: switch]",
        hist_title_prefix(chart_mode)
    );

    if merged.is_empty() {
        let block_widget = Block::default().title(title).borders(Borders::ALL);
        frame.render_widget(block_widget, area);
        return;
    }

    struct Segment {
        count: f64,
        color: Color,
    }
    struct StackedBar {
        lo: f64,
        hi: f64,
        total: f64,
        segments: Vec<Segment>,
    }

    let stacked_bars: Vec<StackedBar> = merged
        .iter()
        .map(|(lo, hi, total)| {
            let mut mask_counts: HashMap<u16, f64> = HashMap::new();
            for &(bucket, mask) in &entries {
                if bucket >= *lo - 1e-9 && bucket <= *hi + 1e-9 {
                    *mask_counts.entry(mask).or_default() += 1.0;
                }
            }

            let mut segments: Vec<Segment> = Vec::new();

            if let Some(&count) = mask_counts.get(&0)
                && count > 0.0
            {
                segments.push(Segment {
                    count,
                    color: Color::DarkGray,
                });
            }

            let mut matched: Vec<(u16, f64)> = mask_counts
                .into_iter()
                .filter(|(mask, _)| *mask != 0)
                .collect();
            matched.sort_by_key(|(mask, _)| *mask);

            for (mask, count) in matched {
                let indices: Vec<usize> = (0..enabled_filters.len())
                    .filter(|i| mask & (1 << *i) != 0)
                    .collect();
                let color = if indices.len() == 1 {
                    filter_color(enabled_filters[indices[0]].color_index)
                } else {
                    let rgbs: Vec<(u8, u8, u8)> = indices
                        .iter()
                        .map(|&i| filter_rgb(enabled_filters[i].color_index))
                        .collect();
                    blend_colors(&rgbs)
                };
                segments.push(Segment { count, color });
            }

            StackedBar {
                lo: *lo,
                hi: *hi,
                total: *total,
                segments,
            }
        })
        .collect();

    let max_label_len = stacked_bars
        .iter()
        .map(|b| format_bucket_label(b.lo, b.hi, chart_mode).len())
        .max()
        .unwrap_or(3);
    let bar_w = max_label_len.max(3) as u16;

    let block_widget = Block::default().title(title).borders(Borders::ALL);
    let inner = block_widget.inner(area);
    frame.render_widget(block_widget, area);

    let chart_height = inner.height.saturating_sub(1);
    if chart_height == 0 || inner.width < 2 {
        return;
    }

    let max_total = stacked_bars.iter().map(|b| b.total).fold(0.0_f64, f64::max);
    if max_total <= 0.0 {
        return;
    }

    let buf = frame.buffer_mut();
    let label_row = inner.y + inner.height - 1;

    for (i, bar) in stacked_bars.iter().enumerate() {
        let bar_x = inner.x + (i as u16) * (bar_w + gap as u16);
        if bar_x + bar_w > inner.x + inner.width {
            break;
        }

        let bar_height = ((bar.total / max_total) * chart_height as f64)
            .round()
            .min(chart_height as f64) as u16;
        let bar_bottom = inner.y + chart_height;
        let mut rows_used = 0u16;

        for (seg_idx, seg) in bar.segments.iter().enumerate() {
            let seg_rows = if bar.total > 0.0 {
                ((seg.count / bar.total) * bar_height as f64).round() as u16
            } else {
                0
            };
            let seg_rows = if seg_idx == bar.segments.len() - 1 {
                bar_height.saturating_sub(rows_used)
            } else {
                seg_rows.min(bar_height.saturating_sub(rows_used))
            };
            if seg_rows == 0 {
                continue;
            }

            let seg_bottom = bar_bottom.saturating_sub(rows_used);
            let seg_top = seg_bottom.saturating_sub(seg_rows);
            for row in seg_top..seg_bottom {
                for col in bar_x..bar_x + bar_w {
                    if col < buf.area.x + buf.area.width && row < buf.area.y + buf.area.height {
                        let cell = &mut buf[(col, row)];
                        cell.set_symbol("█");
                        cell.set_fg(seg.color);
                    }
                }
            }
            rows_used += seg_rows;
        }

        let bar_top = bar_bottom.saturating_sub(bar_height);
        if bar_top > inner.y {
            let value_str = format!("{}", bar.total as u64);
            let value_row = bar_top - 1;
            if value_row >= inner.y && value_str.len() <= bar_w as usize {
                for (j, ch) in value_str.chars().enumerate() {
                    let col = bar_x + j as u16;
                    if col < buf.area.x + buf.area.width
                        && value_row < buf.area.y + buf.area.height
                    {
                        buf[(col, value_row)].set_symbol(&ch.to_string());
                        buf[(col, value_row)].set_fg(Color::White);
                    }
                }
            }
        }

        let label = format_bucket_label(bar.lo, bar.hi, chart_mode);
        let label = truncate_to(&label, bar_w as usize);
        if label_row < buf.area.y + buf.area.height {
            for (j, ch) in label.chars().enumerate() {
                let col = bar_x + j as u16;
                if col < buf.area.x + buf.area.width {
                    buf[(col, label_row)].set_symbol(&ch.to_string());
                    buf[(col, label_row)].set_fg(Color::Gray);
                }
            }
        }
    }
}

fn hist_title_prefix(mode: ChartMode) -> &'static str {
    match mode {
        ChartMode::TxCount => "base fee",
        ChartMode::GasUsed => "gas",
        ChartMode::DaSize => "DA",
    }
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

fn rebucket_uniform(raw: &[(f64, f64)], max_buckets: usize) -> Vec<(f64, f64, f64)> {
    if raw.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<(f64, f64)> = raw.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let lo = sorted.first().unwrap().0;
    let hi = sorted.last().unwrap().0;

    if (hi - lo).abs() < f64::EPSILON || max_buckets <= 1 {
        let total: f64 = sorted.iter().map(|(_, c)| *c).sum();
        return vec![(lo, hi, total)];
    }

    let target = max_buckets.max(1);
    let width = (hi - lo) / target as f64;

    let mut merged: Vec<(f64, f64, f64)> = Vec::with_capacity(target);
    for i in 0..target {
        let b_lo = lo + i as f64 * width;
        let b_hi = if i == target - 1 {
            hi
        } else {
            lo + (i + 1) as f64 * width
        };
        merged.push((b_lo, b_hi, 0.0));
    }

    for &(val, count) in &sorted {
        let idx = if width > 0.0 {
            ((val - lo) / width).floor() as usize
        } else {
            0
        };
        let idx = idx.min(target - 1);
        merged[idx].2 += count;
    }

    merged.retain(|&(_, _, c)| c > 0.0);
    merged
}

fn smart_rebucket(
    raw: &[(f64, f64)],
    max_buckets: usize,
    mode: ChartMode,
) -> Vec<(f64, f64, f64)> {
    match mode {
        ChartMode::TxCount => rebucket(raw, max_buckets),
        ChartMode::GasUsed | ChartMode::DaSize => rebucket_uniform(raw, max_buckets),
    }
}
