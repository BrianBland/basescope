use std::collections::{HashMap, HashSet};

use ratatui::layout::Rect;
use ratatui::prelude::Frame;
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::{Bar, BarChart, BarGroup, Block, Borders, Paragraph};

use crate::domain::{AnalysisSnapshot, FilterId};
use crate::tui::{App, HistogramMode};

use super::colors::{blend_colors, filter_color, filter_rgb};
use super::{
    filter_visible, format_fee_label, group_series_avg, pick_fee_unit, truncate_to, FilterSeries,
};

type HistSlices<'a> = Vec<(&'a str, usize, &'a [(f64, f64)])>;

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

    match app.hist_mode {
        HistogramMode::AllBlocks => {
            render_histogram_all_blocks(app, snapshot, frame, area, inner_width, gap, x_min, x_max);
        }
        HistogramMode::FilterMatches => {
            render_histogram_filter_matches(snapshot, frame, area, inner_width, gap, x_min, x_max);
        }
        HistogramMode::Stacked => {
            render_histogram_stacked(app, snapshot, frame, area, inner_width, gap, x_min, x_max);
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
    snapshot: &AnalysisSnapshot,
    frame: &mut Frame,
    area: Rect,
    inner_width: usize,
    gap: usize,
    x_min: f64,
    x_max: f64,
) {
    let base_fee_lookup: HashMap<u64, f64> = snapshot
        .base_fee_series
        .iter()
        .map(|(block, fee)| (*block as u64, *fee))
        .collect();

    let owned_hists: FilterSeries = if snapshot.show_aggregate {
        let visible = filter_visible(&snapshot.aggregate_series, x_min, x_max);
        let hist = build_fee_histogram(visible, &base_fee_lookup);
        vec![("".to_string(), 0, hist)]
    } else {
        snapshot
            .filters
            .iter()
            .filter(|f| f.enabled)
            .filter_map(|f| {
                snapshot.filter_series.get(&f.id).map(|series| {
                    let visible = filter_visible(series, x_min, x_max);
                    let hist = build_fee_histogram(visible, &base_fee_lookup);
                    (f.label.clone(), f.color_index, hist)
                })
            })
            .collect()
    };

    let raw_hists: HistSlices<'_> = owned_hists
        .iter()
        .map(|(label, color_idx, hist)| (label.as_str(), *color_idx, hist.as_slice()))
        .collect();

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

    let visible_base_fee = filter_visible(&snapshot.base_fee_series, x_min, x_max);
    let grouped;
    let fee_source: &[(f64, f64)] = if g > 1 {
        grouped = group_series_avg(visible_base_fee, g);
        &grouped
    } else {
        visible_base_fee
    };
    let hist_data: Vec<(f64, f64)> = {
        let mut hist: Vec<(f64, f64)> = Vec::new();
        for &(_block, fee) in fee_source {
            let bucket = (fee * 1000.0).floor() / 1000.0;
            if let Some((_, count)) = hist.iter_mut().find(|(b, _)| (*b - bucket).abs() < 1e-9) {
                *count += 1.0;
            } else {
                hist.push((bucket, 1.0));
            }
        }
        hist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        hist
    };

    let merged = rebucket(&hist_data, max_buckets);
    let hist_max = merged.iter().map(|(_, hi, _)| *hi).fold(0.0_f64, f64::max);
    let unit = pick_fee_unit(hist_max);

    let enabled_filters: Vec<_> = snapshot.filters.iter().filter(|f| f.enabled).collect();

    let base_fee_lookup: HashMap<u64, f64> = visible_base_fee
        .iter()
        .map(|(block, fee)| (*block as u64, *fee))
        .collect();
    let visible_filter_hists: HashMap<FilterId, Vec<(f64, f64)>> = enabled_filters
        .iter()
        .filter_map(|f| {
            snapshot.filter_series.get(&f.id).map(|series| {
                let visible = filter_visible(series, x_min, x_max);
                (f.id, build_fee_histogram(visible, &base_fee_lookup))
            })
        })
        .collect();

    let styled_bars: Vec<(String, u64, Style)> = merged
        .iter()
        .map(|(lo, hi, count)| {
            let label = format_fee_label(*lo, *hi, unit);
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
    let title = format!("base fee histogram — all blocks{gran_suffix} [h: switch]");

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
) {
    let g = app.effective_granularity();
    let visible_base_fee = filter_visible(&snapshot.base_fee_series, x_min, x_max);

    // Build per-filter sets of visible matching block numbers.
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

    // Process chunks (respecting granularity): compute fee bucket + filter match bitmask.
    let entries: Vec<(f64, u16)> = visible_base_fee
        .chunks(g.max(1))
        .map(|chunk| {
            let avg_fee = chunk.iter().map(|(_, f)| f).sum::<f64>() / chunk.len() as f64;
            let fee_bucket = (avg_fee * 1000.0).floor() / 1000.0;
            let mask: u16 = filter_match_sets
                .iter()
                .enumerate()
                .filter(|(_, set)| chunk.iter().any(|(b, _)| set.contains(&(*b as u64))))
                .fold(0u16, |acc, (i, _)| acc | (1 << i));
            (fee_bucket, mask)
        })
        .collect();

    // Build raw histogram for rebucketing.
    let mut raw_hist: Vec<(f64, f64)> = Vec::new();
    for &(bucket, _) in &entries {
        if let Some((_, count)) = raw_hist.iter_mut().find(|(b, _)| (*b - bucket).abs() < 1e-9) {
            *count += 1.0;
        } else {
            raw_hist.push((bucket, 1.0));
        }
    }
    raw_hist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let bar_w_init = 6u16;
    let max_buckets = if (bar_w_init as usize + gap) > 0 {
        inner_width / (bar_w_init as usize + gap)
    } else {
        20
    }
    .max(2);
    let merged = rebucket(&raw_hist, max_buckets);

    let gran_suffix = app.granularity_label();
    let title = format!("base fee histogram — stacked{gran_suffix} [h: switch]");

    if merged.is_empty() {
        let block_widget = Block::default().title(title).borders(Borders::ALL);
        frame.render_widget(block_widget, area);
        return;
    }

    // For each merged bucket, group entries by match-mask to form stack segments.
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

            // Unmatched (mask 0) at the bottom.
            if let Some(&count) = mask_counts.get(&0)
                && count > 0.0
            {
                segments.push(Segment {
                    count,
                    color: Color::DarkGray,
                });
            }

            // Matched segments sorted by bitmask for stable ordering.
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

    // Compute dynamic bar width from labels.
    let hist_max_fee = stacked_bars.iter().map(|b| b.hi).fold(0.0_f64, f64::max);
    let unit = pick_fee_unit(hist_max_fee);
    let max_label_len = stacked_bars
        .iter()
        .map(|b| format_fee_label(b.lo, b.hi, unit).len())
        .max()
        .unwrap_or(3);
    let bar_w = max_label_len.max(3) as u16;

    // Render.
    let block_widget = Block::default().title(title).borders(Borders::ALL);
    let inner = block_widget.inner(area);
    frame.render_widget(block_widget, area);

    let chart_height = inner.height.saturating_sub(1); // 1 row for labels
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
            // Give last segment any leftover rows to avoid rounding gaps.
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

        // Value above bar.
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

        // Fee label.
        let label = format_fee_label(bar.lo, bar.hi, unit);
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

/// Build a base-fee histogram from a block series by looking up each block's fee.
fn build_fee_histogram(
    block_series: &[(f64, f64)],
    base_fee_lookup: &HashMap<u64, f64>,
) -> Vec<(f64, f64)> {
    let mut hist: Vec<(f64, f64)> = Vec::new();
    for &(block, tx_count) in block_series {
        if tx_count <= 0.0 {
            continue;
        }
        if let Some(&fee) = base_fee_lookup.get(&(block as u64)) {
            let bucket = (fee * 1000.0).floor() / 1000.0;
            if let Some((_, count)) = hist.iter_mut().find(|(b, _)| (*b - bucket).abs() < 1e-9) {
                *count += tx_count;
            } else {
                hist.push((bucket, tx_count));
            }
        }
    }
    hist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    hist
}
