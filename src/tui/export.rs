#[cfg(feature = "export")]
mod imp {
use std::collections::{BTreeMap, HashMap};


use anyhow::{Context, Result};

use plotters::coord::Shift;

use plotters::prelude::*;


use crate::domain::{AnalysisSnapshot, ChartMode, FilterId};

use crate::tui::{HistogramMode, ScaleMode};


type FilterEntry = (String, usize, Vec<(f64, f64)>);

type FilterSeries = Vec<FilterEntry>;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Png,
    Svg,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportChart {
    Top,
    Middle,
    Histogram,
    All,
}


#[allow(clippy::too_many_arguments)]
pub fn export_charts(
    snapshot: &AnalysisSnapshot,
    chart_mode: ChartMode,
    charts: ExportChart,
    format: ExportFormat,
    view_start: Option<f64>,
    view_end: Option<f64>,
    granularity: usize,
    scale_mode: ScaleMode,
    hist_mode: HistogramMode,
    ema_span: Option<usize>,
) -> Result<String> {
    let block_series = snapshot.block_series_for(chart_mode);
    let (full_x_min, full_x_max) = series_x_bounds(block_series);
    let x_min = view_start.unwrap_or(full_x_min);
    let x_max = view_end.unwrap_or(full_x_max);

    let filename = export_filename(format)?;
    let size = match charts {
        ExportChart::All => (1920, 2160),
        _ => (1920, 1080),
    };

    match format {
        ExportFormat::Png => {
            let backend = BitMapBackend::new(&filename, size);
            render_root(backend, snapshot, chart_mode, charts, x_min, x_max, granularity, scale_mode, hist_mode, ema_span)?;
        }
        ExportFormat::Svg => {
            let backend = SVGBackend::new(&filename, size);
            render_root(backend, snapshot, chart_mode, charts, x_min, x_max, granularity, scale_mode, hist_mode, ema_span)?;
        }
    }

    Ok(filename)
}


#[allow(clippy::too_many_arguments)]
fn render_root<DB: DrawingBackend>(
    backend: DB,
    snapshot: &AnalysisSnapshot,
    chart_mode: ChartMode,
    charts: ExportChart,
    x_min: f64,
    x_max: f64,
    granularity: usize,
    scale_mode: ScaleMode,
    hist_mode: HistogramMode,
    ema_span: Option<usize>,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let root = backend.into_drawing_area();
    root.fill(&RGBColor(20, 20, 30))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    match charts {
        ExportChart::Top => {
            render_top_chart(&root, snapshot, chart_mode, x_min, x_max, granularity, ema_span)?;
        }
        ExportChart::Middle => {
            render_mid_chart(
                &root,
                snapshot,
                chart_mode,
                x_min,
                x_max,
                granularity,
                scale_mode,
            )?;
        }
        ExportChart::Histogram => {
            render_hist_chart(&root, snapshot, chart_mode, x_min, x_max, granularity, hist_mode)?;
        }
        ExportChart::All => {
            let areas = root.split_evenly((3, 1));
            render_top_chart(&areas[0], snapshot, chart_mode, x_min, x_max, granularity, ema_span)?;
            render_mid_chart(
                &areas[1],
                snapshot,
                chart_mode,
                x_min,
                x_max,
                granularity,
                scale_mode,
            )?;
            render_hist_chart(
                &areas[2],
                snapshot,
                chart_mode,
                x_min,
                x_max,
                granularity,
                hist_mode,
            )?;
        }
    }

    root.present().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(())
}


fn render_top_chart<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    snapshot: &AnalysisSnapshot,
    chart_mode: ChartMode,
    x_min: f64,
    x_max: f64,
    granularity: usize,
    ema_span: Option<usize>,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let title = format!(
        "{} per block{}",
        chart_mode.top_title(),
        granularity_suffix(granularity)
    );

    let grouped_filters = build_grouped_filters(snapshot, chart_mode, x_min, x_max, granularity);
    let series_refs: Vec<&[(f64, f64)]> = grouped_filters.iter().map(|(_, _, s)| s.as_slice()).collect();
    let (y_min, y_max) = series_y_bounds(&series_refs);

    let mut chart = ChartBuilder::on(area)
        .margin(20)
        .caption(title, ("sans-serif", 28).into_font().color(&WHITE))
        .x_label_area_size(45)
        .y_label_area_size(70)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let y_label_formatter = |v: &f64| format_top_axis_label(*v, chart_mode);
    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("block")
        .y_desc(chart_mode.y_axis_label())
        .x_label_style(("sans-serif", 16).into_font().color(&WHITE))
        .y_label_style(("sans-serif", 16).into_font().color(&WHITE))
        .axis_style(RGBColor(130, 130, 160))
        .label_style(("sans-serif", 14).into_font().color(&WHITE))
        .y_label_formatter(&y_label_formatter)
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let overlay_series = build_tx_overlays(&grouped_filters, x_min, x_max, y_min, y_max, area);
    for (color, points) in overlay_series {
        let style = ShapeStyle::from(color).filled();
        chart
            .draw_series(points.iter().map(|(x, y)| Circle::new((*x, *y), 3, style)))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    }

    if let Some(span) = ema_span {
        for (_label, color_idx, series) in &grouped_filters {
            let ema = compute_ema(series, span);
            let (r, g, b) = lighten_rgb(filter_rgb(*color_idx), 0.5);
            chart
                .draw_series(LineSeries::new(
                    ema.into_iter(),
                    ShapeStyle::from(RGBColor(r, g, b)).stroke_width(2),
                ))
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        }

        let visible_agg = filter_visible(snapshot.aggregate_series_for(chart_mode), x_min, x_max);
        let grouped_agg = group_series_sum(visible_agg, granularity);
        let ema_agg = compute_ema(&grouped_agg, span);
        let (r, g, b) = lighten_rgb((255, 255, 255), 0.5);
        chart
            .draw_series(LineSeries::new(
                ema_agg.into_iter(),
                ShapeStyle::from(RGBColor(r, g, b)).stroke_width(2),
            ))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    }

    for (label, color_idx, _) in &grouped_filters {
        let (r, g, b) = filter_rgb(*color_idx);
        let color = RGBColor(r, g, b);
        let legend_color = color;
        let dummy_x = x_min - 1.0;
        let dummy_y = y_min - 1.0;
        chart
            .draw_series(std::iter::once(Circle::new((dummy_x, dummy_y), 3, color.filled())))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .label(label.clone())
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], legend_color.filled()));
    }

    chart
        .configure_series_labels()
        .border_style(RGBColor(80, 80, 100))
        .background_style(RGBColor(30, 30, 45))
        .label_font(("sans-serif", 14).into_font().color(&WHITE))
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    Ok(())
}


fn render_mid_chart<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    snapshot: &AnalysisSnapshot,
    chart_mode: ChartMode,
    x_min: f64,
    x_max: f64,
    granularity: usize,
    scale_mode: ScaleMode,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let block_series = snapshot.block_series_for(chart_mode);
    let visible = filter_visible(block_series, x_min, x_max);
    let grouped_mid = match chart_mode {
        ChartMode::TxCount => group_series_avg(visible, granularity),
        ChartMode::GasUsed | ChartMode::TxSize => group_series_sum(visible, granularity),
    };

    let use_scale = chart_mode == ChartMode::TxCount;
    let scale = if use_scale {
        scale_mode.build_transform(&grouped_mid)
    } else {
        ScaleMode::Linear.build_transform(&grouped_mid)
    };

    let scaled_mid: Vec<(f64, f64)> = grouped_mid
        .iter()
        .map(|(x, y)| (*x, scale.apply(*y)))
        .collect();

    let (y_min, y_max) = series_y_bounds(&[&scaled_mid]);
    let title = format!(
        "{}{}{}",
        chart_mode.mid_title(),
        granularity_suffix(granularity),
        if use_scale { scale_mode.label() } else { "" }
    );

    let mut chart = ChartBuilder::on(area)
        .margin(20)
        .caption(title, ("sans-serif", 28).into_font().color(&WHITE))
        .x_label_area_size(45)
        .y_label_area_size(70)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let unit = if use_scale {
        let originals: Vec<f64> = scaled_mid.iter().map(|(_, y)| scale.invert(*y)).collect();
        let max_original = originals.iter().fold(0.0_f64, |a, b| a.max(*b));
        Some(pick_fee_unit(max_original))
    } else {
        None
    };

    let y_label_formatter = move |v: &f64| {
        if let Some(unit) = unit {
            let original = scale.invert(*v);
            format_fee_with_unit(original, unit)
        } else {
            format_mid_axis_label(*v, chart_mode)
        }
    };

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("block")
        .y_desc(chart_mode.y_axis_label())
        .x_label_style(("sans-serif", 16).into_font().color(&WHITE))
        .y_label_style(("sans-serif", 16).into_font().color(&WHITE))
        .axis_style(RGBColor(130, 130, 160))
        .label_style(("sans-serif", 14).into_font().color(&WHITE))
        .y_label_formatter(&y_label_formatter)
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let base_color = RGBColor(169, 169, 169);
    match chart_mode {
        ChartMode::TxCount => {
            chart
                .draw_series(LineSeries::new(
                    scaled_mid.iter().cloned(),
                    ShapeStyle::from(&base_color).stroke_width(2),
                ))
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        }
        ChartMode::GasUsed | ChartMode::TxSize => {
            chart
                .draw_series(scaled_mid.iter().map(|(x, y)| {
                    Circle::new((*x, *y), 2, base_color.filled())
                }))
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        }
    }

    let grouped_filters = build_grouped_filters(snapshot, chart_mode, x_min, x_max, granularity);
    let mid_by_x: HashMap<u64, f64> = scaled_mid
        .iter()
        .map(|(x, y)| (*x as u64, *y))
        .collect();
    let overlay_series = build_mid_overlays(&grouped_filters, &mid_by_x, x_min, x_max, area);
    for (color, points) in overlay_series {
        let style = ShapeStyle::from(color).filled();
        chart
            .draw_series(points.iter().map(|(x, y)| Circle::new((*x, *y), 3, style)))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    }

    for (label, color_idx, _) in &grouped_filters {
        let (r, g, b) = filter_rgb(*color_idx);
        let color = RGBColor(r, g, b);
        let legend_color = color;
        let dummy_x = x_min - 1.0;
        let dummy_y = y_min - 1.0;
        chart
            .draw_series(std::iter::once(Circle::new((dummy_x, dummy_y), 3, color.filled())))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .label(label.clone())
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], legend_color.filled()));
    }

    chart
        .configure_series_labels()
        .border_style(RGBColor(80, 80, 100))
        .background_style(RGBColor(30, 30, 45))
        .label_font(("sans-serif", 14).into_font().color(&WHITE))
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    Ok(())
}


fn render_hist_chart<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    snapshot: &AnalysisSnapshot,
    chart_mode: ChartMode,
    x_min: f64,
    x_max: f64,
    granularity: usize,
    hist_mode: HistogramMode,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let title = format!(
        "{} histogram — {}{}",
        hist_title_prefix(chart_mode),
        hist_mode_label(hist_mode),
        granularity_suffix(granularity)
    );
    match hist_mode {
        HistogramMode::FilterMatches => {
            render_histogram_filter_matches(area, snapshot, chart_mode, x_min, x_max, title)
        }
        HistogramMode::AllBlocks => {
            render_histogram_all_blocks(area, snapshot, chart_mode, x_min, x_max, granularity, title)
        }
        HistogramMode::Stacked => {
            render_histogram_stacked(area, snapshot, chart_mode, x_min, x_max, granularity, title)
        }
    }
}


fn render_histogram_filter_matches<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    snapshot: &AnalysisSnapshot,
    chart_mode: ChartMode,
    x_min: f64,
    x_max: f64,
    title: String,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
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
                snapshot.filter_series_for(chart_mode, &f.id).map(|series| {
                    let visible = filter_visible(series, x_min, x_max);
                    let hist = accumulate_histogram_weighted(visible, &value_lookup, chart_mode);
                    (f.label.clone(), f.color_index, hist)
                })
            })
            .collect()
    };

    let max_buckets = 60usize;
    let mut entries: Vec<(f64, String, f64, RGBColor)> = Vec::new();
    for (label, color_idx, hist) in &owned_hists {
        let merged = smart_rebucket(hist, max_buckets, chart_mode);
        for (lo, hi, count) in merged {
            let prefix = if label.is_empty() { "" } else { label.as_str() };
            let bucket_label = format_bucket_label(lo, hi, chart_mode);
            let display = if prefix.is_empty() {
                bucket_label
            } else {
                format!("{prefix}{bucket_label}")
            };
            let color = if snapshot.show_aggregate {
                RGBColor(255, 255, 255)
            } else {
                let (r, g, b) = filter_rgb(*color_idx);
                RGBColor(r, g, b)
            };
            entries.push((lo, display, count, color));
        }
    }

    if entries.is_empty() {
        return Ok(());
    }

    entries.sort_by(|a, b| {
        a.0
            .partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    let bars: Vec<HistEntry> = entries
        .into_iter()
        .map(|(_, label, count, color)| HistEntry { label, count, color })
        .collect();

    let legend: Vec<(String, RGBColor)> = if snapshot.show_aggregate {
        vec![("aggregate".to_string(), RGBColor(255, 255, 255))]
    } else {
        snapshot
            .filters
            .iter()
            .filter(|f| f.enabled)
            .map(|f| {
                let (r, g, b) = filter_rgb(f.color_index);
                (f.label.clone(), RGBColor(r, g, b))
            })
            .collect()
    };

    draw_histogram_bars(area, chart_mode, title, bars, &legend)
}


fn render_histogram_all_blocks<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    snapshot: &AnalysisSnapshot,
    chart_mode: ChartMode,
    x_min: f64,
    x_max: f64,
    granularity: usize,
    title: String,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let block_series = snapshot.block_series_for(chart_mode);
    let visible = filter_visible(block_series, x_min, x_max);
    let source = if granularity > 1 {
        match chart_mode {
            ChartMode::TxCount => group_series_avg(visible, granularity),
            ChartMode::GasUsed | ChartMode::TxSize => group_series_sum(visible, granularity),
        }
    } else {
        visible.to_vec()
    };

    let hist_data = accumulate_histogram(&source, chart_mode);
    let max_buckets = 60usize;
    let merged = smart_rebucket(&hist_data, max_buckets, chart_mode);

    let enabled_filters: Vec<_> = snapshot.filters.iter().filter(|f| f.enabled).collect();
    let value_lookup = build_value_lookup(snapshot, chart_mode);
    let visible_filter_hists: HashMap<FilterId, Vec<(f64, f64)>> = enabled_filters
        .iter()
        .filter_map(|f| {
            snapshot.filter_series_for(chart_mode, &f.id).map(|series| {
                let vis = filter_visible(series, x_min, x_max);
                (f.id, accumulate_histogram_weighted(vis, &value_lookup, chart_mode))
            })
        })
        .collect();

    let mut entries: Vec<(f64, String, f64, RGBColor)> = Vec::new();
    for (lo, hi, count) in merged {
        let label = format_bucket_label(lo, hi, chart_mode);
        let matching_colors: Vec<(u8, u8, u8)> = enabled_filters
            .iter()
            .filter(|f| {
                visible_filter_hists
                    .get(&f.id)
                    .map(|h| bucket_overlaps(lo, hi, h))
                    .unwrap_or(false)
            })
            .map(|f| filter_rgb(f.color_index))
            .collect();

        let color = if matching_colors.is_empty() {
            RGBColor(169, 169, 169)
        } else {
            blend_colors(&matching_colors)
        };
        entries.push((lo, label, count, color));
    }

    if entries.is_empty() {
        return Ok(());
    }

    entries.sort_by(|a, b| {
        a.0
            .partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    let bars: Vec<HistEntry> = entries
        .into_iter()
        .map(|(_, label, count, color)| HistEntry { label, count, color })
        .collect();

    let mut legend: Vec<(String, RGBColor)> = Vec::new();
    legend.push(("unmatched".to_string(), RGBColor(169, 169, 169)));
    for f in &enabled_filters {
        let (r, g, b) = filter_rgb(f.color_index);
        legend.push((f.label.clone(), RGBColor(r, g, b)));
    }

    draw_histogram_bars(area, chart_mode, title, bars, &legend)
}


fn render_histogram_stacked<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    snapshot: &AnalysisSnapshot,
    chart_mode: ChartMode,
    x_min: f64,
    x_max: f64,
    granularity: usize,
    title: String,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let block_series = snapshot.block_series_for(chart_mode);
    let visible = filter_visible(block_series, x_min, x_max);
    let enabled_filters: Vec<_> = snapshot.filters.iter().filter(|f| f.enabled).collect();

    let use_value_attribution = matches!(chart_mode, ChartMode::GasUsed | ChartMode::TxSize);

    let filter_value_lookups: Vec<HashMap<u64, f64>> = enabled_filters
        .iter()
        .map(|f| {
            snapshot
                .filter_series_for(chart_mode, &f.id)
                .map(|series| {
                    filter_visible(series, x_min, x_max)
                        .iter()
                        .map(|(block, val)| (*block as u64, *val))
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect();

    struct BlockEntry {
        bucket: f64,
        block_value: f64,
        filter_values: Vec<f64>,
    }

    let entries: Vec<BlockEntry> = visible
        .chunks(granularity.max(1))
        .map(|chunk| {
            let agg_value = match chart_mode {
                ChartMode::TxCount => {
                    let avg = chunk.iter().map(|(_, f)| f).sum::<f64>() / chunk.len() as f64;
                    (avg * bucket_precision(chart_mode)).floor() / bucket_precision(chart_mode)
                }
                ChartMode::GasUsed | ChartMode::TxSize => {
                    let sum = chunk.iter().map(|(_, f)| f).sum::<f64>();
                    (sum * bucket_precision(chart_mode)).floor() / bucket_precision(chart_mode)
                }
            };
            let block_value = chunk.iter().map(|(_, v)| v).sum::<f64>();
            let filter_values: Vec<f64> = filter_value_lookups
                .iter()
                .map(|lookup| {
                    chunk
                        .iter()
                        .map(|(b, _)| lookup.get(&(*b as u64)).copied().unwrap_or(0.0))
                        .sum()
                })
                .collect();
            BlockEntry {
                bucket: agg_value,
                block_value,
                filter_values,
            }
        })
        .collect();

    let mut raw_tree: BTreeMap<i64, f64> = BTreeMap::new();
    for entry in &entries {
        let key = quantize_bucket(entry.bucket, chart_mode);
        if use_value_attribution {
            *raw_tree.entry(key).or_default() += entry.block_value;
        } else {
            *raw_tree.entry(key).or_default() += 1.0;
        }
    }
    let raw_hist: Vec<(f64, f64)> = raw_tree
        .into_iter()
        .map(|(k, c)| (key_to_bucket(k, chart_mode), c))
        .collect();

    let max_buckets = 60usize;
    let merged = smart_rebucket(&raw_hist, max_buckets, chart_mode);

    if merged.is_empty() {
        return Ok(());
    }

    let stacked_bars: Vec<StackedBar> = if use_value_attribution {
        merged
            .iter()
            .map(|(lo, hi, total)| {
                let mut per_filter: Vec<f64> = vec![0.0; enabled_filters.len()];
                let mut total_block_value = 0.0_f64;
                for entry in &entries {
                    if entry.bucket >= *lo - 1e-9 && entry.bucket <= *hi + 1e-9 {
                        total_block_value += entry.block_value;
                        for (i, &fv) in entry.filter_values.iter().enumerate() {
                            per_filter[i] += fv;
                        }
                    }
                }
                let matched_total: f64 = per_filter.iter().sum();
                let unmatched = (total_block_value - matched_total).max(0.0);

                let mut segments: Vec<Segment> = Vec::new();
                if unmatched > 0.0 {
                    segments.push(Segment {
                        count: unmatched,
                        color: RGBColor(169, 169, 169),
                    });
                }
                for (i, &val) in per_filter.iter().enumerate() {
                    if val > 0.0 {
                        let (r, g, b) = filter_rgb(enabled_filters[i].color_index);
                        segments.push(Segment {
                            count: val,
                            color: RGBColor(r, g, b),
                        });
                    }
                }
                StackedBar {
                    label: format_bucket_label(*lo, *hi, chart_mode),
                    total: *total,
                    segments,
                }
            })
            .collect()
    } else {
        merged
            .iter()
            .map(|(lo, hi, total)| {
                let mut mask_counts: HashMap<u16, f64> = HashMap::new();
                for entry in &entries {
                    if entry.bucket >= *lo - 1e-9 && entry.bucket <= *hi + 1e-9 {
                        let mask: u16 = entry
                            .filter_values
                            .iter()
                            .enumerate()
                            .filter(|(_, v)| **v > 0.0)
                            .fold(0u16, |acc, (i, _)| acc | (1 << i));
                        *mask_counts.entry(mask).or_default() += 1.0;
                    }
                }

                let mut segments: Vec<Segment> = Vec::new();
                if let Some(&count) = mask_counts.get(&0)
                    && count > 0.0
                {
                    segments.push(Segment {
                        count,
                        color: RGBColor(169, 169, 169),
                    });
                }

                let mut matched: Vec<(u16, f64)> = mask_counts.into_iter().filter(|(m, _)| *m != 0).collect();
                matched.sort_by_key(|(mask, _)| *mask);

                for (mask, count) in matched {
                    let indices: Vec<usize> = (0..enabled_filters.len())
                        .filter(|i| mask & (1 << *i) != 0)
                        .collect();
                    let color = if indices.len() == 1 {
                        let (r, g, b) = filter_rgb(enabled_filters[indices[0]].color_index);
                        RGBColor(r, g, b)
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
                    label: format_bucket_label(*lo, *hi, chart_mode),
                    total: *total,
                    segments,
                }
            })
            .collect()
    };

    let unmatched_label = if use_value_attribution {
        "unmatched"
    } else {
        "no match"
    };
    let mut legend: Vec<(String, RGBColor)> = Vec::new();
    legend.push((unmatched_label.to_string(), RGBColor(169, 169, 169)));
    for f in &enabled_filters {
        let (r, g, b) = filter_rgb(f.color_index);
        legend.push((f.label.clone(), RGBColor(r, g, b)));
    }

    draw_histogram_stacked(area, chart_mode, title, stacked_bars, &legend)
}


struct HistEntry {
    label: String,
    count: f64,
    color: RGBColor,
}


struct Segment {
    count: f64,
    color: RGBColor,
}


struct StackedBar {
    label: String,
    total: f64,
    segments: Vec<Segment>,
}


fn draw_histogram_bars<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    chart_mode: ChartMode,
    title: String,
    entries: Vec<HistEntry>,
    legend: &[(String, RGBColor)],
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    if entries.is_empty() {
        return Ok(());
    }

    let max_count = entries
        .iter()
        .map(|entry| entry.count)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let n = entries.len();
    let labels: Vec<String> = entries.iter().map(|entry| entry.label.clone()).collect();

    let mut chart = ChartBuilder::on(area)
        .margin(20)
        .caption(title, ("sans-serif", 28).into_font().color(&WHITE))
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..n as f64, 0f64..max_count)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc(hist_x_axis_label(chart_mode))
        .y_desc("count")
        .x_label_style(("sans-serif", 14).into_font().color(&WHITE))
        .y_label_style(("sans-serif", 14).into_font().color(&WHITE))
        .axis_style(RGBColor(130, 130, 160))
        .label_style(("sans-serif", 12).into_font().color(&WHITE))
        .x_labels(n.min(10))
        .x_label_formatter(&move |v| {
            let idx = (*v).round() as usize;
            labels.get(idx).cloned().unwrap_or_default()
        })
        .y_label_formatter(&|v| format!("{:.0}", v))
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    for (idx, entry) in entries.iter().enumerate() {
        let x0 = idx as f64 + 0.1;
        let x1 = (idx + 1) as f64 - 0.1;
        let rect = Rectangle::new([(x0, 0.0), (x1, entry.count)], entry.color.filled());
        chart
            .draw_series(std::iter::once(rect))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    }

    if !legend.is_empty() {
        let dummy_x = -1.0_f64;
        let dummy_y = -1.0_f64;
        for (label, color) in legend {
            let legend_color = *color;
            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(dummy_x, dummy_y), (dummy_x, dummy_y)],
                    legend_color.filled(),
                )))
                .map_err(|e| anyhow::anyhow!("{e:?}"))?
                .label(label.clone())
                .legend(move |(x, y)| {
                    Rectangle::new([(x, y - 5), (x + 10, y + 5)], legend_color.filled())
                });
        }

        chart
            .configure_series_labels()
            .border_style(RGBColor(80, 80, 100))
            .background_style(RGBColor(30, 30, 45))
            .label_font(("sans-serif", 14).into_font().color(&WHITE))
            .draw()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    }

    Ok(())
}


fn draw_histogram_stacked<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    chart_mode: ChartMode,
    title: String,
    stacked_bars: Vec<StackedBar>,
    legend: &[(String, RGBColor)],
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    if stacked_bars.is_empty() {
        return Ok(());
    }

    let max_total = stacked_bars
        .iter()
        .map(|b| b.total)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let n = stacked_bars.len();
    let labels: Vec<String> = stacked_bars.iter().map(|bar| bar.label.clone()).collect();

    let y_desc = match chart_mode {
        ChartMode::TxCount => "count",
        ChartMode::GasUsed => "gas",
        ChartMode::TxSize => "bytes",
    };

    let mut chart = ChartBuilder::on(area)
        .margin(20)
        .caption(title, ("sans-serif", 28).into_font().color(&WHITE))
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..n as f64, 0f64..max_total)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc(hist_x_axis_label(chart_mode))
        .y_desc(y_desc)
        .x_label_style(("sans-serif", 14).into_font().color(&WHITE))
        .y_label_style(("sans-serif", 14).into_font().color(&WHITE))
        .axis_style(RGBColor(130, 130, 160))
        .label_style(("sans-serif", 12).into_font().color(&WHITE))
        .x_labels(n.min(10))
        .x_label_formatter(&move |v| {
            let idx = (*v).round() as usize;
            labels.get(idx).cloned().unwrap_or_default()
        })
        .y_label_formatter(&|v| format!("{:.0}", v))
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    for (idx, bar) in stacked_bars.iter().enumerate() {
        let x0 = idx as f64 + 0.1;
        let x1 = (idx + 1) as f64 - 0.1;
        let mut current = 0.0;
        for seg in &bar.segments {
            let y0 = current;
            let y1 = current + seg.count;
            current = y1;
            let rect = Rectangle::new([(x0, y0), (x1, y1)], seg.color.filled());
            chart
                .draw_series(std::iter::once(rect))
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        }
    }

    if !legend.is_empty() {
        let dummy_x = -1.0_f64;
        let dummy_y = -1.0_f64;
        for (label, color) in legend {
            let legend_color = *color;
            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(dummy_x, dummy_y), (dummy_x, dummy_y)],
                    legend_color.filled(),
                )))
                .map_err(|e| anyhow::anyhow!("{e:?}"))?
                .label(label.clone())
                .legend(move |(x, y)| {
                    Rectangle::new([(x, y - 5), (x + 10, y + 5)], legend_color.filled())
                });
        }

        chart
            .configure_series_labels()
            .border_style(RGBColor(80, 80, 100))
            .background_style(RGBColor(30, 30, 45))
            .label_font(("sans-serif", 14).into_font().color(&WHITE))
            .draw()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    }

    Ok(())
}


fn build_grouped_filters(
    snapshot: &AnalysisSnapshot,
    chart_mode: ChartMode,
    x_min: f64,
    x_max: f64,
    granularity: usize,
) -> FilterSeries {
    snapshot
        .filters
        .iter()
        .filter(|f| f.enabled)
        .filter_map(|f| {
            snapshot.filter_series_for(chart_mode, &f.id).map(|series| {
                let visible = filter_visible(series, x_min, x_max);
                (f.label.clone(), f.color_index, group_series_sum(visible, granularity))
            })
        })
        .collect()
}


fn build_tx_overlays<DB: DrawingBackend>(
    grouped_filter_series: &FilterSeries,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    area: &DrawingArea<DB, Shift>,
) -> Vec<(RGBColor, Vec<(f64, f64)>)>
where
    DB::ErrorType: 'static,
{
    let (width, height): (u32, u32) = area.dim_in_pixel();
    let width = width.max(1) as f64;
    let height = height.max(1) as f64;
    let cell_w = (x_max - x_min) / width;
    let cell_h = (y_max - y_min) / height;

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

    let mut cell_colors: HashMap<(i64, i64), RGBColor> = HashMap::new();
    for (cell, mut indices) in cell_filters {
        indices.sort();
        indices.dedup();
        cell_colors.insert(cell, cell_color(&indices, grouped_filter_series));
    }

    let mut color_points: HashMap<RGBColor, Vec<(f64, f64)>> = HashMap::new();
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

    let mut result: Vec<(RGBColor, Vec<(f64, f64)>)> = color_points.into_iter().collect();
    for (_, series) in &mut result {
        series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }
    result
}


fn build_mid_overlays<DB: DrawingBackend>(
    grouped_filter_series: &FilterSeries,
    mid_by_x: &HashMap<u64, f64>,
    x_min: f64,
    x_max: f64,
    area: &DrawingArea<DB, Shift>,
) -> Vec<(RGBColor, Vec<(f64, f64)>)>
where
    DB::ErrorType: 'static,
{
    let (width, _): (u32, u32) = area.dim_in_pixel();
    let width = width.max(1) as f64;
    let cell_w = (x_max - x_min) / width;

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
    for indices in cell_filters.values_mut() {
        indices.sort();
        indices.dedup();
    }

    let cell_colors: HashMap<i64, RGBColor> = cell_filters
        .iter()
        .map(|(cx, indices)| (*cx, cell_color(indices, grouped_filter_series)))
        .collect();

    let mut color_buckets: HashMap<RGBColor, Vec<(f64, f64)>> = HashMap::new();
    for (_label, _color_idx, series) in grouped_filter_series {
        for &(x, y) in series {
            if y > 0.0 {
                let cx = quantize(x, x_min, cell_w);
                if let Some(&color) = cell_colors.get(&cx)
                    && let Some(&mid_val) = mid_by_x.get(&(x as u64))
                {
                    color_buckets.entry(color).or_default().push((x, mid_val));
                }
            }
        }
    }

    let mut result: Vec<(RGBColor, Vec<(f64, f64)>)> = color_buckets.into_iter().collect();
    for (_, series) in &mut result {
        series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }
    result
}


fn quantize(val: f64, min: f64, cell_size: f64) -> i64 {
    if cell_size <= 0.0 {
        return val as i64;
    }
    ((val - min) / cell_size).floor() as i64
}


fn cell_color(
    key: &[usize],
    grouped_filter_series: &FilterSeries,
) -> RGBColor {
    if key.len() == 1 {
        let (r, g, b) = filter_rgb(grouped_filter_series[key[0]].1);
        RGBColor(r, g, b)
    } else {
        let rgbs: Vec<(u8, u8, u8)> = key.iter().map(|&i| filter_rgb(grouped_filter_series[i].1)).collect();
        blend_colors(&rgbs)
    }
}


fn series_x_bounds(series: &[(f64, f64)]) -> (f64, f64) {
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


fn filter_visible(series: &[(f64, f64)], x_min: f64, x_max: f64) -> &[(f64, f64)] {
    let start = series.partition_point(|(x, _)| *x < x_min);
    let end = series.partition_point(|(x, _)| *x <= x_max);
    &series[start..end]
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


fn granularity_suffix(granularity: usize) -> String {
    if granularity > 1 {
        format!(" ({}blk)", granularity)
    } else {
        String::new()
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


fn format_fee_label(lo: f64, hi: f64, unit: FeeUnit) -> String {
    if (hi - lo).abs() < 0.0005 {
        format_fee_with_unit(lo, unit)
    } else {
        let lo_s = format_fee_with_unit(lo, unit);
        let hi_s = format_fee_with_unit(hi, unit);
        format!("{lo_s}-{hi_s}")
    }
}


fn format_bytes(bytes: f64) -> String {
    format_si(bytes, "B")
}


fn format_gas(gas: f64) -> String {
    format_si(gas, "gas")
}


fn format_si(value: f64, unit: &str) -> String {
    if value >= 1_000_000_000.0 {
        format!("{:.1}G{unit}", value / 1_000_000_000.0)
    } else if value >= 1_000_000.0 {
        format!("{:.1}M{unit}", value / 1_000_000.0)
    } else if value >= 1_000.0 {
        format!("{:.0}K{unit}", value / 1_000.0)
    } else {
        format!("{:.0}{unit}", value)
    }
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


fn format_top_axis_label(value: f64, mode: ChartMode) -> String {
    match mode {
        ChartMode::TxCount => format!("{:.0}", value),
        ChartMode::GasUsed => format_gas(value),
        ChartMode::TxSize => format_bytes(value),
    }
}


fn format_mid_axis_label(value: f64, mode: ChartMode) -> String {
    match mode {
        ChartMode::TxCount => format_fee_value(value),
        ChartMode::GasUsed => format_gas(value),
        ChartMode::TxSize => format_bytes(value),
    }
}


fn hist_title_prefix(mode: ChartMode) -> &'static str {
    match mode {
        ChartMode::TxCount => "base fee",
        ChartMode::GasUsed => "gas",
        ChartMode::TxSize => "bytes",
    }
}


fn hist_mode_label(mode: HistogramMode) -> &'static str {
    match mode {
        HistogramMode::FilterMatches => "filter matches",
        HistogramMode::AllBlocks => "all blocks",
        HistogramMode::Stacked => "stacked",
    }
}


fn hist_x_axis_label(mode: ChartMode) -> &'static str {
    match mode {
        ChartMode::TxCount => "base fee",
        ChartMode::GasUsed => "gas",
        ChartMode::TxSize => "bytes",
    }
}


fn bucket_precision(mode: ChartMode) -> f64 {
    match mode {
        ChartMode::TxCount => 1000.0,
        ChartMode::GasUsed | ChartMode::TxSize => 1.0,
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


fn format_bucket_label(lo: f64, hi: f64, mode: ChartMode) -> String {
    match mode {
        ChartMode::TxCount => {
            let unit = pick_fee_unit(hi);
            format_fee_label(lo, hi, unit)
        }
        ChartMode::GasUsed => {
            if (hi - lo).abs() < 1.0 {
                format_gas(lo)
            } else {
                format!("{}-{}", format_gas(lo), format_gas(hi))
            }
        }
        ChartMode::TxSize => {
            if (hi - lo).abs() < 1.0 {
                format_bytes(lo)
            } else {
                format!("{}-{}", format_bytes(lo), format_bytes(hi))
            }
        }
    }
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
    series.iter().map(|(block, val)| (*block as u64, *val)).collect()
}


fn bucket_overlaps(bucket_lo: f64, bucket_hi: f64, raw: &[(f64, f64)]) -> bool {
    raw.iter()
        .any(|(b, count)| *count > 0.0 && *b >= bucket_lo - 1e-9 && *b <= bucket_hi + 1e-9)
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

    let lo = sorted.first().map(|v| v.0).unwrap_or(0.0);
    let hi = sorted.last().map(|v| v.0).unwrap_or(0.0);

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
        let idx = if width > 0.0 { ((val - lo) / width).floor() as usize } else { 0 };
        let idx = idx.min(target - 1);
        merged[idx].2 += count;
    }

    merged.retain(|&(_, _, c)| c > 0.0);
    merged
}


fn smart_rebucket(raw: &[(f64, f64)], max_buckets: usize, mode: ChartMode) -> Vec<(f64, f64, f64)> {
    match mode {
        ChartMode::TxCount => rebucket(raw, max_buckets),
        ChartMode::GasUsed | ChartMode::TxSize => rebucket_uniform(raw, max_buckets),
    }
}


const FILTER_PALETTE: [(u8, u8, u8); 6] = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
];


fn filter_rgb(index: usize) -> (u8, u8, u8) {
    FILTER_PALETTE[index % FILTER_PALETTE.len()]
}


fn compute_ema(series: &[(f64, f64)], span: usize) -> Vec<(f64, f64)> {
    if series.is_empty() || span == 0 {
        return Vec::new();
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut result = Vec::with_capacity(series.len());
    let mut ema = series[0].1;
    result.push((series[0].0, ema));
    for &(x, y) in &series[1..] {
        ema = alpha * y + (1.0 - alpha) * ema;
        result.push((x, ema));
    }
    result
}


fn lighten_rgb(rgb: (u8, u8, u8), factor: f64) -> (u8, u8, u8) {
    let f = factor.clamp(0.0, 1.0);
    let lift = |c: u8| {
        let lin = (c as f64 / 255.0).powi(2);
        let mixed = lin + (1.0 - lin) * f;
        (mixed.sqrt() * 255.0).round() as u8
    };
    (lift(rgb.0), lift(rgb.1), lift(rgb.2))
}


fn blend_colors(colors: &[(u8, u8, u8)]) -> RGBColor {
    if colors.is_empty() {
        return RGBColor(255, 255, 255);
    }
    if colors.len() == 1 {
        let (r, g, b) = colors[0];
        return RGBColor(r, g, b);
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
    RGBColor(r, g, b)
}


fn export_filename(format: ExportFormat) -> Result<String> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system time before unix epoch")?
        .as_secs() as i64;

    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&now, &mut tm) };
    let year = tm.tm_year + 1900;
    let month = tm.tm_mon + 1;
    let day = tm.tm_mday;
    let hour = tm.tm_hour;
    let min = tm.tm_min;
    let sec = tm.tm_sec;

    let ext = match format {
        ExportFormat::Png => "png",
        ExportFormat::Svg => "svg",
    };

    Ok(format!(
        "basescope_{:04}{:02}{:02}_{:02}{:02}{:02}.{}",
        year, month, day, hour, min, sec, ext
    ))
}

}

#[cfg(feature = "export")]
pub use imp::{export_charts, ExportChart, ExportFormat};
