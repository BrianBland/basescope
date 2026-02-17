use ratatui::style::Color;

pub(super) const FILTER_PALETTE: [Color; 6] = [
    Color::Rgb(255, 0, 0),
    Color::Rgb(0, 0, 255),
    Color::Rgb(0, 255, 0),
    Color::Rgb(255, 255, 0),
    Color::Rgb(0, 255, 255),
    Color::Rgb(255, 0, 255),
];

pub(super) fn filter_color(index: usize) -> Color {
    let palette = FILTER_PALETTE;
    palette[index % palette.len()]
}

pub(super) fn filter_rgb(index: usize) -> (u8, u8, u8) {
    match FILTER_PALETTE[index % FILTER_PALETTE.len()] {
        Color::Rgb(r, g, b) => (r, g, b),
        _ => (255, 255, 255),
    }
}

pub(super) fn blend_colors(colors: &[(u8, u8, u8)]) -> Color {
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
