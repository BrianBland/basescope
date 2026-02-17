use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Frame;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};

use crate::domain::approx_head_block;
use crate::tui::{App, RangeField};

pub(super) fn render_range_input(app: &App, frame: &mut Frame, area: Rect) {
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

    let start_block = Paragraph::new(app.input.start_block_input.as_str())
        .block(Block::default().title("start block").borders(Borders::ALL));
    let end_block = Paragraph::new(app.input.end_block_input.as_str())
        .block(Block::default().title("end block").borders(Borders::ALL));
    frame.render_widget(start_block, layout[1]);
    frame.render_widget(end_block, layout[2]);

    let hint = Paragraph::new(format!("approx head block: {}", approx_head_block()));
    frame.render_widget(hint, layout[3]);

    let instructions = Paragraph::new("?: help").block(Block::default().borders(Borders::TOP));
    frame.render_widget(instructions, layout[4]);

    let cursor_x = match app.input.range_field {
        RangeField::Start => layout[1].x + 1 + app.input.start_block_input.len() as u16,
        RangeField::End => layout[2].x + 1 + app.input.end_block_input.len() as u16,
    };
    let cursor_y = match app.input.range_field {
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

pub(super) fn render_filter_input(app: &App, frame: &mut Frame, area: Rect) {
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
            let style = if idx == app.input.selected_filter {
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

    let input = Paragraph::new(app.input.current_filter_input.as_str())
        .block(Block::default().title("filter input").borders(Borders::ALL));
    frame.render_widget(input, layout[2]);
    frame.set_cursor_position((
        layout[2].x + 1 + app.input.current_filter_input.len() as u16,
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
