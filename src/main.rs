mod analysis;
mod cache;
mod domain;
mod error;
mod pipeline;
mod rpc;
mod tui;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use crate::domain::{parse_filter, ScanSpec, TxFilter};
use crate::rpc::RpcClient;
use crate::tui::log_layer::{LogBuffer, TuiLogLayer};

const DEFAULT_RPC_URL: &str = "https://mainnet.base.org";

fn init_logging(data_dir: &std::path::Path, log_buffer: &LogBuffer) -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    std::fs::create_dir_all(data_dir)?;
    let log_file = std::fs::File::create(data_dir.join("basescope.log"))?;

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(log_file)
        .with_ansi(false);

    let tui_layer = TuiLogLayer::new(log_buffer.clone());

    tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .with(tui_layer)
        .init();

    Ok(())
}

#[derive(Debug, Parser)]
#[command(name = "basescope")]
struct Cli {
    #[arg(long, env = "BASESCOPE_RPC_URLS", value_delimiter = ',')]
    rpc_url: Vec<String>,

    #[arg(long, default_value = "./data", env = "BASESCOPE_DATA_DIR")]
    data_dir: PathBuf,

    #[arg(long, env = "BASESCOPE_START_BLOCK")]
    start_block: Option<u64>,

    #[arg(long, env = "BASESCOPE_END_BLOCK")]
    end_block: Option<u64>,

    #[arg(long)]
    filter: Vec<String>,

    #[arg(long, default_value = "10", env = "BASESCOPE_CONCURRENCY")]
    concurrency: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let cli = Cli::parse();
    let log_buffer = LogBuffer::new();
    init_logging(&cli.data_dir, &log_buffer)?;
    let rpc_urls = if cli.rpc_url.is_empty() {
        vec![DEFAULT_RPC_URL.to_string()]
    } else {
        cli.rpc_url
    };
    let rpc_client = RpcClient::new(&rpc_urls)?;
    let cache = crate::cache::Cache::new(cli.data_dir)?;

    let mut filters: Vec<TxFilter> = Vec::new();
    for (idx, raw) in cli.filter.iter().enumerate() {
        let (kind, label) = parse_filter(raw)
            .map_err(|e| anyhow::anyhow!("bad --filter `{raw}`: {e}"))?;
        filters.push(TxFilter::new(idx, kind, label));
    }

    let cli_spec = match (cli.start_block, cli.end_block, filters.is_empty()) {
        (Some(start), Some(end), false) => {
            if start > end {
                return Err(anyhow::anyhow!("start block {start} > end block {end}"));
            }
            Some(ScanSpec {
                start_block: start,
                end_block: end,
                filters,
            })
        }
        _ => None,
    };

    let mut app = tui::App::new(rpc_client, cache, cli.concurrency, cli_spec, log_buffer);

    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture);
        let _ = crossterm::terminal::disable_raw_mode();
        ratatui::restore();
        original_hook(info);
    }));

    crossterm::terminal::enable_raw_mode()?;
    crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture)?;
    let mut terminal = ratatui::init();
    let result = app.run(&mut terminal);
    crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture)?;
    crossterm::terminal::disable_raw_mode()?;
    ratatui::restore();

    result
}
