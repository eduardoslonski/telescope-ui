<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ui/public/logo-full-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="ui/public/logo-full.svg">
    <img alt="Telescope" src="https://raw.githubusercontent.com/eduardoslonski/telescope-ui/main/ui/public/logo-full.svg" height="80">
  </picture>
</p>

<p align="center">
  The visualization dashboard for <a href="https://github.com/eduardoslonski/telescope">Telescope</a> — a framework for post-training LLMs with reinforcement learning, designed for reasoning and agentic capabilities.
</p>

# Telescope UI

Telescope UI runs locally on your machine and syncs training data from Weights & Biases into a local DuckDB database, giving you a real-time dashboard to monitor runs, inspect rollouts, and analyze metrics.

## Installation

```bash
pip install telescope-ui
```

Requires Python 3.11+.

## Quick start

```bash
wandb login    # authenticate with W&B (once)
telescope      # starts the dashboard at http://localhost:8005
```

Telescope automatically discovers runs tagged with `telescope` in your W&B projects. The default training config already includes this tag, so runs appear in the UI automatically.

| Flag | Description |
|------|-------------|
| `--port PORT` | Port to serve on (default: 8005) |
| `--host HOST` | Host to bind to (default: 127.0.0.1) |
| `--no-browser` | Don't open the browser automatically |
| `--debug` | Enable verbose logging |

## Documentation

Full documentation at [docs.telescope.training](https://docs.telescope.training).

## Pages

### Metrics

![Metrics](https://raw.githubusercontent.com/eduardoslonski/telescope-ui/main/assets/metrics.png)

### Rollouts

![Rollouts](https://raw.githubusercontent.com/eduardoslonski/telescope-ui/main/assets/rollouts.png)

### Timeline

![Timeline — Inference](https://raw.githubusercontent.com/eduardoslonski/telescope-ui/main/assets/timeline-inference.png)

![Timeline — Orchestrator & Trainer](https://raw.githubusercontent.com/eduardoslonski/telescope-ui/main/assets/timeline-orchestrator-trainer.png)

### Infra

![Infra — Topology](https://raw.githubusercontent.com/eduardoslonski/telescope-ui/main/assets/topology.png)

### Evals

![Evals](https://raw.githubusercontent.com/eduardoslonski/telescope-ui/main/assets/evals.png)

### Code Compare

![Code Compare](https://raw.githubusercontent.com/eduardoslonski/telescope-ui/main/assets/code.png)

## Data storage

All synced data is stored locally at `~/.telescope/`. Override with the `TELESCOPE_DATA_DIR` environment variable. Database compression is available through the sidebar menu and can reduce size by 20-30%.

## License

MIT
