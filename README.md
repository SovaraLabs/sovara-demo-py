# Sovara FinanceBench Demo

This demo runs a FinanceBench RAG workflow and checks the generated answer
against the benchmark answer.

## 1. Clone the demo

This repository uses Git LFS for the FinanceBench PDFs and PageIndex data.
Install and enable Git LFS before cloning:

```sh
brew install git-lfs
git lfs install
```

On Windows, install Git LFS from https://git-lfs.com/ and then run:

```powershell
git lfs install
```

Clone this repository so the path is exactly `~/sovara-demo`:

```sh
git clone https://github.com/SovaraLabs/sovara-demo-py ~/sovara-demo
```

Then open a terminal in the folder:

```sh
cd ~/sovara-demo
```

## 2. Install uv

If you do not already have `uv`, install it with:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows PowerShell, use:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If your terminal still says `uv: command not found`, close and reopen the
terminal, then run `uv --version`.

## 3. Add model API keys

Create a local `.env` file from the example:

```sh
cp .env.example .env
```

Then open `~/sovara-demo/.env` and fill in the required values:

```text
OPENAI_API_KEY=<your OpenAI API key>
ANTHROPIC_API_KEY=<your Anthropic API key>
```

## 4. Run the first sample

From the demo folder, run sample `81`:

```sh
uv run main.py --sample-id 81
```

The first run can take a few minutes because `uv` may need to create the Python
environment and install dependencies. When the command completes, it prints a
JSON result in the terminal.

## 5. Run three more samples

Run a few more sample IDs to compare their results:

```sh
uv run main.py --sample-id 82
uv run main.py --sample-id 83
uv run main.py --sample-id 84
```

## Questions

Get in touch at [hello@sovara-labs.com](mailto:hello@sovara-labs.com) or join
our Discord server: https://discord.gg/y8qFDUwX
