# Discord Upscaling Bot

A Discord bot that performs image upscaling using various super-resolution models. Users can easily upload images to this bot and have them upscaled with the model of their choice.

## Features

- Supports multiple super-resolution models through [spandrel](https://github.com/chaiNNer-org/spandrel/)
- Configurable settings for VRAM usage and processing limits
- Supports both .pth and .safetensors model formats
- Implements a customizable queuing system for handling multiple upscale requests
- Automatically adjusts tile size based on available VRAM
- Resize an image with typical scaling filters
- Fuzzy model name matching
- Alpha channel handling options: upscale, resize, or discard
- Detailed image information retrieval

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU
- Discord Bot Token

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Kim2091/upscale-bot.git
   cd upscale-bot
   ```

2. Install the latest PyTorch CUDA version for your system here: https://pytorch.org/get-started/locally/

3. Install the other required dependencies:
```
   pip install -r requirements.txt
   ```
4. Install [ImageMagick](https://imagemagick.org/script/download.php)

5. Open the `config.ini` file in the project root directory and replace `YOUR_DISCORD_BOT_TOKEN` and `YOUR_DISCORD_USER_ID` with your actual Discord bot token and user ID. Set this up at https://discord.com/developers/
     - The bot needs the `bot` scope (and maybe `applications.commands`), then:
       - Send Messages
       - Attach Files
       - View Channels
       - Use Slash Commands (maybe)

6. Place your models (.pth or .safetensors files) in the directory specified by `ModelPath` in the config file.

## Usage

1. Start the bot:
   ```
   python upscale-bot.py
   ```

2. In a Discord channel where the bot is present, use the following commands:

   - To upscale an image:
     ```
     --upscale <model_name>
     ```
     Attach the image you want to upscale when sending this command.

   - To list available models:
     ```
     --models
     ```
   - To resize an image:
     ```
     --resize <scale_factor> <method>
     ```
   - To get detailed information about an image:
     ```
     --info
     ```
     Attach the image or provide an image URL when sending this command.
## Configuration

You can adjust various settings in the `config.ini` file:

- `Precision`: Can be 'auto', 'fp16', 'bf16', or 'fp32'
- `DefaultTileSize` and `MaxTileSize`: Control the tile size for processing large images
- `MaxTotalPixels`: Maximum allowed input image size (width * height)
- `VRAMSafetyMultiplier` and `AvailableVRAMUsageFraction`: Fine-tune VRAM usage
- `ThreadPoolWorkers`: Number of worker threads for processing
- `MaxConcurrentUpscales`: Maximum number of concurrent upscale operations
- `UpscaleTimeout` and `OtherStepTimeout`: Timeouts for upscaling and other operations
- `UpscaleTimeout` and `OtherStepTimeout`: Timeouts for upscaling and other operations
- `DefaultAlphaHandling`: Default method for handling alpha channels

## Security

- Never share your Discord bot token.
- The bot implements basic input validation.

## Acknowledgements

- This bot uses the [Spandrel](https://github.com/chaiNNer-org/spandrel) library for loading and handling models.
- Thanks to [@the-database](https://github.com/the-database) for the benchmarks contained in `vram_data.csv`
- Thanks to the Discord.py team for their excellent Discord API wrapper.
- The bot uses [Wand](https://docs.wand-py.org/) (ImageMagick) for additional image processing capabilities.
