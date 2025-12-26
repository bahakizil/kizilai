"""Main entrypoint for SesAI Voice AI Platform."""

import asyncio
import signal
import sys

from src.config.settings import get_settings
from src.services.telephony.audiosocket import AudioSocketServer
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


async def main() -> None:
    """Main application entrypoint."""
    # Load settings
    settings = get_settings()

    # Setup logging
    setup_logging(
        level=settings.log_level,
        format=settings.log_format,
    )

    logger.info(
        "sesai_starting",
        version="0.1.0",
        stt_model=settings.stt_model,
        tts_language=settings.tts_language,
        llm_model=settings.llm_model,
    )

    # Create AudioSocket server
    server = AudioSocketServer(settings)

    # Handle shutdown signals
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # Start server
    try:
        server_task = asyncio.create_task(server.start())

        logger.info(
            "sesai_ready",
            audiosocket_host=settings.audiosocket_host,
            audiosocket_port=settings.audiosocket_port,
        )

        # Wait for shutdown signal
        await shutdown_event.wait()

        # Graceful shutdown
        logger.info("sesai_shutting_down")
        server_task.cancel()

        try:
            await server_task
        except asyncio.CancelledError:
            pass

        await server.stop()

    except Exception as e:
        logger.error("sesai_error", error=str(e))
        raise

    logger.info("sesai_stopped")


def run() -> None:
    """Run the application."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
