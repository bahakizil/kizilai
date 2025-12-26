"""AudioSocket server for FreeSWITCH telephony integration."""

import asyncio
import uuid
from typing import Callable, Awaitable

from src.config.settings import Settings
from src.core.pipeline import VoiceAgentPipeline, create_voice_agent
from src.utils.exceptions import TelephonyError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AudioSocketServer:
    """TCP server for AudioSocket connections from FreeSWITCH.

    AudioSocket is a simple protocol for bidirectional audio streaming:
    - Receives 8kHz, 16-bit, mono PCM audio from FreeSWITCH
    - Sends 8kHz, 16-bit, mono PCM audio back to FreeSWITCH
    - Frame size: 320 bytes (20ms at 8kHz)
    """

    def __init__(
        self,
        settings: Settings,
        pipeline_factory: Callable[[str, Settings], Awaitable[VoiceAgentPipeline]] | None = None,
    ):
        """Initialize the AudioSocket server.

        Args:
            settings: Application settings
            pipeline_factory: Factory function to create voice agent pipelines.
                              Defaults to create_voice_agent.
        """
        self.settings = settings
        self.host = settings.audiosocket_host
        self.port = settings.audiosocket_port
        self.frame_size = settings.frame_size_bytes

        self._pipeline_factory = pipeline_factory or create_voice_agent
        self._server: asyncio.Server | None = None
        self._active_calls: dict[str, VoiceAgentPipeline] = {}

        logger.info(
            "audiosocket_server_initialized",
            host=self.host,
            port=self.port,
            frame_size=self.frame_size,
        )

    async def start(self) -> None:
        """Start the AudioSocket server."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )

        logger.info(
            "audiosocket_server_started",
            host=self.host,
            port=self.port,
        )

        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        """Stop the server gracefully."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Stop all active calls
        for call_id, pipeline in self._active_calls.items():
            await pipeline.stop()

        self._active_calls.clear()
        logger.info("audiosocket_server_stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a new AudioSocket connection.

        Args:
            reader: Stream reader for incoming audio
            writer: Stream writer for outgoing audio
        """
        call_id = str(uuid.uuid4())
        peer = writer.get_extra_info("peername")

        logger.info("call_started", call_id=call_id, peer=peer)

        try:
            # Create voice agent pipeline
            pipeline = await self._pipeline_factory(call_id, self.settings)
            self._active_calls[call_id] = pipeline

            # Start pipeline
            await pipeline.start()

            # Run input and output handlers concurrently
            await asyncio.gather(
                self._handle_input(reader, pipeline),
                self._handle_output(writer, pipeline),
            )

        except asyncio.CancelledError:
            logger.info("call_cancelled", call_id=call_id)
        except Exception as e:
            logger.error("call_error", call_id=call_id, error=str(e))
        finally:
            # Cleanup
            await self._cleanup_call(call_id, writer)

    async def _handle_input(
        self,
        reader: asyncio.StreamReader,
        pipeline: VoiceAgentPipeline,
    ) -> None:
        """Handle incoming audio from FreeSWITCH.

        Args:
            reader: Stream reader
            pipeline: Voice agent pipeline
        """
        silence_frames = 0
        speech_active = False
        SILENCE_THRESHOLD_FRAMES = 15  # 300ms of silence at 20ms frames

        while True:
            try:
                # Read one frame of audio
                audio_bytes = await reader.read(self.frame_size)

                if not audio_bytes:
                    # Connection closed
                    break

                # Process audio
                await pipeline.process_audio_input(audio_bytes)

                # Simple silence detection for end-of-speech
                # In production, use Silero VAD for better accuracy
                from src.core.audio_utils import bytes_to_numpy, is_speech_present

                audio = bytes_to_numpy(audio_bytes)
                is_speech = is_speech_present(audio, threshold=0.01)

                if is_speech:
                    speech_active = True
                    silence_frames = 0
                elif speech_active:
                    silence_frames += 1

                    # Detected end of speech
                    if silence_frames >= SILENCE_THRESHOLD_FRAMES:
                        speech_active = False
                        silence_frames = 0

                        # Process accumulated speech
                        async for output_audio in pipeline.process_end_of_speech():
                            await pipeline.get_audio_output().put(output_audio)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "input_handler_error",
                    call_id=pipeline.call_id,
                    error=str(e),
                )
                break

    async def _handle_output(
        self,
        writer: asyncio.StreamWriter,
        pipeline: VoiceAgentPipeline,
    ) -> None:
        """Handle outgoing audio to FreeSWITCH.

        Args:
            writer: Stream writer
            pipeline: Voice agent pipeline
        """
        output_queue = pipeline.get_audio_output()

        while True:
            try:
                # Wait for audio to send
                audio_bytes = await asyncio.wait_for(
                    output_queue.get(),
                    timeout=1.0,
                )

                # Send audio frame
                writer.write(audio_bytes)
                await writer.drain()

            except asyncio.TimeoutError:
                # No audio to send, continue waiting
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "output_handler_error",
                    call_id=pipeline.call_id,
                    error=str(e),
                )
                break

    async def _cleanup_call(
        self,
        call_id: str,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Clean up resources for a call.

        Args:
            call_id: Call identifier
            writer: Stream writer to close
        """
        # Stop pipeline
        if call_id in self._active_calls:
            pipeline = self._active_calls.pop(call_id)
            await pipeline.stop()

        # Close connection
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

        logger.info("call_ended", call_id=call_id)

    @property
    def active_call_count(self) -> int:
        """Number of active calls."""
        return len(self._active_calls)

    def get_active_calls(self) -> list[str]:
        """Get list of active call IDs."""
        return list(self._active_calls.keys())
