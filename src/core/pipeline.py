"""Voice agent pipeline using Pipecat."""

import asyncio
from typing import AsyncIterator

import numpy as np

from src.config.settings import Settings
from src.core.audio_utils import AudioBuffer, bytes_to_numpy, numpy_to_bytes, resample_audio
from src.core.state_machine import CallState, CallStateMachine
from src.services.llm.service import ConversationContext, LLMService, Message
from src.services.stt.service import TurkishSTTService
from src.services.tts.service import ChatterboxTTSService
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VoiceAgentPipeline:
    """Voice agent pipeline for real-time conversation.

    Orchestrates the flow:
    Audio Input -> VAD -> STT -> LLM -> TTS -> Audio Output

    Handles barge-in (user interruption) and state management.
    """

    def __init__(
        self,
        call_id: str,
        settings: Settings,
        stt_service: TurkishSTTService,
        tts_service: ChatterboxTTSService,
        llm_service: LLMService,
    ):
        """Initialize the voice agent pipeline.

        Args:
            call_id: Unique identifier for this call
            settings: Application settings
            stt_service: Speech-to-text service
            tts_service: Text-to-speech service
            llm_service: Language model service
        """
        self.call_id = call_id
        self.settings = settings
        self.stt = stt_service
        self.tts = tts_service
        self.llm = llm_service

        # State management
        self.state_machine = CallStateMachine(call_id)

        # Audio buffer for accumulating speech
        self.audio_buffer = AudioBuffer(
            sample_rate=settings.model_sample_rate,
            min_duration_ms=500,
            max_duration_ms=30000,
        )

        # Conversation context
        self.context = ConversationContext(
            system_prompt=settings.system_prompt,
            max_history=10,
        )

        # Control flags
        self._is_speaking = False
        self._cancel_speech = asyncio.Event()
        self._audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()

        logger.info("pipeline_initialized", call_id=call_id)

    async def start(self) -> None:
        """Start the pipeline (transition to LISTENING state)."""
        self.state_machine.start_call()
        logger.info("pipeline_started", call_id=self.call_id)

    async def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self._cancel_speech.set()
        self.state_machine.end_call()
        logger.info("pipeline_stopped", call_id=self.call_id)

    async def process_audio_input(self, audio_bytes: bytes) -> None:
        """Process incoming audio from the user.

        Args:
            audio_bytes: Raw PCM audio bytes (8kHz, 16-bit, mono)
        """
        if self.state_machine.state == CallState.IDLE:
            return

        # Convert to numpy and resample to model sample rate
        audio = bytes_to_numpy(audio_bytes)
        audio = resample_audio(
            audio,
            self.settings.telephony_sample_rate,
            self.settings.model_sample_rate,
        )

        # Add to buffer
        self.audio_buffer.append(audio)

        # Check for barge-in (user speaking while AI is speaking)
        if self.state_machine.can_interrupt():
            # Simple energy-based detection for barge-in
            from src.core.audio_utils import is_speech_present

            if is_speech_present(audio, threshold=0.02):
                await self._handle_barge_in()

    async def _handle_barge_in(self) -> None:
        """Handle user interrupting AI speech."""
        logger.info("barge_in_detected", call_id=self.call_id)

        # Cancel current TTS output
        self._cancel_speech.set()
        self._is_speaking = False

        # Clear output queue
        while not self._audio_output_queue.empty():
            try:
                self._audio_output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Transition state
        self.state_machine.user_started_speaking()

    async def process_end_of_speech(self) -> AsyncIterator[bytes]:
        """Process accumulated audio when user stops speaking.

        This is called when VAD detects end of speech.

        Yields:
            Audio chunks for TTS output
        """
        if not self.audio_buffer.has_enough():
            logger.debug("audio_buffer_too_short", call_id=self.call_id)
            return

        # Get buffered audio and clear
        audio = self.audio_buffer.flush()

        # Transition to processing
        self.state_machine.user_stopped_speaking()

        try:
            # 1. Speech-to-Text
            logger.debug("starting_stt", call_id=self.call_id)
            transcription = self.stt.transcribe(audio)

            if not transcription.text:
                logger.debug("empty_transcription", call_id=self.call_id)
                self.state_machine.transition(CallState.LISTENING)
                return

            logger.info(
                "user_said",
                call_id=self.call_id,
                text=transcription.text,
                confidence=transcription.confidence,
            )

            # 2. LLM Generation
            logger.debug("starting_llm", call_id=self.call_id)
            response = await self.llm.generate_with_context(
                context=self.context,
                user_input=transcription.text,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
            )

            if not response:
                logger.warning("empty_llm_response", call_id=self.call_id)
                self.state_machine.transition(CallState.LISTENING)
                return

            logger.info("assistant_response", call_id=self.call_id, text=response)

            # 3. Text-to-Speech
            self.state_machine.ai_started_speaking()
            self._is_speaking = True
            self._cancel_speech.clear()

            logger.debug("starting_tts", call_id=self.call_id)

            # Synthesize to PCM at telephony sample rate
            # Run in thread pool since Chatterbox TTS is synchronous
            audio_pcm = await asyncio.get_event_loop().run_in_executor(
                None,
                self.tts.synthesize_to_pcm,
                response,
                self.settings.telephony_sample_rate,
            )

            if len(audio_pcm) == 0:
                self.state_machine.ai_finished_speaking()
                self._is_speaking = False
                return

            # Stream audio in chunks
            chunk_samples = int(
                self.settings.telephony_sample_rate * self.settings.frame_duration_ms / 1000
            )

            for i in range(0, len(audio_pcm), chunk_samples):
                # Check for cancellation (barge-in)
                if self._cancel_speech.is_set():
                    logger.debug("speech_cancelled", call_id=self.call_id)
                    break

                chunk = audio_pcm[i : i + chunk_samples]
                yield numpy_to_bytes(chunk)

                # Small delay to simulate real-time streaming
                await asyncio.sleep(self.settings.frame_duration_ms / 1000)

            # Finished speaking
            self.state_machine.ai_finished_speaking()
            self._is_speaking = False

        except Exception as e:
            logger.error("pipeline_error", call_id=self.call_id, error=str(e))
            self.state_machine.transition(CallState.LISTENING)
            self._is_speaking = False

    def get_audio_output(self) -> asyncio.Queue[bytes]:
        """Get the audio output queue.

        Returns:
            Queue containing audio bytes to send to user
        """
        return self._audio_output_queue

    @property
    def is_speaking(self) -> bool:
        """Check if AI is currently speaking."""
        return self._is_speaking

    @property
    def current_state(self) -> CallState:
        """Get current call state."""
        return self.state_machine.state


async def create_voice_agent(
    call_id: str,
    settings: Settings,
) -> VoiceAgentPipeline:
    """Factory function to create a voice agent pipeline.

    Args:
        call_id: Unique identifier for the call
        settings: Application settings

    Returns:
        Configured VoiceAgentPipeline instance
    """
    # Initialize services
    stt_service = TurkishSTTService(
        model_path=settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
    )

    tts_service = ChatterboxTTSService(
        device=settings.tts_device,
        language=settings.tts_language,
        reference_audio=settings.tts_reference_audio,
    )

    llm_service = LLMService(
        base_url=settings.vllm_base_url,
        model=settings.llm_model,
    )

    # Create pipeline
    pipeline = VoiceAgentPipeline(
        call_id=call_id,
        settings=settings,
        stt_service=stt_service,
        tts_service=tts_service,
        llm_service=llm_service,
    )

    return pipeline
