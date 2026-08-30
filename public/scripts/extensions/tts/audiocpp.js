import { getRequestHeaders } from '../../../script.js';
import { delay } from '../../utils.js';
import { getPreviewString, saveTtsProviderSettings } from './index.js';

export { AudioCppTtsProvider };

const AUDIOCPP_API = '/api/speech/audiocpp';

/** Scheduling lead, in seconds, that keeps the first streamed chunk from starting in the past. */
const STREAM_LEAD_TIME = 0.1;

/**
 * Reads an SSE response body and yields the parsed payload of every data event.
 * @param {Response} response Streaming response from the audio.cpp proxy
 * @returns {AsyncGenerator<object>} Parsed SSE payloads
 */
async function* readSseEvents(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
        while (true) {
            const { done, value } = await reader.read();

            if (done) {
                break;
            }

            buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');

            let separatorIndex;
            while ((separatorIndex = buffer.indexOf('\n\n')) !== -1) {
                const event = buffer.slice(0, separatorIndex);
                buffer = buffer.slice(separatorIndex + 2);

                const data = event
                    .split('\n')
                    .filter(line => line.startsWith('data:'))
                    .map(line => line.slice('data:'.length).trim())
                    .join('\n');

                if (!data || data === '[DONE]') {
                    continue;
                }

                try {
                    yield JSON.parse(data);
                } catch {
                    console.warn('audio.cpp TTS: could not parse SSE payload', data);
                }
            }
        }
    } finally {
        // Runs on an early break too, so the response body is never left locked.
        await reader.cancel().catch(() => { });
        reader.releaseLock();
    }
}

/**
 * Checks whether a voice map entry names a reference WAV rather than a voice preset.
 * @param {string} voiceId Voice map entry
 * @returns {boolean} True if the entry looks like a path to a reference file
 */
function isVoiceReference(voiceId) {
    return typeof voiceId === 'string' && (/\.wav$/i.test(voiceId) || voiceId.includes('/') || voiceId.includes('\\'));
}

/**
 * Schedules signed 16-bit little-endian PCM chunks for gapless playback via the Web Audio API.
 * Used for streamed generations, where the extension's audio element is bypassed entirely.
 */
class AudioCppStreamPlayer {
    /** @type {AudioContext|null} */
    context = null;
    /** @type {Set<AudioBufferSourceNode>} */
    sources = new Set();
    /** Playback position, in context time, where the next chunk should start. */
    cursor = 0;

    /**
     * Creates (or recreates) the audio context for a given sample rate.
     * @param {number} sampleRate Sample rate of the incoming PCM stream
     * @returns {Promise<AudioContext>} Ready audio context
     */
    async prepare(sampleRate) {
        if (this.context && this.context.sampleRate !== sampleRate) {
            await this.stop();
        }

        if (!this.context) {
            this.context = new AudioContext({ sampleRate: sampleRate });
            this.cursor = 0;
        }

        if (this.context.state === 'suspended') {
            await this.context.resume();
        }

        return this.context;
    }

    /**
     * Queues a PCM chunk right after whatever is already scheduled.
     * @param {Uint8Array} bytes Signed 16-bit little-endian mono PCM
     * @param {number} sampleRate Sample rate of the chunk
     */
    enqueue(bytes, sampleRate) {
        const context = this.context;

        if (!context) {
            return;
        }

        const sampleCount = Math.floor(bytes.byteLength / 2);

        if (sampleCount === 0) {
            return;
        }

        const view = new DataView(bytes.buffer, bytes.byteOffset, sampleCount * 2);
        const buffer = context.createBuffer(1, sampleCount, sampleRate);
        const channel = buffer.getChannelData(0);

        for (let i = 0; i < sampleCount; i++) {
            channel[i] = view.getInt16(i * 2, true) / 32768;
        }

        const source = context.createBufferSource();
        source.buffer = buffer;
        source.connect(context.destination);
        source.onended = () => {
            this.sources.delete(source);
            source.disconnect();
        };

        const startAt = Math.max(this.cursor, context.currentTime + STREAM_LEAD_TIME);
        source.start(startAt);
        this.cursor = startAt + buffer.duration;
        this.sources.add(source);
    }

    /**
     * Plays an audio.cpp SSE stream and resolves once the last scheduled chunk has finished.
     * @param {Response} response Streaming response from the audio.cpp proxy
     * @param {number} sampleRate Sample rate of the PCM payloads
     * @returns {Promise<void>}
     */
    async play(response, sampleRate) {
        await this.prepare(sampleRate);

        for await (const event of readSseEvents(response)) {
            if (event.type === 'speech.audio.done') {
                break;
            }

            if (event.type && event.type !== 'speech.audio.delta') {
                continue;
            }

            // The payload field is not pinned down by the docs, so accept the plausible spellings.
            const payload = event.audio ?? event.delta ?? event.data;

            if (typeof payload !== 'string' || !payload) {
                continue;
            }

            const binary = atob(payload);
            const bytes = new Uint8Array(binary.length);

            for (let i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
            }

            this.enqueue(bytes, sampleRate);
        }

        const remaining = this.context ? this.cursor - this.context.currentTime : 0;

        if (remaining > 0) {
            await delay(Math.ceil(remaining * 1000));
        }
    }

    /**
     * Stops playback and releases the audio context.
     * @returns {Promise<void>}
     */
    async stop() {
        for (const source of this.sources) {
            try {
                source.onended = null;
                source.stop();
                source.disconnect();
            } catch {
                // Already stopped.
            }
        }

        this.sources.clear();
        this.cursor = 0;

        if (this.context) {
            const context = this.context;
            this.context = null;
            await context.close().catch(() => { });
        }
    }
}

class AudioCppTtsProvider {
    settings;
    voices = [];
    separator = ' . ';

    audioElement = document.createElement('audio');
    player = new AudioCppStreamPlayer();

    defaultSettings = {
        voiceMap: {},
        provider_endpoint: 'http://127.0.0.1:8080',
        model: '',
        available_voices: [],
        voice_refs: '',
        reference_text: '',
        max_tokens: 0,
        seed: -1,
        extra_options: '',
        streaming: false,
        stream_sample_rate: 24000,
    };

    get settingsHtml() {
        return `
        <label for="audiocpp_endpoint">Provider Endpoint:</label>
        <input id="audiocpp_endpoint" type="text" class="text_pole" maxlength="500" value="${this.defaultSettings.provider_endpoint}"/>
        <small>
            Base URL of an <a target="_blank" href="https://github.com/0xShug0/audio.cpp">audio.cpp</a> server
            (<code>audiocpp_server</code>). Requests are proxied by SillyTavern, so the server does
            <b>not</b> need <code>--cors-origins</code>. If it sits behind an authenticating reverse proxy, add the
            headers via <code>requestOverrides</code> in <code>config.yaml</code>.
        </small>

        <label for="audiocpp_model">Model:</label>
        <select id="audiocpp_model"></select>
        <small>Model IDs come from the server's <code>/v1/models</code>. Use the refresh button above to reload models and voices.</small>

        <label for="audiocpp_voice_refs">Voice references:</label>
        <textarea id="audiocpp_voice_refs" class="text_pole textarea_compact" rows="4" placeholder="voices/alice.wav | Hello, this is Alice speaking.&#10;voices/bob.wav | And this is Bob."></textarea>
        <small>
            Server-side WAV paths for voice cloning, one per line. They are added to the voice list, so each character
            can be mapped to a different clone. Add that reference's own transcript after a <code>|</code>.
        </small>

        <label class="checkbox_label alignItemsCenter flexGap5" for="audiocpp_streaming">
            <input id="audiocpp_streaming" type="checkbox"/>
            <span>Streaming</span>
        </label>
        <small>
            Plays audio as it is generated, using the server's SSE stream. Streamed audio bypasses the extension's
            audio element, so the playback rate slider, the play/pause button and RVC do not apply to it.
        </small>

        <div id="audiocpp_streaming_block">
            <label for="audiocpp_stream_sample_rate">Stream sample rate:</label>
            <input id="audiocpp_stream_sample_rate" type="number" class="text_pole" min="8000" max="48000" step="1000"/>
            <small>The SSE stream carries raw PCM without a header, so the model's output sample rate must be set here.</small>
        </div>

        <div class="inline-drawer marginTop10">
            <div class="inline-drawer-toggle inline-drawer-header">
                <b>Advanced</b>
                <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
            </div>
            <div class="inline-drawer-content">
                <label for="audiocpp_max_tokens">Max tokens (0 = server default):</label>
                <input id="audiocpp_max_tokens" type="number" class="text_pole" min="0" step="1"/>

                <label for="audiocpp_seed">Seed (-1 = random):</label>
                <input id="audiocpp_seed" type="text" class="text_pole" maxlength="20"/>

                <label for="audiocpp_reference_text">Fallback reference text:</label>
                <input id="audiocpp_reference_text" type="text" class="text_pole" maxlength="1000"/>
                <small>Used only for voices whose reference line carries no transcript of its own.</small>

                <label for="audiocpp_extra_options">Extra options (JSON):</label>
                <textarea id="audiocpp_extra_options" class="text_pole textarea_compact" rows="3" placeholder="{ }"></textarea>
                <small>Model-specific options, passed through as the request's <code>options</code> object.</small>
            </div>
        </div>`;
    }

    async loadSettings(settings) {
        // Populate Provider UI given input settings
        if (Object.keys(settings).length == 0) {
            console.info('Using default TTS Provider settings');
        }

        // Only accept keys defined in defaultSettings
        this.settings = this.defaultSettings;

        for (const key in settings) {
            if (key in this.settings) {
                this.settings[key] = settings[key];
            } else {
                throw `Invalid setting passed to TTS Provider: ${key}`;
            }
        }

        $('#audiocpp_endpoint').val(this.settings.provider_endpoint);
        $('#audiocpp_endpoint').on('input', () => { this.onSettingsChange(); });

        $('#audiocpp_model').on('change', () => { this.onSettingsChange(); });

        $('#audiocpp_voice_refs').val(this.settings.voice_refs);
        $('#audiocpp_voice_refs').on('input', () => { this.onSettingsChange(); });

        $('#audiocpp_reference_text').val(this.settings.reference_text);
        $('#audiocpp_reference_text').on('input', () => { this.onSettingsChange(); });

        $('#audiocpp_streaming').prop('checked', this.settings.streaming);
        $('#audiocpp_streaming').on('change', () => { this.onSettingsChange(); });

        $('#audiocpp_stream_sample_rate').val(this.settings.stream_sample_rate);
        $('#audiocpp_stream_sample_rate').on('input', () => { this.onSettingsChange(); });

        $('#audiocpp_max_tokens').val(this.settings.max_tokens);
        $('#audiocpp_max_tokens').on('input', () => { this.onSettingsChange(); });

        $('#audiocpp_seed').val(this.settings.seed);
        $('#audiocpp_seed').on('input', () => { this.onSettingsChange(); });

        $('#audiocpp_extra_options').val(this.settings.extra_options);
        $('#audiocpp_extra_options').on('input', () => { this.onSettingsChange(); });

        $('#audiocpp_streaming_block').toggle(this.settings.streaming);

        // Discovery failures are reported by checkReady(), so don't let them break provider loading.
        try {
            await this.refreshModels();
        } catch (error) {
            console.warn('audio.cpp TTS: model list unavailable', error);
            this.renderModelOptions([]);
        }

        try {
            this.voices = await this.fetchTtsVoiceObjects();
        } catch (error) {
            console.warn('audio.cpp TTS: voice list unavailable', error);
        }

        console.debug('audio.cpp TTS: Settings loaded');
    }

    onSettingsChange() {
        this.settings.provider_endpoint = String($('#audiocpp_endpoint').val());
        this.settings.model = String($('#audiocpp_model').val() ?? '');
        this.settings.voice_refs = String($('#audiocpp_voice_refs').val());
        this.settings.reference_text = String($('#audiocpp_reference_text').val());
        this.settings.streaming = $('#audiocpp_streaming').is(':checked');
        this.settings.stream_sample_rate = Number($('#audiocpp_stream_sample_rate').val()) || this.defaultSettings.stream_sample_rate;
        this.settings.max_tokens = Number($('#audiocpp_max_tokens').val()) || 0;
        this.settings.seed = String($('#audiocpp_seed').val()).trim() || '-1';
        this.settings.extra_options = String($('#audiocpp_extra_options').val());

        $('#audiocpp_streaming_block').toggle(this.settings.streaming);

        saveTtsProviderSettings();
    }

    // Perform a readiness check by listing the server's models and voices.
    async checkReady() {
        await this.refreshModels();
        this.voices = await this.fetchTtsVoiceObjects();
    }

    async onRefreshClick() {
        await this.refreshModels();
        this.voices = await this.fetchTtsVoiceObjects();
        console.info('audio.cpp TTS voices refreshed');
    }

    dispose() {
        this.player.stop();
    }

    //#################//
    //  TTS Interfaces //
    //#################//

    async getVoice(voiceName) {
        if (this.voices.length == 0) {
            this.voices = await this.fetchTtsVoiceObjects();
        }

        const match = this.voices.filter(
            audioCppVoice => audioCppVoice.name == voiceName,
        )[0];

        if (!match) {
            throw `TTS Voice name ${voiceName} not found`;
        }

        return match;
    }

    async generateTts(text, voiceId) {
        const streaming = this.settings.streaming;
        const response = await this.fetchTtsGeneration(text, voiceId, streaming);

        if (!streaming) {
            return response;
        }

        await this.player.play(response, this.settings.stream_sample_rate);

        // The audio has already been played, so hand back an empty async iterable:
        // the extension iterates it, queues nothing, and leaves its audio queue untouched.
        return (async function* () { })();
    }

    async fetchTtsVoiceObjects() {
        const voices = [];

        try {
            const response = await fetch(`${AUDIOCPP_API}/voices`, {
                method: 'POST',
                headers: getRequestHeaders(),
                body: JSON.stringify({
                    provider_endpoint: this.settings.provider_endpoint,
                    model: this.settings.model,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }

            const responseJson = await response.json();
            const names = Array.isArray(responseJson?.voices) ? responseJson.voices : [];
            this.settings.available_voices = names.map(voice => String(voice?.id ?? voice?.name ?? voice));
        } catch (error) {
            console.warn('audio.cpp TTS: voice discovery failed, using the cached list', error);
        }

        for (const name of this.settings.available_voices) {
            voices.push({ name: name, voice_id: name, lang: 'en-US' });
        }

        for (const reference of this.getVoiceReferences()) {
            voices.push({ name: reference.path, voice_id: reference.path, lang: 'en-US' });
        }

        return voices;
    }

    async previewTtsVoice(voiceId) {
        this.audioElement.pause();
        this.audioElement.currentTime = 0;

        const text = getPreviewString('en-US');
        // Previews always go through the audio element, so never stream them.
        const response = await this.fetchTtsGeneration(text, voiceId, false);

        const audio = await response.blob();
        const url = URL.createObjectURL(audio);
        this.audioElement.src = url;
        this.audioElement.play();
        this.audioElement.onended = () => URL.revokeObjectURL(url);
    }

    //###########//
    // API CALLS //
    //###########//

    /**
     * Parses the configured voice references, one per line, as `path | reference text`.
     * A line with no separator is read as the legacy comma separated list of bare paths.
     * @returns {{path: string, text: string}[]} Reference paths and their transcripts
     */
    getVoiceReferences() {
        const references = [];

        for (const line of String(this.settings.voice_refs ?? '').split('\n')) {
            const trimmed = line.trim();

            if (!trimmed) {
                continue;
            }

            const separatorIndex = trimmed.indexOf('|');

            if (separatorIndex === -1) {
                for (const path of trimmed.split(',').map(entry => entry.trim()).filter(entry => entry.length > 0)) {
                    references.push({ path: path, text: '' });
                }

                continue;
            }

            const path = trimmed.slice(0, separatorIndex).trim();

            if (path) {
                references.push({ path: path, text: trimmed.slice(separatorIndex + 1).trim() });
            }
        }

        return references;
    }

    /**
     * Fetches the server's model list and rebuilds the model dropdown.
     * @returns {Promise<string[]>} Available model IDs
     */
    async refreshModels() {
        const response = await fetch(`${AUDIOCPP_API}/models`, {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({ provider_endpoint: this.settings.provider_endpoint }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }

        const responseJson = await response.json();
        const entries = Array.isArray(responseJson?.data)
            ? responseJson.data
            : (Array.isArray(responseJson) ? responseJson : []);
        const models = entries
            .map(model => String(model?.id ?? model))
            .filter(model => model.length > 0);

        this.renderModelOptions(models);
        return models;
    }

    /**
     * Rebuilds the model dropdown, keeping the saved selection even if the server didn't list it.
     * @param {string[]} models Available model IDs
     */
    renderModelOptions(models) {
        const selected = this.settings.model;
        const options = [...models];

        if (selected && !options.includes(selected)) {
            options.unshift(selected);
        }

        const select = $('#audiocpp_model');
        select.empty();

        if (options.length === 0) {
            select.append($('<option />').val('').text('- No models found -'));
            return;
        }

        for (const model of options) {
            select.append($('<option />').val(model).text(model));
        }

        // Adopt the first model when nothing was ever selected.
        if (!selected) {
            this.settings.model = options[0];
        }

        select.val(this.settings.model);
    }

    /**
     * Builds and sends a synthesis request through the SillyTavern proxy.
     * @param {string} inputText Text to synthesize
     * @param {string} voiceId Voice name or server-side reference path
     * @param {boolean} streaming Whether to request an SSE stream
     * @returns {Promise<Response>} Audio or SSE response
     */
    async fetchTtsGeneration(inputText, voiceId, streaming) {
        console.info(`Generating new TTS for voice_id ${voiceId}`);

        const request = {
            model: this.settings.model,
            input: inputText,
            response_format: streaming ? 'pcm' : 'wav',
        };

        const reference = this.getVoiceReferences().find(item => item.path === voiceId);

        // A reference path selects a cloned voice; anything else is a preset or cached voice ID.
        if (reference || isVoiceReference(voiceId)) {
            request.voice_ref = { type: 'path', path: voiceId };
        } else if (voiceId) {
            request.voice = voiceId;
        }

        // Each reference carries its own transcript; the global field only fills the gaps.
        const referenceText = reference?.text || this.settings.reference_text;

        if (referenceText) {
            request.reference_text = referenceText;
        }

        if (this.settings.max_tokens > 0) {
            request.max_tokens = this.settings.max_tokens;
        }

        const seed = String(this.settings.seed).trim();

        // Seeds are sent as strings so that the full uint64 range survives.
        if (seed && seed !== '-1') {
            request.seed = seed;
        }

        if (streaming) {
            request.stream_format = 'sse';
        }

        const extraOptions = String(this.settings.extra_options ?? '').trim();

        if (extraOptions) {
            try {
                request.options = JSON.parse(extraOptions);
            } catch {
                toastr.error('Extra options must be valid JSON.', 'audio.cpp TTS');
                throw new Error('audio.cpp TTS: extra options are not valid JSON');
            }
        }

        const response = await fetch(`${AUDIOCPP_API}/generate`, {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({
                provider_endpoint: this.settings.provider_endpoint,
                request: request,
            }),
        });

        if (!response.ok) {
            toastr.error(response.statusText, 'TTS Generation Failed');
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }

        return response;
    }
}
