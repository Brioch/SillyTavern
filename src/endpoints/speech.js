import { Buffer } from 'node:buffer';
import fs from 'node:fs';
import express from 'express';
import wavefile from 'wavefile';
import fetch from 'node-fetch';
import FormData from 'form-data';
import mime from 'mime-types';
import { getPipeline } from '../transformers.js';
import { forwardFetchResponse } from '../util.js';
import { readSecret, SECRET_KEYS } from './secrets.js';

export const router = express.Router();

/**
 * Gets the audio data from a base64-encoded audio file.
 * @param {string} audio Base64-encoded audio
 * @returns {Float64Array} Audio data
 */
function getWaveFile(audio) {
    const wav = new wavefile.WaveFile();
    wav.fromDataURI(audio);
    wav.toBitDepth('32f');
    wav.toSampleRate(16000);
    let audioData = wav.getSamples();
    if (Array.isArray(audioData)) {
        if (audioData.length > 1) {
            const SCALING_FACTOR = Math.sqrt(2);

            // Merge channels (into first channel to save memory)
            for (let i = 0; i < audioData[0].length; ++i) {
                audioData[0][i] = SCALING_FACTOR * (audioData[0][i] + audioData[1][i]) / 2;
            }
        }

        // Select first channel
        audioData = audioData[0];
    }

    return audioData;
}

router.post('/recognize', async (req, res) => {
    try {
        const TASK = 'automatic-speech-recognition';
        const { model, audio, lang } = req.body;
        const pipe = await getPipeline(TASK, model);
        const wav = getWaveFile(audio);
        const start = performance.now();
        const result = await pipe(wav, { language: lang || null, task: 'transcribe' });
        const end = performance.now();
        console.info(`Execution duration: ${(end - start) / 1000} seconds`);
        console.info('Transcribed audio:', result.text);

        return res.json({ text: result.text });
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

router.post('/synthesize', async (req, res) => {
    try {
        const TASK = 'text-to-speech';
        const { text, model, speaker } = req.body;
        const pipe = await getPipeline(TASK, model);
        const speaker_embeddings = speaker
            ? new Float32Array(new Uint8Array(Buffer.from(speaker.startsWith('data:') ? speaker.split(',')[1] : speaker, 'base64')).buffer)
            : null;
        const start = performance.now();
        const result = await pipe(text, { speaker_embeddings: speaker_embeddings });
        const end = performance.now();
        console.debug(`Execution duration: ${(end - start) / 1000} seconds`);

        const wav = new wavefile.WaveFile();
        wav.fromScratch(1, result.sampling_rate, '32f', result.audio);
        const buffer = wav.toBuffer();

        res.set('Content-Type', 'audio/wav');
        return res.send(Buffer.from(buffer));
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

const pollinations = express.Router();

pollinations.post('/voices', async (req, res) => {
    try {
        const model = req.body.model || 'openai-audio';

        const response = await fetch('https://gen.pollinations.ai/text/models');

        if (!response.ok) {
            throw new Error('Failed to fetch Pollinations models');
        }

        const data = await response.json();

        if (!Array.isArray(data)) {
            throw new Error('Invalid data format received from Pollinations');
        }

        const audioModelData = data.find(m => m.name === model);
        if (!audioModelData || !Array.isArray(audioModelData.voices)) {
            throw new Error('No voices found for the specified model');
        }

        const voices = audioModelData.voices;
        return res.json(voices);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

pollinations.post('/generate', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.POLLINATIONS);
        if (!key) {
            console.warn('No API key saved for Pollinations TTS.');
            return res.sendStatus(400);
        }

        const text = req.body.text;
        const model = req.body.model || 'openai-audio';
        const voice = req.body.voice || 'alloy';

        console.debug('Pollinations TTS request', { text, model, voice });

        const response = await fetch('https://gen.pollinations.ai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${key}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: model,
                stream: false,
                modalities: ['text', 'audio'],
                seed: Math.floor(Math.random() * Math.pow(2, 32)),
                audio: {
                    format: 'mp3',
                    voice: voice,
                },
                messages: [{
                    role: 'user',
                    content: text,
                }],
            }),
        });

        if (!response.ok) {
            const text = await response.text();
            throw new Error(`Failed to generate audio from Pollinations: ${text}`);
        }

        /** @type {any} */
        const data = await response.json();
        const audioData = data?.choices?.[0]?.message?.audio?.data;

        if (!audioData) {
            console.warn('Pollinations TTS audio data is missing from the response');
            return res.sendStatus(500);
        }

        res.set('Content-Type', 'audio/mpeg');
        return res.send(Buffer.from(audioData, 'base64'));
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

router.use('/pollinations', pollinations);

const elevenlabs = express.Router();

elevenlabs.post('/voices', async (req, res) => {
    try {
        const apiKey = readSecret(req.user.directories, SECRET_KEYS.ELEVENLABS);
        if (!apiKey) {
            console.warn('ElevenLabs API key not found');
            return res.sendStatus(400);
        }

        const response = await fetch('https://api.elevenlabs.io/v1/voices', {
            headers: {
                'xi-api-key': apiKey,
            },
        });

        if (!response.ok) {
            const text = await response.text();
            console.warn(`ElevenLabs voices fetch failed: HTTP ${response.status} - ${text}`);
            return res.sendStatus(500);
        }

        const responseJson = await response.json();
        return res.json(responseJson);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

elevenlabs.post('/voice-settings', async (req, res) => {
    try {
        const apiKey = readSecret(req.user.directories, SECRET_KEYS.ELEVENLABS);
        if (!apiKey) {
            console.warn('ElevenLabs API key not found');
            return res.sendStatus(400);
        }

        const response = await fetch('https://api.elevenlabs.io/v1/voices/settings/default', {
            headers: {
                'xi-api-key': apiKey,
            },
        });

        if (!response.ok) {
            const text = await response.text();
            console.warn(`ElevenLabs voice settings fetch failed: HTTP ${response.status} - ${text}`);
            return res.sendStatus(500);
        }
        const responseJson = await response.json();
        return res.json(responseJson);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

elevenlabs.post('/synthesize', async (req, res) => {
    try {
        const apiKey = readSecret(req.user.directories, SECRET_KEYS.ELEVENLABS);
        if (!apiKey) {
            console.warn('ElevenLabs API key not found');
            return res.sendStatus(400);
        }

        const { voiceId, request } = req.body;

        if (!voiceId || !request) {
            console.warn('ElevenLabs synthesis request missing voiceId or request body');
            return res.sendStatus(400);
        }

        console.debug('ElevenLabs TTS request:', request);

        const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
            method: 'POST',
            headers: {
                'xi-api-key': apiKey,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const text = await response.text();
            console.warn(`ElevenLabs synthesis failed: HTTP ${response.status} - ${text}`);
            return res.sendStatus(500);
        }

        res.set('Content-Type', 'audio/mpeg');
        await forwardFetchResponse(response, res);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

elevenlabs.post('/history', async (req, res) => {
    try {
        const apiKey = readSecret(req.user.directories, SECRET_KEYS.ELEVENLABS);
        if (!apiKey) {
            console.warn('ElevenLabs API key not found');
            return res.sendStatus(400);
        }

        const response = await fetch('https://api.elevenlabs.io/v1/history', {
            headers: {
                'xi-api-key': apiKey,
            },
        });

        if (!response.ok) {
            const text = await response.text();
            console.warn(`ElevenLabs history fetch failed: HTTP ${response.status} - ${text}`);
            return res.sendStatus(500);
        }

        const responseJson = await response.json();
        return res.json(responseJson);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

elevenlabs.post('/history-audio', async (req, res) => {
    try {
        const apiKey = readSecret(req.user.directories, SECRET_KEYS.ELEVENLABS);
        if (!apiKey) {
            console.warn('ElevenLabs API key not found');
            return res.sendStatus(400);
        }

        const { historyItemId } = req.body;
        if (!historyItemId) {
            console.warn('ElevenLabs history audio request missing historyItemId');
            return res.sendStatus(400);
        }

        console.debug('ElevenLabs history audio request for ID:', historyItemId);

        const response = await fetch(`https://api.elevenlabs.io/v1/history/${historyItemId}/audio`, {
            headers: {
                'xi-api-key': apiKey,
            },
        });

        if (!response.ok) {
            const text = await response.text();
            console.warn(`ElevenLabs history audio fetch failed: HTTP ${response.status} - ${text}`);
            return res.sendStatus(500);
        }

        res.set('Content-Type', 'audio/mpeg');
        await forwardFetchResponse(response, res);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

elevenlabs.post('/voices/add', async (req, res) => {
    try {
        const apiKey = readSecret(req.user.directories, SECRET_KEYS.ELEVENLABS);
        if (!apiKey) {
            console.warn('ElevenLabs API key not found');
            return res.sendStatus(400);
        }

        const { name, description, labels, files } = req.body;

        const formData = new FormData();
        formData.append('name', name || 'Custom Voice');
        formData.append('description', description || 'Uploaded via SillyTavern');
        formData.append('labels', labels || '');

        for (const fileData of (files || [])) {
            const [mimeType, base64Data] = /^data:(.+);base64,(.+)$/.exec(fileData)?.slice(1) || [];
            if (!mimeType || !base64Data) {
                console.warn('Invalid audio file data provided for ElevenLabs voice upload');
                continue;
            }
            const buffer = Buffer.from(base64Data, 'base64');
            formData.append('files', buffer, {
                filename: `audio.${mime.extension(mimeType) || 'wav'}`,
                contentType: mimeType,
            });
        }

        console.debug('ElevenLabs voice upload request:', { name, description, labels, files: files?.length || 0 });

        const response = await fetch('https://api.elevenlabs.io/v1/voices/add', {
            method: 'POST',
            headers: {
                'xi-api-key': apiKey,
            },
            body: formData,
        });

        if (!response.ok) {
            const text = await response.text();
            console.warn(`ElevenLabs voice upload failed: HTTP ${response.status} - ${text}`);
            return res.sendStatus(500);
        }

        const responseJson = await response.json();
        return res.json(responseJson);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

elevenlabs.post('/recognize', async (req, res) => {
    try {
        const apiKey = readSecret(req.user.directories, SECRET_KEYS.ELEVENLABS);
        if (!apiKey) {
            console.warn('ElevenLabs API key not found');
            return res.sendStatus(400);
        }

        if (!req.file) {
            console.warn('No audio file found');
            return res.sendStatus(400);
        }

        console.info('Processing audio file with ElevenLabs', req.file.path);
        const formData = new FormData();
        formData.append('file', fs.createReadStream(req.file.path), { filename: 'audio.wav', contentType: 'audio/wav' });
        formData.append('model_id', req.body.model);

        const response = await fetch('https://api.elevenlabs.io/v1/speech-to-text', {
            method: 'POST',
            headers: {
                'xi-api-key': apiKey,
            },
            body: formData,
        });

        if (!response.ok) {
            const text = await response.text();
            console.warn(`ElevenLabs speech recognition failed: HTTP ${response.status} - ${text}`);
            return res.sendStatus(500);
        }

        fs.unlinkSync(req.file.path);
        const responseJson = await response.json();
        console.debug('ElevenLabs speech recognition response:', responseJson);
        return res.json(responseJson);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

router.use('/elevenlabs', elevenlabs);

const audiocpp = express.Router();

/**
 * Normalizes a user-supplied audio.cpp server address into a base URL without the API prefix.
 * Tolerates a bare host:port, a trailing slash, or a full endpoint path being pasted in.
 * @param {string} endpoint Address as entered in the provider settings
 * @returns {string|null} Base URL without a trailing slash, or null if unusable
 */
function getAudioCppBaseUrl(endpoint) {
    if (!endpoint || typeof endpoint !== 'string') {
        return null;
    }

    const value = endpoint.trim();

    if (!value) {
        return null;
    }

    try {
        const url = new URL(/^https?:\/\//i.test(value) ? value : `http://${value}`);
        url.search = '';
        url.hash = '';

        const path = url.pathname.replace(/\/+$/, '');
        const apiIndex = path.toLowerCase().indexOf('/v1');
        url.pathname = apiIndex >= 0 ? path.slice(0, apiIndex) : path;

        return url.toString().replace(/\/+$/, '');
    } catch {
        return null;
    }
}

audiocpp.post('/models', async (req, res) => {
    try {
        const base = getAudioCppBaseUrl(req.body.provider_endpoint);

        if (!base) {
            console.warn('No audio.cpp provider endpoint provided');
            return res.sendStatus(400);
        }

        const response = await fetch(`${base}/v1/models`);

        if (!response.ok) {
            const text = await response.text();
            console.warn(`audio.cpp model list fetch failed: HTTP ${response.status} - ${text}`);
            return res.sendStatus(500);
        }

        const responseJson = await response.json();
        return res.json(responseJson);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

audiocpp.post('/voices', async (req, res) => {
    try {
        const base = getAudioCppBaseUrl(req.body.provider_endpoint);

        if (!base) {
            console.warn('No audio.cpp provider endpoint provided');
            return res.sendStatus(400);
        }

        const url = new URL(`${base}/v1/audio/voices`);

        // Omitting the model is only valid when the server has exactly one model configured.
        if (req.body.model) {
            url.searchParams.set('model', String(req.body.model));
        }

        const response = await fetch(url);

        if (!response.ok) {
            const text = await response.text();
            console.warn(`audio.cpp voice list fetch failed: HTTP ${response.status} - ${text}`);
            return res.sendStatus(500);
        }

        const responseJson = await response.json();
        return res.json(responseJson);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

audiocpp.post('/generate', async (req, res) => {
    try {
        const base = getAudioCppBaseUrl(req.body.provider_endpoint);

        if (!base) {
            console.warn('No audio.cpp provider endpoint provided');
            return res.sendStatus(400);
        }

        const request = req.body.request;

        if (!request || typeof request !== 'object') {
            console.warn('audio.cpp synthesis request missing request body');
            return res.sendStatus(400);
        }

        const streaming = request.stream_format === 'sse';
        console.debug('audio.cpp TTS request:', request);

        const response = await fetch(`${base}/v1/audio/speech`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(streaming ? { 'Accept': 'text/event-stream' } : {}),
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const text = await response.text();
            console.warn(`audio.cpp synthesis failed: HTTP ${response.status} - ${text}`);
            return res.status(500).send(text);
        }

        // forwardFetchResponse only pipes the body, so the content type has to be set here.
        if (streaming) {
            // Compression is applied globally and text/event-stream is compressible;
            // no-transform makes the middleware skip it so deltas aren't buffered.
            res.set('Content-Type', 'text/event-stream');
            res.set('Cache-Control', 'no-cache, no-transform');
            res.set('Connection', 'keep-alive');
        } else {
            res.set('Content-Type', response.headers.get('content-type') || 'audio/wav');
        }

        await forwardFetchResponse(response, res);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

router.use('/audiocpp', audiocpp);
