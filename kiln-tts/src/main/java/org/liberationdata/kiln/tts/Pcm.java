package org.liberationdata.kiln.tts;

import org.liberationdata.kiln.Audio;

final class Pcm {
    private Pcm() {}

    /** Interleaved PCM of 1 or 2 channels as stereo {@link Audio} (mono plays on both sides). */
    static Audio toAudio(float[] pcm, int channels, int sampleRate) {
        if (channels <= 1) return new Audio(pcm, pcm.clone(), sampleRate);
        int n = pcm.length / channels;
        float[] l = new float[n], r = new float[n];
        for (int i = 0; i < n; i++) { l[i] = pcm[i * channels]; r[i] = pcm[i * channels + 1]; }
        return new Audio(l, r, sampleRate);
    }
}
