package org.liberationdata.kiln.hub;

import org.junit.jupiter.api.Test;

import java.net.URI;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HubHttpTest {
    private static final URI HUB = URI.create("https://huggingface.co");

    @Test void theTokenGoesToTheHubItself() {
        assertTrue(HubHttp.sendsToken(URI.create("https://huggingface.co/org/repo/resolve/main/a.gguf"), HUB));
        assertTrue(HubHttp.sendsToken(URI.create("HTTPS://HuggingFace.co/x"), HUB));
    }

    @Test void theTokenIsNotSentInTheClearOnARedirectToHttp() {
        assertFalse(HubHttp.sendsToken(URI.create("http://huggingface.co/org/repo/resolve/main/a.gguf"), HUB));
    }

    @Test void theTokenIsNotSentToAnotherHostOrPort() {
        assertFalse(HubHttp.sendsToken(URI.create("https://cdn-lfs.huggingface.co/x"), HUB));
        assertFalse(HubHttp.sendsToken(URI.create("https://huggingface.co:8443/x"), HUB));
    }
}
