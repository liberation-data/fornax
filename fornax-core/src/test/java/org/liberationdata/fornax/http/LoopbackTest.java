package org.liberationdata.fornax.http;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LoopbackTest {
    @Test void loopbackNamesAndAddressesWithTheServersPort() {
        for (String h : new String[]{"localhost:8080", "LOCALHOST:8080", "127.0.0.1:8080", "127.1.2.3:8080", "127.255.255.255:8080", "[::1]:8080"})
            assertTrue(Loopback.host(h, 8080), h);
    }

    @Test void anythingElseIsRefused() {
        for (String h : new String[]{"evil.example:8080", "localhost:8081", "localhost", "127.0.0.1", "128.0.0.1:8080",
                "127.0.0.256:8080", "127.0.0.01:8080", "127.0.0.1.evil.example:8080", "[::2]:8080", "0.0.0.0:8080", ""})
            assertFalse(Loopback.host(h, 8080), h);
        assertFalse(Loopback.host(null, 8080));
    }
}
