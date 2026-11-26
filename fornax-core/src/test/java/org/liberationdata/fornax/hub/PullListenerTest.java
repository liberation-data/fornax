package org.liberationdata.fornax.hub;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.ModelSet.Part;
import org.liberationdata.fornax.hub.PullProgress.Phase;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.*;

class PullListenerTest {
    @Test void loggingNamesEachFileAndLeavesOutASizeTheHubDidntGive() {
        Part part = Part.of("gguf", "o/r", "m.gguf");
        ModelSet set = ModelSet.of("m", Kind.CHAT, "m", "t", part);
        List<String> lines = new ArrayList<>();
        Logger jul = Logger.getLogger("org.liberationdata.fornax.hub");
        Handler h = new Handler() {
            @Override public void publish(LogRecord r) { lines.add(r.getMessage()); }
            @Override public void flush() {}
            @Override public void close() {}
        };
        jul.addHandler(h);
        try {
            PullListener log = PullListener.logging();
            log.progress(new PullProgress(set, part, 0, 1, Phase.DOWNLOADING, 0, -1, 0, -1, 0));
            log.progress(new PullProgress(set, part, 0, 1, Phase.DOWNLOADING, 10, -1, 10, -1, 5));   // within ten seconds: quiet
            log.progress(new PullProgress(set, part, 0, 1, Phase.DONE, 10, 10, 10, 10, 5));
        } finally {
            jul.removeHandler(h);
        }
        assertEquals(List.of("Downloading m: o/r m.gguf", "m: m.gguf done"), lines);
    }
}
