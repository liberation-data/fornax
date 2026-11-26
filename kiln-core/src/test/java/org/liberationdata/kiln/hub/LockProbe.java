package org.liberationdata.kiln.hub;

import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** Tries the file lock on one path in a JVM of its own and prints "free" or "held". Args: lock path. */
final class LockProbe {
    public static void main(String[] args) throws Exception {
        try (FileChannel ch = FileChannel.open(Path.of(args[0]), StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
            FileLock l = ch.tryLock();
            System.out.println(l != null ? "free" : "held");
        }
    }

    private LockProbe() {}
}
