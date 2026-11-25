# Kiln: open models as plain Java calls

The Kiln explainer: about two minutes. One voice, cards and real screenshots, over a baroque violin bed rendered with
Muse Sounds. It segues into Endless Country episode 2 as an example of what these open models make. Episode 2 is being
remade in Kiln before this video is published; the video says it was made in Kiln. The script may run ahead of the
code (vocal isolation, transcription): the video is made first, and the features land before it ships. Before
publishing, check every claim against the README.

voice: aura-2-hyperion-en
pace: 1.05
duck: on
outro: stinger/out/stinger-1344x768-dub.mp4 | stinger/out/stinger-1344x768-hold.mp4 | stinger/out/stinger-1344x768.mp4 | stinger/out/stinger-352x192.mp4

## 1. The what

Cards: title 4s, one-line

> Kiln runs open models in your JVM. [pause 0.5] Read invoices, spot defects, flag X-rays, transcribe calls,
> or write a song and render a clip. [pause 0.5] Plain Java calls, on your GPU.

## 2. The why

Cards: options-today, stuck @"you were stuck", stack-before, stack-after @"Kiln calls"

> If you want a local language model or embeddings in Java, there are options. They work, though they're often slower
> than they should be. [pause] But what if you want to flag a fracture on an X-ray for review? Spot defects on a
> production line? Read a stack of invoices? Isolate the vocals from a track? Or any of a plethora of other
> capabilities? [pause] Until now, from the
> JVM, you were stuck. It meant a Python stack, or a tool like ComfyUI behind HTTP: another runtime, another process,
> and a network hop on every call.
>
> Kiln calls the native engines directly. No Python, no server, nothing in between.

## 3. How

Cards: code, models @"Weights load once"

> It uses Java's Foreign Function and Memory API, so the call into the engine is a plain native call. Weights load
> once and stay on the GPU between requests, and when memory runs short, Kiln unloads the models you've used least
> recently. Text streams back token by token, and models can call your tools.

## 4. Show me

Cards: studio-invoice, studio-defects @"spot a defect", studio-review @"an X-ray", studio @"Or chat", numbers @"twice as fast"

(studio is the two-window screenshot; studio-* are screen recordings of Studio's Inspect window, in footage/.)

> In Kiln Studio, read an invoice into JSON, spot a defect on a part, or flag an X-ray for a doctor's review.
> Or chat, paint an image, write a song, or render a clip.
> On our test laptop, Kiln made a one-minute song up to twice as fast as ComfyUI, on the same GPU.

(Backed by kiln-bench, ACE-Step 1.5 turbo, 60 s song: cold 17.3 s against 38.2 s (2.2×), warm 14.7 s against 23.0 s
(1.6×). "Up to" covers both.)

## 5. Your own

Cards: pull, embabel @"Embabel Agent"

> Pull a model from Hugging Face with one command, and try it in Kiln Studio. Kiln is plain Java with no
> dependencies, so it works in any app, and it's a great companion for Embabel Agent: local models, in process.

## 6. An example

Cards: endless-country

> Here's what open models can make, running in Kiln. This is Endless Country, episode two.

Clip: endless-country-ep2-titled (8 s from 2:15: "…what will the dog think?", the tear, the dog's stare; with its own sound; the violin bed ducks under it)

## 7. Close

Cards: end

> Kiln! Open models, inside your JVM! [pause 0.5] Sponsored by Rod Johnson, and created by Jasper Blues! [pause 0.8]
> kiln4j.org!
