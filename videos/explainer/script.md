# Kiln: open models as plain Java calls

The Kiln explainer: about two minutes. One voice, cards and real screenshots, over a baroque violin bed rendered with
Muse Sounds. It segues into Endless Country episode 2 as an example of what these open models make. That episode was
made with ComfyUI; episode 3 will be made with Kiln, and its clip replaces this one then. The script may run ahead of the
code (vocal isolation, transcription): the video is made first, and the features land before it ships. Before
publishing, check every claim against the README.

voice: aura-2-hyperion-en
pace: 1.05
duck: on

## 1. The what

Cards: title 4s, one-line

> Kiln runs open generative models inside your Java application: language models, images, songs, video, and audio
> tools. You call them like any other Java method, and they run on your GPU.

## 2. The why

Cards: options-today, stuck @"you're stuck", stack-before, stack-after @"Kiln calls"

> If you want a local language model or embeddings in Java, there are options. They work, though they're often slower
> than they should be. [pause] But what if you want to isolate the vocals from a track? Generate product shots for a
> catalogue? Turn a storyboard into footage? Or any of a plethora of other capabilities? [pause] From the JVM, you're
> stuck. It's a Python stack, or a tool like
> ComfyUI behind HTTP: another runtime, another process, and a network hop on every call.
>
> Kiln calls the native engines directly. No Python, no server, nothing in between.

## 3. How

Cards: code, models @"Weights load once"

> It uses Java's Foreign Function and Memory API, so the call into the engine is a plain native call. Weights load
> once and stay on the GPU between requests. Text streams back token by token, and models can call your tools.

## 4. Show me

Cards: studio, numbers @"twice as fast"

(studio is the two-window screenshot.)

> In Kiln Studio, chat with a model and watch the tokens per second, paint an image, write a song, or render a clip.
> On our test laptop, Kiln made a one-minute song up to twice as fast as ComfyUI, on the same GPU.

(Backed by kiln-bench, ACE-Step 1.5 turbo, 60 s song: cold 17.3 s against 38.2 s (2.2×), warm 14.7 s against 23.0 s
(1.6×). "Up to" covers both.)

## 5. Your own

Cards: pull, embabel @"Embabel Agent"

> Pull a model from Hugging Face with one command, and try it in Kiln Studio. Kiln is plain Java with no
> dependencies, so it works in any app, and it's a great companion for Embabel Agent: local models, in process.

## 6. An example

Cards: endless-country

> Here's what these open models can make. This is Endless Country, episode two, made with ComfyUI. Every local step in
> it can now run in Kiln, and episode three will.

Clip: endless-country-ep2 (8 s from 2:15: "…what will the dog think?", the tear, the dog's stare; with its own sound; the violin bed ducks under it)

## 7. Close

Cards: end

> Kiln. Open models, inside your JVM. kiln4j.org.
