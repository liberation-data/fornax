# Kiln: open models as plain Java calls

The Kiln explainer: about two minutes. One voice, cards and real screenshots, over a baroque violin bed rendered with
Muse Sounds. It segues into Endless Country episode 2 as an example of what these open models make. That episode was
made with ComfyUI; episode 3 will be made with Kiln, and its clip replaces this one then. Every claim here matches the
README; keep it that way.

voice: aura-2-hyperion-en
pace: 1.05

## 1. The what

Cards: title 4s, one-line

> Kiln runs open generative models inside your Java application. Language models, image models, video, even songs.
> You call them like any other Java method, and they run on your GPU.

## 2. The why

Cards: stack-before, stack-after @"Kiln calls"

> Today, using an open model from Java usually means standing up a Python server, or a tool like ComfyUI, and talking
> to it over HTTP. That's another runtime to install, another process to keep alive, and a network hop on every call.
> [pause] Kiln calls the native engines directly: llama.cpp, stable-diffusion.cpp, and the song engines built on the
> same foundation. No Python, no server, nothing in between.

## 3. How

Cards: code, models @"Weights load once"

> It uses Java's Foreign Function and Memory API, so the call into the engine is a plain native call. Weights load
> once and stay on the GPU between requests. Text streams back token by token, and models can call your tools.

## 4. Show me

Cards: single-demo, numbers @"twice as fast"

> In one JVM, a language model writes a country song, ACE-Step sings it, and FLUX paints the cover. On our test
> machine, Kiln generated a one-minute song more than twice as fast as ComfyUI, on the same GPU.

## 5. Your own

Cards: pull, embabel @"Embabel Agent"

> Pull a model from Hugging Face with one command, and try it in Kiln Studio. Kiln is plain Java with no
> dependencies, so it works in any app, and it's a great companion for Embabel Agent: local models, in process.

## 6. An example

Cards: endless-country 4s

> Here's what these open models can make. This is Endless Country, episode two. Episode three will be made with
> Kiln.

Clip: endless-country-ep2 (an excerpt, with its own sound; the violin bed ducks under it)

## 7. Close

Cards: end

> Kiln. Open models, inside your JVM.
