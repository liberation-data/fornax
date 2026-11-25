import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

// The docs: src/content/docs/docs/ is served at /docs/, beside the landing page at /.
export const collections = {
    docs: defineCollection({ loader: docsLoader(), schema: docsSchema() })
};
