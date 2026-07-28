import type { ProviderStreams } from "../types.ts";
import { lazyApi } from "./lazy.ts";

export const stdioOpenAIApi = (): ProviderStreams => lazyApi(() => import("./stdio-openai.ts"));
