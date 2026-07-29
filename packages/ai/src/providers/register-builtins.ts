/**
 * Air-gapped builtin provider registration.
 *
 * SECURITY NOTE: This fork intentionally omits network-capable providers
 * (e.g., OpenAI, Anthropic, Brave Search) to enforce the air-gap.
 * Only stdio-local (MLX on Apple Silicon) is registered.
 *
 * If you need additional local providers, add them here explicitly.
 * Do NOT restore auto-discovery of extensions or HTTP-based skills.
 */

import type { Provider } from "../models.ts";
import { stdioLocalProvider } from "./stdio-local.ts";

export interface ProviderRegistry {
	register(provider: Provider): void;
}

export function registerBuiltins(registry: ProviderRegistry): void {
	// Local MLX provider — zero external network surface
	registry.register(stdioLocalProvider());

	// NOTE: If you add other local providers (e.g., llama.cpp via stdio),
	// ensure they also use stdio-local and do not open HTTP ports.
}
