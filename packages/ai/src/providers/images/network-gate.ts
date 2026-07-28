/**
 * Air-gap opt-in for image generation.
 *
 * SECURITY NOTE: this fork registers only local providers for LLM inference (see
 * ../register-builtins.ts). The built-in image providers are network-backed, so
 * they stay unregistered unless the operator explicitly opts in.
 */
export function networkImagesAllowed(): boolean {
	const value = process.env.PI_ALLOW_NETWORK_IMAGES;
	if (!value) return false;
	const normalized = value.toLowerCase();
	return normalized === "1" || normalized === "true" || normalized === "yes";
}
