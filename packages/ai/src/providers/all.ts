import { type CreateModelsOptions, createModels, type MutableModels, type Provider } from "../models.ts";
import { stdioLocalProvider } from "./stdio-local.ts";

/** All built-in providers, freshly constructed. */
export function builtinProviders(): Provider[] {
	return [stdioLocalProvider()];
}

/** A `Models` collection with every built-in provider registered. */
export function builtinModels(options?: CreateModelsOptions): MutableModels {
	const models = createModels(options);
	for (const provider of builtinProviders()) {
		models.setProvider(provider);
	}
	return models;
}